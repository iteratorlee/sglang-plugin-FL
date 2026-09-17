"""Regression: NPU MTP must sample the target, not silently use argmax."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("NPU MTP sampling tests require an available NPU", allow_module_level=True)

from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang_fl.models.glm_53_flash.mtp_sampling import sample_target_nodes, sample_logits_npu


def _info(n, device, temperature=1.0, top_k=3, top_p=1.0, min_p=0.0):
    return SimpleNamespace(
        temperatures=torch.full((n, 1), temperature, device=device),
        top_ks=torch.full((n,), top_k, device=device, dtype=torch.int32),
        top_ps=torch.full((n,), top_p, device=device),
        min_ps=torch.full((n,), min_p, device=device),
        need_min_p_sampling=min_p > 0,
    )


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("top_k", [3, TOP_K_ALL])
def test_target_distribution_npu(temperature, top_k):
    import torch_npu
    torch.npu.set_device(0)
    n, steps = 8192, 4
    logits = torch.tensor([1.0, 0.0, -1.0], device="npu").repeat(n * steps, 1)
    original = logits.clone()
    info = _info(n, "npu", temperature=temperature, top_k=top_k)
    top_ks = info.top_ks.clone()
    draws = sample_target_nodes(logits, info, steps,
                                sample_logits_npu)
    empirical = torch.bincount(draws.cpu().long().flatten(), minlength=3).float() / draws.numel()
    expected = torch.softmax(torch.tensor([1.0, 0.0, -1.0]) / temperature, 0)
    torch.testing.assert_close(empirical, expected, atol=0.015, rtol=0)
    torch.testing.assert_close(logits, original, rtol=0, atol=0)
    torch.testing.assert_close(info.top_ks, top_ks, rtol=0, atol=0)


@pytest.mark.parametrize("top_k,top_p,min_p", [(1, 1.0, 0.0), (3, 0.5, 0.0), (3, 1.0, 0.8)])
def test_sampling_filters_npu(top_k, top_p, min_p):
    import torch_npu
    torch.npu.set_device(0)
    logits = torch.tensor([2.0, 0.0, -2.0], device="npu").repeat(512, 1)
    info = _info(128, "npu", top_k=top_k, top_p=top_p, min_p=min_p)
    draws = sample_target_nodes(logits, info, 4,
                                sample_logits_npu)
    assert not draws.any().item()


def test_mixed_greedy_and_stochastic_npu():
    import torch_npu
    torch.npu.set_device(0)
    n, steps = 2048, 4
    info = _info(n, "npu")
    info.top_ks[::2] = 1
    logits = torch.tensor([0.3, 0.2, 0.1], device="npu").repeat(n * steps, 1)
    draws = sample_target_nodes(logits, info, steps,
                                sample_logits_npu)
    assert not draws[::2].any().item()
    assert (draws[1::2] != 0).float().mean().item() > 0.4


def test_sampled_tree_prefix_and_bonus_npu():
    import torch_npu
    from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
    torch.npu.set_device(0)
    n, steps = 256, 4
    logits = torch.tensor([0.3, 0.2, 0.1], device="npu").repeat(n * steps, 1)
    draws = sample_target_nodes(logits, _info(n, "npu"), steps,
                                sample_logits_npu)
    candidates = torch.tensor([0, 1, 0, 2], dtype=torch.int64, device="npu").repeat(n, 1)
    predict, accepted, counts = verify_tree_greedy_func(
        predicts=torch.empty(n * steps + 1, dtype=torch.int32, device="npu"),
        accept_index=torch.full((n, steps), -1, dtype=torch.int32, device="npu"),
        accept_token_num=torch.empty(n, dtype=torch.int32, device="npu"),
        candidates=candidates,
        retrieve_index=torch.arange(n * steps, dtype=torch.int64, device="npu").reshape(n, steps),
        retrieve_next_token=torch.tensor([1, 2, 3, -1], device="npu").repeat(n, 1),
        retrieve_next_sibling=torch.full((n, steps), -1, device="npu", dtype=torch.int64),
        target_predict=draws,
        topk=1,
    )
    predict, accepted, counts, draws = [t.cpu() for t in (predict, accepted, counts, draws)]
    assert counts.min() == 0 and counts.max() == 3
    for row in range(n):
        length = int(counts[row]) + 1
        ids = accepted[row, :length].long()
        assert torch.equal(predict[ids], draws[row, :length])
        assert (accepted[row, length:] == -1).all()
        if length > 1:
            assert torch.equal(draws[row, :length-1].long(), torch.tensor([1, 0, 2])[:length-1])


def test_mismatched_metadata_fails():
    with pytest.raises(ValueError, match="metadata"):
        sample_target_nodes(torch.zeros(3, 3), _info(1, "cpu"), 4, None)


@pytest.mark.parametrize("greedy,architecture,handled", [
    (False, "Glm5NextForConditionalGeneration", True),
    (True, "Glm5NextForConditionalGeneration", False),
    (False, "OtherModel", False),
])
def test_verify_adapter_scope_and_tp_sync_npu(monkeypatch, greedy, architecture, handled):
    import torch_npu
    from sglang.srt.speculative import eagle_info
    import sglang.srt.distributed as distributed
    import sglang.srt.layers.dp_attention as dp
    from sglang_fl.models.glm_53_flash import mtp_sampling as mod
    torch.npu.set_device(0)
    calls = []

    def broadcast(value, src):
        calls.append((value.cpu(), src))
        value.fill_(2)  # The rank-0 result must control prefix matching.

    monkeypatch.setattr(distributed, "get_tp_group", lambda: SimpleNamespace(world_size=16, broadcast=broadcast))
    monkeypatch.setattr(dp, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(eagle_info, "TREE_SPEC_KERNEL_AVAILABLE", False)
    monkeypatch.setattr(eagle_info, "verify_tree_greedy_func", lambda **kw: kw['target_predict'])

    def original(self, batch, output):
        # Simulate a grammar/custom processor applied inside the original
        # verifier: the adapter must see these processed logits, not raw ones.
        output.next_token_logits[:, 0] = -torch.inf
        return eagle_info.verify_tree_greedy_func(
            target_predict=output.next_token_logits.argmax(-1).reshape(-1, 4))

    monkeypatch.setattr(eagle_info.EagleVerifyInput, "verify", original)
    monkeypatch.setattr(eagle_info.logger, "filters", list(eagle_info.logger.filters))
    monkeypatch.setattr(mod, "_PATCHED", False)
    mod.patch_mtp_sampling()
    class Info(SimpleNamespace):
        def __len__(self): return self.temperatures.shape[0]
    info = Info(**vars(_info(128, 'npu')), is_all_greedy=greedy)
    batch = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
                            sampling_info=info, forward_mode=SimpleNamespace(is_idle=lambda:False))
    verify = SimpleNamespace(topk=1, draft_token_num=4, retrieve_index=torch.zeros(128,4))
    output = SimpleNamespace(next_token_logits=torch.zeros(512,3,device='npu'))
    result = eagle_info.EagleVerifyInput.verify(verify, batch, output)
    if handled:
        assert len(calls) == 1 and calls[0][1] == 0
        assert not (calls[0][0] == 0).any()  # Mask applied before sampling.
        assert (calls[0][0] == 2).float().mean() > .3
        assert (result == 2).all()  # Broadcast applied before tree matching.
    else:
        assert not calls
        assert (result == 1).all()
    assert mod._CONTEXT.get() is None
