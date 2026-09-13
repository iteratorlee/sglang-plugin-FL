from types import SimpleNamespace
import pytest
import torch

@pytest.mark.parametrize('accepted', [[], [0], [1], [0,1,3]])
@pytest.mark.parametrize('upstream_commits', [False, True])
def test_eagle_calls_kda_commit(accepted,upstream_commits):
    from sglang_fl.models.glm_53_flash.ascend_kda import AscendKDAAttnBackend
    from sglang_fl.models.glm_53_flash.mtp_compat import wrap_glm_verify
    calls=[]
    linear=AscendKDAAttnBackend.__new__(AscendKDAAttnBackend)
    backend=SimpleNamespace(linear_attn_backend=linear,
        update_mamba_state_after_mtp_verify=lambda **kw: calls.append(kw))
    runner=SimpleNamespace(attn_backend=backend,device='cpu',model=object(),
        hybrid_gdn_config=object() if upstream_commits else None)
    worker=SimpleNamespace(target_worker=SimpleNamespace(model_runner=runner))
    result=(object(),SimpleNamespace(num_accepted_drafts_per_req_cpu=accepted),object(),False)
    fn=wrap_glm_verify(lambda *_: result)
    assert fn(worker,object(),object()) is result
    assert len(calls)==int(bool(accepted) and not upstream_commits)
    if calls:
        torch.testing.assert_close(calls[0]['accepted_steps'],torch.tensor(accepted,dtype=torch.int64))
        assert calls[0]['mamba_track_indices'] is None
        assert calls[0]['model'] is runner.model
