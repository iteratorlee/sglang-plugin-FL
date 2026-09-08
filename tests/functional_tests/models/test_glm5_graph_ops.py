from types import SimpleNamespace

import pytest
import torch


def test_glm5_registers_v0511_hybrid_kda_backend():
    """The external model must select HybridLinearAttnBackend, not plain MLA."""

    from sglang.srt.configs.linear_attn_model_registry import (
        get_linear_attn_spec_by_arch,
    )
    from sglang_fl.models.register import register_glm5_next

    register_glm5_next()
    spec = get_linear_attn_spec_by_arch("Glm5NextForConditionalGeneration")
    assert spec is not None
    assert spec.backend_class_name.endswith("AscendKDAAttnBackend")
    assert spec.unwrap_text_config


def test_glm5_config_bootstraps_v0511_mla_and_nsa_markers():
    from sglang_fl.models.glm5_next_config import Glm5NextConfig

    config = Glm5NextConfig(
        text_config={
            "index_head_dim": 128,
            "index_topk": 2048,
            "index_n_heads": 64,
            "indexer_rope_interleave": True,
            "qk_rope_head_dim": 0,
        }
    )
    assert config.architectures == [
        "Glm5NextForConditionalGeneration",
        "GlmMoeDsaForCausalLM",
    ]
    assert config.text_config.architectures == ["GlmMoeDsaForCausalLM"]
    assert config.text_config.qk_rope_head_dim == 0
    assert config.text_config.indexer_rope_interleave


def test_glm5_kpool_builder_matches_checkpoint_contract(monkeypatch):
    import sglang_fl.models.glm5_next as model_module

    captured = {}

    class FakeKPool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(model_module, "IndexerKPool", FakeKPool)
    config = SimpleNamespace(
        index_n_heads=32,
        index_head_dim=128,
        qk_rope_head_dim=0,
        index_topk=2048,
        q_lora_rank=1536,
        indexer_rope_interleave=True,
        index_kpool=4,
    )
    result = model_module._build_glm_kpool_indexer(
        config,
        hidden_size=4096,
        layer_id=7,
        max_position_embeddings=1048576,
        rope_theta=10000.0,
        rope_scaling=None,
        prefix="model.layers.7.self_attn",
        quant_config=None,
        alt_stream=None,
    )

    assert isinstance(result, FakeKPool)
    assert captured["index_n_heads"] == 32
    assert captured["index_head_dim"] == 128
    assert captured["rope_head_dim"] == 0
    assert captured["index_topk"] == 2048
    assert captured["q_lora_rank"] == 1536
    assert captured["is_neox_style"] is False
    assert captured["prefix"] == "model.layers.7.self_attn.indexer"
    assert captured["config"] is config


def test_glm5_kpool_head_gate_is_persistent_fp32_with_stable_topk(monkeypatch):
    import torch.nn.functional as F

    import sglang_fl.models.kpool_indexer as kpool_module

    class FakeReplicatedLinear(torch.nn.Module):
        def __init__(
            self,
            input_size,
            output_size,
            bias=False,
            params_dtype=None,
            prefix="",
        ):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.zeros(output_size, input_size, dtype=params_dtype)
            )
            self.weight.weight_loader = lambda param, loaded: param.data.copy_(loaded)

        def forward(self, x):
            return F.linear(x, self.weight), None

    def fake_indexer_init(
        self,
        hidden_size,
        index_n_heads,
        index_head_dim,
        **_kwargs,
    ):
        torch.nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.softmax_scale = index_head_dim**-0.5

    monkeypatch.setattr(kpool_module.Indexer, "__init__", fake_indexer_init)
    monkeypatch.setattr(kpool_module, "ReplicatedLinear", FakeReplicatedLinear)
    monkeypatch.setattr(kpool_module, "is_npu", lambda: False)
    monkeypatch.setattr(
        kpool_module,
        "get_global_server_args",
        lambda: SimpleNamespace(max_running_requests=1, device="cpu"),
    )
    config = SimpleNamespace(
        index_kpool=4,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
    )
    indexer = kpool_module.IndexerKPool(
        hidden_size=4096,
        index_n_heads=32,
        index_head_dim=128,
        rope_head_dim=0,
        index_topk=2048,
        q_lora_rank=1536,
        max_position_embeddings=1048576,
        rope_theta=10000.0,
        layer_id=3,
        scale_fmt="ue8m0",
        config=config,
    )

    assert indexer.weights_proj.weight.dtype == torch.float32
    assert callable(indexer.weights_proj.weight.weight_loader)
    with torch.no_grad():
        indexer.weights_proj.weight.zero_()
        indexer.weights_proj.weight[:16, 0] = torch.tensor(
            [1.0 + i / 128 for i in range(16)], dtype=torch.float32
        )
    hidden = torch.zeros(1, 4096, dtype=torch.bfloat16)
    hidden[:, 0] = 1
    gates = indexer._project_head_weights(hidden)
    reference = F.linear(hidden.float(), indexer.weights_proj.weight)
    reference *= indexer.softmax_scale * indexer.n_heads**-0.5
    assert gates.dtype == torch.float32
    torch.testing.assert_close(gates, reference, rtol=0, atol=0)
    assert torch.equal(
        torch.topk(gates, 8, dim=-1).indices,
        torch.arange(15, 7, -1).view(1, 8),
    )


def test_glm5_lightning_indexer_cast_is_only_at_operator_boundary():
    from sglang_fl.models.kpool_indexer import _ascend_lightning_weights

    fp32_weights = torch.tensor([[1.25, -0.75]], dtype=torch.float32)
    query = torch.empty(1, 2, 128, dtype=torch.bfloat16)
    adapted = _ascend_lightning_weights(fp32_weights, query)
    assert fp32_weights.dtype == torch.float32
    assert adapted.dtype == torch.bfloat16
    torch.testing.assert_close(adapted.float(), fp32_weights, rtol=0, atol=0)

    with pytest.raises(ValueError, match="must produce FP32"):
        _ascend_lightning_weights(fp32_weights.bfloat16(), query)


def test_glm5_kpool_indices_match_ascend_sparse_attention_abi():
    from sglang_fl.models.kpool_indexer import _as_ascend_sparse_indices

    logical = torch.arange(2 * 11, dtype=torch.int32).view(2, 11)
    physical = _as_ascend_sparse_indices(logical)
    assert physical.shape == (2, 1, 11)
    assert physical.dtype == torch.int32
    assert physical.data_ptr() == logical.data_ptr()
    torch.testing.assert_close(physical[:, 0], logical, rtol=0, atol=0)

    with pytest.raises(ValueError, match="logical KPool indices"):
        _as_ascend_sparse_indices(physical)
    with pytest.raises(ValueError, match="requires int32"):
        _as_ascend_sparse_indices(logical.to(torch.int64))


def test_glm5_moe_scopes_bf16_deepep_dispatch_on_npu(monkeypatch):
    import sglang_fl.models.glm5_next as model_module
    from sglang.srt.environ import envs

    seen = []

    def fake_parent_forward(_self, hidden_states, forward_batch):
        seen.append(envs.SGLANG_DEEPEP_BF16_DISPATCH.get())
        return hidden_states, forward_batch

    monkeypatch.setattr(model_module, "is_npu", lambda: True)
    monkeypatch.setattr(model_module.DeepseekV2MoE, "forward", fake_parent_forward)
    moe = model_module.Glm5NextMoE.__new__(model_module.Glm5NextMoE)
    hidden = torch.empty(16, 4096, dtype=torch.bfloat16)
    batch = object()
    with envs.SGLANG_DEEPEP_BF16_DISPATCH.override(False):
        result = model_module.Glm5NextMoE.forward(moe, hidden, batch)
        assert not envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
    assert seen == [True]
    assert result[0] is hidden
    assert result[1] is batch


def test_glm5_moe_topk_uses_loaded_gate_correction_bias():
    from sglang_fl.models.glm5_next import _rebind_glm_moe_correction_bias

    stale = torch.nn.Parameter(torch.empty(288, dtype=torch.float32))
    loaded = torch.nn.Parameter(torch.arange(288, dtype=torch.float32))
    moe = SimpleNamespace(
        gate=SimpleNamespace(e_score_correction_bias=loaded),
        topk=SimpleNamespace(topk_config=SimpleNamespace(correction_bias=stale)),
    )
    assert moe.topk.topk_config.correction_bias is stale
    _rebind_glm_moe_correction_bias(moe)
    assert moe.topk.topk_config.correction_bias is loaded
    torch.testing.assert_close(
        moe.topk.topk_config.correction_bias,
        torch.arange(288, dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_glm5_decode_uses_plugin_bounded_varlen_kernel(monkeypatch):
    import sglang_fl.models.kda_recurrent_npu as recurrent_module
    from sglang_fl.models.ascend_kda import _ascend_kda_decode

    calls = []

    def fake_recurrent(**kwargs):
        calls.append(kwargs)
        return kwargs["v"]

    monkeypatch.setattr(
        recurrent_module, "glm_kda_varlen_recurrent_npu", fake_recurrent
    )
    q = torch.ones(1, 2, 4, 8)
    result = _ascend_kda_decode(
        q=q,
        k=q.clone(),
        v=q.clone(),
        a=torch.ones(2, 32),
        b=torch.ones(1, 2, 4),
        A_log=torch.zeros(1, 1, 4, 1),
        dt_bias=torch.zeros(4, 8),
        ssm_states=torch.zeros(5, 4, 8, 8),
        cache_indices=torch.tensor([1, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        lower_bound=-5.0,
    )
    assert result.shape == q.shape
    assert len(calls) == 1
    assert calls[0]["a"].shape == (1, 2, 4, 8)
    assert calls[0]["b"].shape == (1, 2, 4)
    assert calls[0]["lower_bound"] == -5.0


def test_glm5_kda_prefill_uses_npu_causal_conv(monkeypatch):
    import sgl_kernel_npu.mamba.causal_conv1d as npu_conv

    from sglang_fl.models.ascend_kda import _ascend_kda_prefill_conv

    seen = {}

    def fake_npu_conv(x, weight, bias, **kwargs):
        seen.update(kwargs)
        assert weight.shape == (4, 4)
        assert bias is None
        return x + 1

    monkeypatch.setattr(npu_conv, "causal_conv1d_fn_npu", fake_npu_conv)
    x = torch.zeros(4, 3)
    state = torch.zeros(2, 4, 3)
    cache_indices = torch.tensor([1], dtype=torch.int32)
    starts = torch.tensor([0, 3], dtype=torch.int32)
    has_initial = torch.tensor([False])
    result = _ascend_kda_prefill_conv(
        x,
        torch.zeros(4, 4),
        None,
        state=state,
        has_initial_state=has_initial,
        cache_indices=cache_indices,
        query_start_loc=starts,
    )

    assert result.shape == (3, 4)
    assert torch.equal(result, torch.ones(3, 4))
    assert seen["activation"] == "silu"
    assert seen["conv_states"] is state
    assert seen["has_initial_state"] is has_initial
    assert seen["cache_indices"] is cache_indices
    assert seen["query_start_loc"] is starts


def test_glm5_kda_gate_layout_and_activation_match_v0511_contract():
    from sglang_fl.models.glm5_next import _prepare_glm_kda_gates

    raw_gate = torch.arange(22 * 512, dtype=torch.bfloat16).view(22, 512) / 512
    raw_beta = torch.linspace(-3, 3, 22 * 4, dtype=torch.bfloat16).view(22, 4)
    prefill_gate, prefill_beta = _prepare_glm_kda_gates(
        raw_gate, raw_beta, head_dim=128, is_decode=False
    )
    assert prefill_gate.shape == (1, 22, 4, 128)
    assert prefill_gate.dtype == torch.bfloat16
    assert prefill_beta.shape == (1, 22, 4)
    assert prefill_beta.dtype == torch.bfloat16
    torch.testing.assert_close(prefill_gate[0].flatten(-2), raw_gate, rtol=0, atol=0)
    torch.testing.assert_close(prefill_beta[0], raw_beta, rtol=0, atol=0)

    decode_gate, decode_beta = _prepare_glm_kda_gates(
        raw_gate[:16], raw_beta[:16], head_dim=128, is_decode=True
    )
    assert decode_gate is raw_gate[:16] or torch.equal(decode_gate, raw_gate[:16])
    assert decode_gate.shape == (16, 512)
    assert decode_beta.shape == (1, 16, 4)
    assert decode_beta.dtype == torch.bfloat16
    torch.testing.assert_close(decode_beta[0], raw_beta[:16], rtol=0, atol=0)


def test_glm5_kda_prefill_recurrent_uses_active_tokens_and_zero_pads(monkeypatch):
    import sglang_fl.models.ascend_kda as kda_module
    import sglang_fl.models.kda_recurrent_npu as recurrent_module

    calls = []

    def fake_recurrent(**kwargs):
        calls.append(kwargs)
        return kwargs["v"] + 100

    monkeypatch.setattr(
        recurrent_module, "glm_kda_varlen_recurrent_npu", fake_recurrent
    )
    values = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)
    q = values.expand(1, 4, 2, 3).clone()
    result = kda_module._ascend_kda_prefill_recurrent(
        q=q,
        k=q,
        v=q,
        a=q,
        b=torch.zeros(1, 4, 2),
        A_log=torch.zeros(1, 1, 2, 1),
        dt_bias=torch.zeros(6),
        ssm_states=torch.zeros(8, 2, 3, 3),
        cache_indices=torch.tensor([5, 7], dtype=torch.int32),
        seq_lens_cpu=[1, 3],
    )
    assert len(calls) == 1
    assert calls[0]["cu_seqlens"].tolist() == [0, 1, 4]
    assert calls[0]["initial_state_indices"].tolist() == [5, 7]
    assert calls[0]["lower_bound"] is None
    assert result.shape == (1, 4, 2, 3)
    torch.testing.assert_close(result, q + 100, rtol=0, atol=0)

    with pytest.raises(ValueError, match="token count mismatch"):
        kda_module._ascend_kda_prefill_recurrent(
            q=q,
            k=q,
            v=q,
            a=q,
            b=torch.zeros(1, 4, 2),
            A_log=torch.zeros(1, 1, 2, 1),
            dt_bias=torch.zeros(6),
            ssm_states=torch.zeros(8, 2, 3, 3),
            cache_indices=torch.tensor([5, 7], dtype=torch.int32),
            seq_lens_cpu=[2, 3],
        )

    padded_q = torch.zeros(1, 16, 2, 3)
    padded_q[:, :12] = torch.arange(12).view(1, 12, 1, 1)
    calls.clear()
    padded = kda_module._ascend_kda_prefill_recurrent(
        q=padded_q,
        k=padded_q,
        v=padded_q,
        a=padded_q,
        b=torch.zeros(1, 16, 2),
        A_log=torch.zeros(1, 1, 2, 1),
        dt_bias=torch.zeros(6),
        ssm_states=torch.zeros(8, 2, 3, 3),
        cache_indices=torch.tensor([5], dtype=torch.int32),
        seq_lens_cpu=[12],
    )
    assert len(calls) == 1
    assert calls[0]["q"].shape[1] == 12
    assert calls[0]["k"].shape[1] == 12
    assert calls[0]["v"].shape[1] == 12
    assert calls[0]["a"].shape[1] == 12
    assert calls[0]["b"].shape[1] == 12
    assert calls[0]["cu_seqlens"].tolist() == [0, 12]
    assert padded.shape == padded_q.shape
    torch.testing.assert_close(padded[:, :12], padded_q[:, :12] + 100, rtol=0, atol=0)
    assert torch.count_nonzero(padded[:, 12:]).item() == 0


def test_glm5_pool_index_dimension_context_is_scoped(monkeypatch):
    import sglang_fl.models.register as register_module
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
        ModelRunnerKVCacheMixin,
    )

    seen = []

    def original(self):
        seen.append(register_module._GLM_POOL_INDEX_HEAD_DIM.get())

    monkeypatch.setattr(
        register_module.patch_glm5_pool_context,
        "_original_init_pools",
        original,
    )
    glm_runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Glm5NextForConditionalGeneration"]
            ),
            hf_text_config=SimpleNamespace(index_head_dim=128),
        )
    )
    other_runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["OtherForCausalLM"]),
            hf_text_config=SimpleNamespace(index_head_dim=999),
        )
    )
    ModelRunnerKVCacheMixin._init_pools(glm_runner)
    assert register_module._GLM_POOL_INDEX_HEAD_DIM.get() is None
    ModelRunnerKVCacheMixin._init_pools(other_runner)
    assert register_module._GLM_POOL_INDEX_HEAD_DIM.get() is None
    assert seen == [128, None]


def test_glm5_pool_selection_uses_hybrid_not_packed_nsa():
    """GLM is DSA-capable but must enter v0.5.11's hybrid pool branch."""

    import sglang.srt.models.deepseek_v2 as dsv2
    from sglang.srt.model_executor import model_runner_kv_cache_mixin
    from sglang_fl.models.register import (
        patch_deepseek_dsa_compat,
        patch_glm5_pool_context,
    )

    patch_deepseek_dsa_compat()
    patch_glm5_pool_context()
    glm = SimpleNamespace(
        architectures=["Glm5NextForConditionalGeneration"],
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=64,
    )
    assert dsv2.is_deepseek_nsa(glm)
    assert not model_runner_kv_cache_mixin.is_deepseek_nsa(glm)

    # The pool-only predicate must delegate every unrelated architecture to
    # the ordinary DSA detector instead of changing process-wide semantics.
    other = SimpleNamespace(architectures=["OtherForCausalLM"])
    assert model_runner_kv_cache_mixin.is_deepseek_nsa(other) == dsv2.is_deepseek_nsa(
        other
    )


def test_glm5_pool_runtime_check_accepts_tensor_index_cache(monkeypatch):
    """The Ascend MLA index cache is one tensor, not a Python list."""

    import sglang_fl.models.register as register_module
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
        ModelRunnerKVCacheMixin,
    )

    index_cache = torch.empty(2, 8, 1, 128, dtype=torch.bfloat16)
    full_pool = SimpleNamespace(
        index_k_buffer=index_cache,
        get_index_k_buffer=lambda layer_id: index_cache[layer_id],
    )

    def original(self):
        self.token_to_kv_pool = SimpleNamespace(full_kv_pool=full_pool)

    monkeypatch.setattr(
        register_module.patch_glm5_pool_context,
        "_original_init_pools",
        original,
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Glm5NextForConditionalGeneration"]
            ),
            hf_text_config=SimpleNamespace(index_head_dim=128),
        )
    )
    ModelRunnerKVCacheMixin._init_pools(runner)


def test_glm5_ascend_kda_rejects_unverified_modes(monkeypatch):
    from sglang_fl.models.ascend_kda import AscendKDAAttnBackend
    import sglang_fl.models.ascend_kda as kda_module

    class _SpecAlgorithm:
        def __init__(self, is_none):
            self._is_none = is_none

        def is_none(self):
            return self._is_none

    with pytest.raises(NotImplementedError, match="speculative"):
        AscendKDAAttnBackend(
            SimpleNamespace(
                spec_algorithm=_SpecAlgorithm(False),
                server_args=SimpleNamespace(
                    disable_radix_cache=True, chunked_prefill_size=-1
                ),
            )
        )

    with pytest.raises(ValueError, match="disable-radix-cache"):
        AscendKDAAttnBackend(
            SimpleNamespace(
                spec_algorithm=_SpecAlgorithm(True),
                server_args=SimpleNamespace(
                    disable_radix_cache=False, chunked_prefill_size=4096
                ),
            )
        )

    initialized = []
    monkeypatch.setattr(
        kda_module.KDAAttnBackend,
        "__init__",
        lambda self, runner: initialized.append(runner),
    )
    with pytest.raises(ValueError, match="multiple of 128"):
        AscendKDAAttnBackend(
            SimpleNamespace(
                spec_algorithm=_SpecAlgorithm(True),
                server_args=SimpleNamespace(
                    disable_radix_cache=True, chunked_prefill_size=96
                ),
            )
        )
    chunked_runner = SimpleNamespace(
        spec_algorithm=_SpecAlgorithm(True),
        server_args=SimpleNamespace(
            disable_radix_cache=True, chunked_prefill_size=4096
        ),
    )
    AscendKDAAttnBackend(chunked_runner)
    assert initialized == [chunked_runner]


def test_glm5_ascend_kda_conv_cache_layout_is_zero_copy():
    from sglang_fl.models.ascend_kda import _physical_npu_conv_state

    physical = torch.arange(2 * 12 * 3).view(2, 12, 3)
    validated = _physical_npu_conv_state(physical, kernel_width=4)
    assert validated.shape == (2, 12, 3)
    assert validated.data_ptr() == physical.data_ptr()

    with pytest.raises(ValueError, match="invalid Ascend KDA conv cache layout"):
        _physical_npu_conv_state(physical.transpose(-1, -2), kernel_width=4)


def test_glm5_ascend_kda_decode_rejects_invalid_static_contract():
    from sglang_fl.models.ascend_kda import _ascend_kda_decode

    inputs = {
        "q": torch.empty(1, 2, 4, 8),
        "k": torch.empty(1, 2, 4, 8),
        "v": torch.empty(1, 2, 4, 8),
        "a": torch.empty(2, 4, 8),
        "b": torch.empty(2, 4),
        "A_log": torch.empty(4),
        "dt_bias": torch.empty(4, 8),
        "ssm_states": torch.empty(5, 4, 8, 8),
        "cache_indices": torch.empty(2, dtype=torch.int32),
        "query_start_loc": torch.empty(2, dtype=torch.int32),
    }
    with pytest.raises(ValueError, match="one varlen sequence per token"):
        _ascend_kda_decode(**inputs)

    inputs["query_start_loc"] = torch.empty(3, dtype=torch.int32)
    for name in ("q", "k", "v"):
        inputs[name] = torch.empty(1, 2, 3, 8)
    with pytest.raises(ValueError, match="even local head count"):
        _ascend_kda_decode(**inputs)


def test_glm5_kda_prefill_uses_one_varlen_kernel_and_preserves_padding(
    monkeypatch,
):
    from sglang_fl.models.ascend_kda import _ascend_kda_prefill_recurrent
    import sglang_fl.models.kda_recurrent_npu as recurrent_module

    calls = []

    def fake_recurrent(**kwargs):
        calls.append(kwargs)
        return kwargs["v"] + 7

    monkeypatch.setattr(
        recurrent_module, "glm_kda_varlen_recurrent_npu", fake_recurrent
    )
    q = torch.ones(1, 16, 4, 8)
    inputs = {
        "q": q,
        "k": q.clone(),
        "v": q.clone(),
        "a": q.clone(),
        "b": torch.ones(1, 16, 4),
        "A_log": torch.zeros(1, 1, 4, 1),
        "dt_bias": torch.zeros(4, 8),
        "ssm_states": torch.zeros(4, 4, 8, 8),
        "cache_indices": torch.tensor([1, 2], dtype=torch.int32),
        "seq_lens_cpu": [3, 5],
        "lower_bound": -5.0,
    }
    output = _ascend_kda_prefill_recurrent(**inputs)

    assert len(calls) == 1
    assert calls[0]["q"].shape == (1, 8, 4, 8)
    assert calls[0]["a"].shape == (1, 8, 4, 8)
    assert calls[0]["b"].shape == (1, 8, 4)
    assert calls[0]["cu_seqlens"].tolist() == [0, 3, 8]
    assert calls[0]["lower_bound"] == -5.0
    assert output.shape == (1, 16, 4, 8)
    torch.testing.assert_close(output[:, :8], torch.full_like(output[:, :8], 8))
    torch.testing.assert_close(output[:, 8:], torch.zeros_like(output[:, 8:]))


def test_glm5_kpool_chunked_prefill_matches_one_shot(monkeypatch):
    import sglang_fl.models.kpool_indexer as kpool_module

    def eager_scatter(dst, rows, src):
        dst.reshape(-1, dst.shape[-1])[rows.long()] = src
        return dst

    monkeypatch.setattr(kpool_module, "scatter_rows_", eager_scatter)

    def make_indexer():
        indexer = kpool_module.IndexerKPool.__new__(kpool_module.IndexerKPool)
        torch.nn.Module.__init__(indexer)
        indexer.index_kpool = 4
        indexer.head_dim = 2
        indexer.index_kpool_compress_ape = torch.nn.Parameter(torch.zeros(4, 2))
        indexer.register_buffer("_kpool_tail_k", torch.zeros(3, 4, 2))
        indexer.register_buffer("_kpool_tail_score", torch.zeros(3, 4, 2))
        return indexer

    keys = torch.arange(24, dtype=torch.float32).reshape(12, 2).bfloat16()
    scores = torch.linspace(-1, 1, 24).reshape(12, 2)
    block_tables = torch.arange(16).reshape(1, 16)

    def make_batch(cache, q_len, seq_len):
        return SimpleNamespace(
            batch_size=1,
            extend_seq_lens_cpu=[q_len],
            seq_lens_cpu=[seq_len],
            req_pool_indices=torch.tensor([1]),
            token_to_kv_pool=SimpleNamespace(
                get_index_k_buffer=lambda _layer_id: cache
            ),
        )

    one_shot = make_indexer()
    one_shot_cache = torch.zeros(16, 64, 1, 2, dtype=torch.bfloat16)
    expected = one_shot._store_prefill_pools(
        keys,
        scores,
        make_batch(one_shot_cache, 12, 12),
        block_tables,
        layer_id=3,
    )[0]

    chunked = make_indexer()
    chunked_cache = torch.zeros_like(one_shot_cache)
    first = chunked._store_prefill_pools(
        keys[:5],
        scores[:5],
        make_batch(chunked_cache, 5, 5),
        block_tables,
        layer_id=3,
    )[0]
    assert first.shape == (1, 2)
    actual = chunked._store_prefill_pools(
        keys[5:],
        scores[5:],
        make_batch(chunked_cache, 7, 12),
        block_tables,
        layer_id=3,
    )[0]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(chunked_cache, one_shot_cache, rtol=0, atol=0)
    assert torch.count_nonzero(chunked._kpool_tail_k[1]) == 0


def test_glm5_kpool_prefill_topk_is_chunk_invariant(monkeypatch):
    import sglang_fl.models.kpool_indexer as kpool_module

    indexer = kpool_module.IndexerKPool.__new__(kpool_module.IndexerKPool)
    torch.nn.Module.__init__(indexer)
    indexer.index_kpool = 4
    indexer.index_topk = 64

    torch.manual_seed(20260905)
    q = torch.randn(256, 2, 8)
    weights = torch.randn(256, 2)
    history = torch.randn(64, 8)
    positions = torch.arange(256)

    def make_batch(q_len, seq_len):
        return SimpleNamespace(
            batch_size=1,
            extend_seq_lens_cpu=[q_len],
            seq_lens_cpu=[seq_len],
        )

    real_einsum = torch.einsum
    key_lengths = []

    def record_einsum(equation, queries, keys):
        key_lengths.append(keys.shape[0])
        return real_einsum(equation, queries, keys)

    monkeypatch.setattr(kpool_module.torch, "einsum", record_einsum)
    one_shot = indexer._prefill_topk(
        q,
        weights,
        [history],
        make_batch(256, 256),
        positions,
    )
    assert key_lengths == [32, 64]

    key_lengths.clear()
    first = indexer._prefill_topk(
        q[:128],
        weights[:128],
        [history[:32]],
        make_batch(128, 128),
        positions[:128],
    )
    second = indexer._prefill_topk(
        q[128:],
        weights[128:],
        [history],
        make_batch(128, 256),
        positions[128:],
    )
    assert key_lengths == [32, 64]
    assert torch.equal(one_shot, torch.cat((first, second), dim=0))


@pytest.mark.gpu
def test_glm5_kda_multitoken_varlen_matches_fp32_reference(device):
    if device.type != "npu":
        pytest.skip("GLM-5.3 fused long-prefill KDA is Ascend-specific")

    from sglang_fl.models.kda_recurrent_npu import (
        glm_kda_varlen_recurrent_npu,
    )

    torch.manual_seed(20260905)
    lengths = [5, 12]
    total = sum(lengths)
    num_heads = 4
    head_dim = 128

    def sample(*shape):
        return (torch.randn(*shape, device=device) * 0.2).bfloat16()

    q = sample(1, total, num_heads, head_dim)
    k = sample(1, total, num_heads, head_dim)
    v = sample(1, total, num_heads, head_dim)
    a = sample(1, total, num_heads, head_dim)
    b = sample(1, total, num_heads)
    A_log = torch.linspace(-1.3, 0.4, num_heads, device=device).reshape(
        1, 1, num_heads, 1
    )
    dt_bias = torch.linspace(-0.4, 0.6, num_heads * head_dim, device=device).reshape(
        num_heads, head_dim
    )
    state = (
        torch.randn(
            3, num_heads, head_dim, head_dim, dtype=torch.float32, device=device
        )
        * 0.2
    )
    q_ref, k_ref, v_ref = q.cpu(), k.cpu(), v.cpu()
    a_ref, b_ref = a.cpu(), b.cpu()
    A_log_ref, dt_bias_ref = A_log.cpu(), dt_bias.cpu()
    expected_state = state.cpu().clone()
    indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
    starts = torch.tensor([0, 5, 17], dtype=torch.int32, device=device)

    actual = glm_kda_varlen_recurrent_npu(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state,
        initial_state_indices=indices,
        cu_seqlens=starts,
        lower_bound=-5.0,
    )

    expected = torch.zeros_like(v_ref)
    scale = head_dim**-0.5
    offset = 0
    for sequence, length in enumerate(lengths):
        hidden = expected_state[int(indices[sequence].item())].float()
        for token in range(offset, offset + length):
            q_token = q_ref[0, token].float()
            k_token = k_ref[0, token].float()
            value = v_ref[0, token].float()
            q_token /= (q_token.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            k_token /= (k_token.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            gate = -5.0 * torch.sigmoid(
                A_log_ref.float().reshape(-1, 1).exp()
                * (a_ref[0, token].float() + dt_bias_ref.float())
            )
            beta = torch.sigmoid(b_ref[0, token].float())
            hidden *= gate.exp().unsqueeze(-1)
            residual = value - (hidden * k_token.unsqueeze(-1)).sum(-2)
            residual *= beta.unsqueeze(-1)
            hidden += k_token.unsqueeze(-1) * residual.unsqueeze(-2)
            expected[0, token] = (hidden * (q_token * scale).unsqueeze(-1)).sum(-2)
        expected_state[int(indices[sequence].item())] = hidden
        offset += length

    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0.002)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0.005)
    assert torch.equal(state[0].cpu(), expected_state[0])


@pytest.mark.gpu
def test_glm5_rms_norm_gated_is_chunk_invariant_and_graph_safe(device):
    if device.type != "npu":
        pytest.skip("GLM-5.3 gated RMSNorm row kernel is Ascend-specific")

    from sglang_fl.models.rms_norm_gated_npu import glm_rms_norm_gated_npu

    torch.manual_seed(20260905)
    eps = 1e-5
    x = (torch.randn(1, 257, 4, 128, device=device) * 0.2).bfloat16()
    gate = (torch.randn(257, 4, 128, device=device) * 1.3).bfloat16()
    weight = (torch.randn(128, device=device) * 0.2).bfloat16()

    def reference(value, gate_value):
        value_fp32 = value.float()
        normalized = value_fp32 * torch.rsqrt(
            value_fp32.square().mean(-1, keepdim=True) + eps
        )
        return (normalized * weight.float() * torch.sigmoid(gate_value.float())).to(
            value.dtype
        )

    one_shot = glm_rms_norm_gated_npu(x, gate, weight, eps)
    chunked = torch.cat(
        (
            glm_rms_norm_gated_npu(x[:, :128], gate[:128], weight, eps),
            glm_rms_norm_gated_npu(x[:, 128:], gate[128:], weight, eps),
        ),
        dim=1,
    )
    expected = reference(x, gate)
    torch.npu.synchronize()
    torch.testing.assert_close(one_shot, expected, rtol=0, atol=0.002)
    torch.testing.assert_close(chunked, expected, rtol=0, atol=0.002)
    assert torch.equal(one_shot, chunked)
    assert one_shot.data_ptr() != x.data_ptr()

    static_x = torch.zeros(1, 16, 4, 128, dtype=torch.bfloat16, device=device)
    static_gate = torch.zeros(16, 4, 128, dtype=torch.bfloat16, device=device)
    glm_rms_norm_gated_npu(static_x, static_gate, weight, eps)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = glm_rms_norm_gated_npu(static_x, static_gate, weight, eps)

    for scale in (0.5, 1.5):
        changed_x = (torch.randn_like(static_x.float()) * scale).bfloat16()
        changed_gate = (torch.randn_like(static_gate.float()) * scale).bfloat16()
        static_x.copy_(changed_x)
        static_gate.copy_(changed_gate)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            graph_output,
            reference(changed_x, changed_gate),
            rtol=0,
            atol=0.002,
        )


def test_glm5_v0511_attention_context_cleanup():
    from sglang_fl.models.compat import clear_attn_inputs, set_attn_hidden_states_local

    attn_inputs = SimpleNamespace(hidden_states_local=None)
    old_context = SimpleNamespace(attn_inputs_=attn_inputs)
    hidden_states = torch.empty(2, 8)
    set_attn_hidden_states_local(old_context, hidden_states)
    assert attn_inputs.hidden_states_local is hidden_states
    clear_attn_inputs(old_context)
    assert old_context.attn_inputs_ is None


def test_glm5_v0511_no_rope_dsa_prepare_contract():
    from sglang_fl.models.ascend_dsa import forward_glm5_dsa_prepare_npu

    tokens = 3
    fused = torch.arange(tokens * 7, dtype=torch.float32).view(tokens, 7)
    expected_topk = torch.tensor([[0, 1]])
    index_calls = []

    def indexer(*args):
        index_calls.append(args)
        return expected_topk

    m = SimpleNamespace(
        rotary_emb=None,
        qk_rope_head_dim=0,
        q_lora_rank=4,
        kv_lora_rank=3,
        num_local_heads=2,
        qk_nope_head_dim=2,
        fused_qkv_a_proj_with_mqa=lambda _hidden: (fused,),
        q_a_layernorm=lambda x: x,
        q_b_proj=lambda x: (x,),
        kv_a_layernorm=lambda x: x,
        w_kc=torch.ones(2, 2, 3),
        skip_topk=False,
        indexer=indexer,
        layer_id=3,
    )
    positions = torch.arange(tokens)
    hidden = torch.empty(tokens, 8)
    batch = object()
    scatter = object()
    out = forward_glm5_dsa_prepare_npu(m, positions, hidden, batch, None, scatter)
    q_pe, k_pe, q_nope_out, k_nope, topk, *_ = out
    assert q_pe.shape == (tokens, 2, 0)
    assert k_pe.shape == (tokens, 1, 0)
    assert q_nope_out.shape == (tokens, 2, 3)
    assert k_nope.shape == (tokens, 1, 3)
    assert topk is expected_topk
    assert index_calls[0][0] is hidden
    assert index_calls[0][-1] is None


def test_glm5_physical_zero_rope_is_shape_only():
    from sglang_fl.models.ascend_dsa import physical_zero_rope

    q_logical = torch.empty(3, 2, 0, dtype=torch.bfloat16)
    k_logical = torch.empty(3, 1, 0, dtype=torch.bfloat16)
    q_physical, k_physical = physical_zero_rope(q_logical, k_logical, 64)
    assert q_physical.shape == (3, 2, 64)
    assert k_physical.shape == (3, 1, 64)
    assert q_physical.dtype == k_physical.dtype == torch.bfloat16
    assert torch.count_nonzero(q_physical) == 0
    assert torch.count_nonzero(k_physical) == 0


def test_glm5_kpool_bridges_torch_npu_false_cuda_dispatch(monkeypatch):
    import sglang_fl.models.kpool_indexer as kpool_module

    IndexerKPool = kpool_module.IndexerKPool

    sentinel = object()
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_available=lambda: True), raising=False
    )
    monkeypatch.setattr(
        IndexerKPool,
        "forward_npu",
        lambda self, *args, **kwargs: (sentinel, args, kwargs),
    )
    monkeypatch.setattr(kpool_module, "is_npu", lambda: True)
    indexer = object.__new__(IndexerKPool)
    indexer._forward_method = indexer.forward_cuda
    indexer._bind_ascend_forward()
    result = indexer.forward(1, key=2)
    assert result == (sentinel, (1,), {"key": 2})


def test_glm5_kpool_resolves_hybrid_mla_metadata():
    from sglang_fl.models.kpool_indexer import _get_full_attn_metadata

    metadata = SimpleNamespace(block_tables=object())
    full_backend = SimpleNamespace(forward_metadata=metadata)
    batch = SimpleNamespace(
        attn_backend=SimpleNamespace(full_attn_backend=full_backend)
    )
    assert _get_full_attn_metadata(batch) is metadata


def test_glm5_kpool_resolves_hybrid_index_cache_and_layer_mapping():
    from sglang_fl.models.kpool_indexer import _get_index_k_buffer

    expected = object()
    calls = []

    class _FullPool:
        def get_index_k_buffer(self, layer_id):
            calls.append(("get", layer_id))
            return expected

    class _HybridPool:
        full_kv_pool = _FullPool()

        def _wait_for_layer(self, layer_id):
            calls.append(("wait", layer_id))

        def _transfer_full_attention_id(self, layer_id):
            calls.append(("map", layer_id))
            return 7

    batch = SimpleNamespace(token_to_kv_pool=_HybridPool())
    assert _get_index_k_buffer(batch, 23) is expected
    assert calls == [("wait", 23), ("map", 23), ("get", 7)]


def test_glm5_router_is_fp32_reference():
    from sglang_fl.models.glm5_next import Glm5NextMoEGate

    gate = object.__new__(Glm5NextMoEGate)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(
        torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
    )
    hidden = torch.tensor([[2.0, -1.0]], dtype=torch.bfloat16)
    logits = gate(hidden)
    reference = torch.nn.functional.linear(hidden.float(), gate.weight)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, reference, rtol=0, atol=0)
    assert torch.equal(torch.topk(logits, k=2, dim=-1).indices, torch.tensor([[0, 1]]))


@pytest.mark.gpu
def test_glm5_dynamic_cache_scatter_replays_current_rows(device):
    """Regression for CANN baking scatter indices into an NPU graph.

    Eight decode steps cross two KPool-4 closing boundaries.  Each replay must
    read the new row and payload from the static input tensors.
    """

    if device.type != "npu":
        pytest.skip("GLM-5.3 graph-safe scatter is Ascend-specific")

    from sglang_fl.models.graph_ops import scatter_rows_

    cache = torch.zeros((16, 8), dtype=torch.bfloat16, device=device)
    static_rows = torch.zeros((1,), dtype=torch.int64, device=device)
    static_data = torch.zeros((1, 8), dtype=torch.bfloat16, device=device)

    # Compile Triton before capture.
    scatter_rows_(cache, static_rows, static_data)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        scatter_rows_(cache, static_rows, static_data)

    expected = torch.zeros_like(cache)
    for step in range(8):
        row = step
        value = torch.full_like(static_data, step + 1)
        row_source = torch.tensor([row], device=device)
        static_rows.copy_(row_source)
        static_data.copy_(value)
        graph.replay()
        expected[row] = value[0]
        torch.npu.synchronize()
        torch.testing.assert_close(cache.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.gpu
def test_glm5_dynamic_tail_scatter_crosses_kpool_boundary(device):
    if device.type != "npu":
        pytest.skip("GLM-5.3 graph-safe scatter is Ascend-specific")

    from sglang_fl.models.graph_ops import scatter_rows_

    kpool = 4
    tail = torch.zeros((3, kpool, 4), dtype=torch.float32, device=device)
    static_row = torch.zeros((1,), dtype=torch.int64, device=device)
    static_data = torch.zeros((1, 4), dtype=torch.float32, device=device)
    scatter_rows_(tail, static_row, static_data)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        scatter_rows_(tail, static_row, static_data)

    expected = torch.zeros_like(tail)
    req = 2
    for position in range(8):
        slot = position % kpool
        flat_row = req * kpool + slot
        value = torch.full_like(static_data, 100 + position)
        row_source = torch.tensor([flat_row], device=device)
        static_row.copy_(row_source)
        static_data.copy_(value)
        graph.replay()
        expected[req, slot] = value[0]
        torch.npu.synchronize()
        torch.testing.assert_close(tail.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.gpu
def test_glm5_graph_cache_and_topk_match_eager_across_pool_boundaries(device):
    """Compare cache contents and selected ids across eight graph replays."""

    if device.type != "npu":
        pytest.skip("GLM-5.3 graph-safe scatter is Ascend-specific")

    from sglang_fl.models.graph_ops import scatter_rows_

    width = 8
    initial = -torch.arange(16, dtype=torch.float32, device=device).view(-1, 1)
    cache = initial.expand(-1, width).clone()
    static_row = torch.zeros((1,), dtype=torch.int64, device=device)
    static_data = torch.zeros((1, width), dtype=torch.float32, device=device)
    query = torch.arange(1, width + 1, dtype=torch.float32, device=device)

    scatter_rows_(cache, static_row, static_data)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        scatter_rows_(cache, static_row, static_data)
        graph_topk = torch.topk(cache @ query, k=4).indices

    expected_cache = initial.expand(-1, width).clone()
    for position in range(8):
        # Positions 3 -> 4 and 7 -> 8 cross KPool-4 closing boundaries.
        row = position + 1
        value = torch.full_like(static_data, 20 + position)
        row_source = torch.tensor([row], device=device)
        static_row.copy_(row_source)
        static_data.copy_(value)
        graph.replay()
        torch.npu.synchronize()

        expected_cache[row] = value[0]
        eager_topk = torch.topk(expected_cache @ query, k=4).indices
        torch.testing.assert_close(cache.cpu(), expected_cache.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(graph_topk.cpu(), eager_topk.cpu(), rtol=0, atol=0)
