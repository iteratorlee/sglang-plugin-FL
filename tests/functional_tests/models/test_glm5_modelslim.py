"""ModelSlim naming and actual Ascend W8A8 expert regression tests."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang_fl.models.glm_53_flash.checkpoint import canonical_weight_name


@pytest.mark.parametrize(
    "source,expected",
    [
        ("model.language_model.layers.3.attn_hc.fn", "model.layers.3.hc_attn_fn"),
        ("model.language_model.layers.3.ffn_hc.scale", "model.layers.3.hc_ffn_scale"),
        (
            "model.language_model.layers.0.self_attn.forget_gate.f_a_proj.weight",
            "model.layers.0.self_attn.f_a_proj.weight",
        ),
        ("model.visual.blocks.0.attn.qkv.weight", "visual.blocks.0.attn.qkv.weight"),
        ("model.layers.0.hc_attn_fn", "model.layers.0.hc_attn_fn"),
        ("lm_head.weight", "lm_head.weight"),
    ],
)
def test_checkpoint_names(source, expected):
    assert canonical_weight_name(source) == expected
    assert canonical_weight_name(expected) == expected


def test_modelslim_config_is_scoped():
    from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
    from sglang_fl.models.glm_53_flash.modelslim import GlmModelSlimConfig

    key = "model.language_model.layers.3.mlp.experts.0.gate_proj.weight"
    base = ModelSlimConfig({key: "W8A8_DYNAMIC"})
    base.packed_modules_mapping = {"model": {"gate_up_proj": ["gate_proj", "up_proj"]}}
    adapted = GlmModelSlimConfig(base)
    assert adapted.quant_description[canonical_weight_name(key)] == "W8A8_DYNAMIC"
    assert key in base.quant_description
    adapted.packed_modules_mapping["model"]["gate_up_proj"].append("test")
    assert base.packed_modules_mapping["model"]["gate_up_proj"] == [
        "gate_proj",
        "up_proj",
    ]
    with pytest.raises(ValueError, match="Unsupported GLM"):
        adapted.get_moe_scheme(None, "missing")


def _quantize_reference(x):
    scale = x.float().abs().amax(-1) / 127
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = (x.float() / scale[:, None]).round().clamp(-128, 127)
    return quantized, scale


def _expert_reference(x, w13, w2, s13, s2, counts):
    """CPU FP32 dequantization, clamp, SiLU and activation quantization."""
    outputs = []
    start = 0
    for expert, count in enumerate(counts):
        part = x[start : start + count]
        q, scale = _quantize_reference(part)
        gate_up = (q @ w13[expert].float().T) * scale[:, None] * s13[expert]
        gate, up = gate_up.chunk(2, -1)
        activated = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
        q, scale = _quantize_reference(activated)
        outputs.append(
            ((q @ w2[expert].float().T) * scale[:, None] * s2[expert]).bfloat16()
        )
        start += count
    return torch.cat(outputs)


@pytest.mark.parametrize("counts", [[5, 0, 27], [0, 32, 0]])
def test_w8a8_experts_reference_and_graph(counts, record_property):
    import torch_npu
    from sglang_fl.models.glm_53_flash.modelslim import GlmW8A8MoEMethod

    torch.npu.set_device(0)
    # Match init_npu_backend: otherwise format casts silently remain ND and
    # cannot expose production-only NZ storage/layout regressions.
    torch.npu.config.allow_internal_format = True
    generator = torch.Generator().manual_seed(20260911)
    hidden, intermediate, experts = 4096, 2048, len(counts)
    w13 = torch.randint(
        -64,
        65,
        (experts, 2 * intermediate, hidden),
        generator=generator,
        dtype=torch.int8,
    )
    w2 = torch.randint(
        -64, 65, (experts, hidden, intermediate), generator=generator, dtype=torch.int8
    )
    s13 = torch.rand(experts, 2 * intermediate, generator=generator) * 0.001 + 0.003
    s2 = torch.rand(experts, hidden, generator=generator) * 0.001 + 0.001
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(w13.npu(), requires_grad=False),
        w2_weight=torch.nn.Parameter(w2.npu(), requires_grad=False),
        w13_weight_scale=torch.nn.Parameter(
            s13.float().unsqueeze(-1).npu(), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            s2.float().unsqueeze(-1).npu(), requires_grad=False
        ),
        w13_weight_offset=torch.zeros(experts, 2 * intermediate, 1, device="npu"),
        w2_weight_offset=torch.zeros(experts, hidden, 1, device="npu"),
        _sglang_fl_swiglu_limit=10.0,
    )
    kernel = GlmW8A8MoEMethod()
    torch.npu.synchronize()
    allocated_before = torch.npu.memory_allocated()
    kernel.process_weights_after_loading(layer)
    torch.npu.synchronize()
    storage_growth = torch.npu.memory_allocated() - allocated_before
    record_property("postprocess_storage_growth_bytes", storage_growth)
    # Aligned NZ weights need no padding; only small scale caches are added.
    assert storage_growth < 5 * 1024**2
    assert torch_npu.get_npu_format(layer.w13_weight) == 29
    assert torch_npu.get_npu_format(layer.w2_weight) == 29
    assert layer.w13_weight.dtype == layer.w2_weight.dtype == torch.int8
    group_list = torch.tensor(counts, dtype=torch.int64, device="npu")
    x_cpu = (torch.randn(sum(counts), hidden, generator=generator) * 3).bfloat16()
    x = x_cpu.npu()

    def run():
        return kernel.apply_without_routing_weights(
            layer, x, None, 1, group_list, torch.bfloat16
        )

    eager = run()
    reference = _expert_reference(x_cpu, w13, w2, s13, s2, counts)
    actual = eager.cpu().float()
    # The CPU reference and CANN round dynamic quantization at different
    # boundaries. Assert relative error, cosine similarity and finite output.
    relative_rmse = (
        (actual - reference.float()).square().mean() / reference.float().square().mean()
    ).sqrt()
    # Random FP32 GEMM2 scales catch accidental reintroduction of the stock
    # BF16 scale cache (about 0.33% error for these inputs).
    assert relative_rmse < 0.003, relative_rmse.item()
    cosine = F.cosine_similarity(actual.flatten(), reference.float().flatten(), dim=0)
    record_property("relative_rmse", relative_rmse.item())
    record_property("cosine_similarity", cosine.item())
    assert cosine > 0.99999
    assert torch.isfinite(actual).all()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
    x.copy_((x_cpu * 0.25).npu())
    changed_eager = run()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured, changed_eager, rtol=0, atol=0)

    # Routing changes every token. Replay must consume the new histogram,
    # including transitions between empty and non-empty experts.
    for new_counts in ([27, 5, 0], [0, 0, 32], [16, 0, 16]):
        group_list.copy_(torch.tensor(new_counts, dtype=torch.int64, device="npu"))
        changed_eager = run()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured, changed_eager, rtol=0, atol=0)

    # Graph padding includes zero activations; its dynamic scales must not
    # introduce NaNs or nonzero expert output.
    x.zero_()
    graph.replay()
    torch.npu.synchronize()
    assert torch.isfinite(captured).all()
    assert torch.count_nonzero(captured).item() == 0


def test_standard_dispatch_fails_explicitly():
    from sglang_fl.models.glm_53_flash.modelslim import GlmW8A8MoEMethod

    with pytest.raises(NotImplementedError, match="deepep"):
        GlmW8A8MoEMethod().apply(None, None)


@pytest.mark.parametrize("counts", [[2, 0, 3], [0, 0, 0]])
def test_grouped_dequantization_fp32_scales_and_padding(counts):
    import torch_npu
    from sglang_fl.models.glm_53_flash.quant_ops import dequantize_grouped_int32

    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(20260912)
    accum = torch.randint(-1000000, 1000000, (8, 4096), generator=generator)
    weights = torch.rand(3, 4096, generator=generator) * 0.001
    tokens = torch.rand(8, generator=generator) * 0.01
    expected = torch.zeros(8, 4096, dtype=torch.bfloat16)
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        expected[start:end] = (
            accum[start:end].float() * tokens[start:end, None] * weights[expert]
        ).bfloat16()
        start = end
    accum_npu = accum.to(device="npu", dtype=torch.int32)
    assert torch_npu.get_npu_format(accum_npu) == 2
    weights_npu, tokens_npu = weights.npu(), tokens.npu()
    counts_npu = torch.tensor(counts, dtype=torch.int64, device="npu")

    def run():
        return dequantize_grouped_int32(
            accum_npu, weights_npu, tokens_npu, counts_npu, torch.bfloat16
        )

    eager = run()
    torch.testing.assert_close(eager.cpu(), expected, rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    counts_npu.zero_()
    graph.replay()
    torch.npu.synchronize()
    assert torch.count_nonzero(captured).item() == 0


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scales_are_rejected(invalid):
    from sglang_fl.models.glm_53_flash.modelslim import GlmW8A8MoEMethod

    layer = SimpleNamespace(
        w13_weight_offset=torch.zeros(1),
        w2_weight_offset=torch.zeros(1),
        w13_weight_scale=torch.tensor([invalid]),
    )
    with pytest.raises(ValueError, match="Invalid GLM W8A8 scales"):
        GlmW8A8MoEMethod().process_weights_after_loading(layer)


def test_asymmetric_offsets_are_rejected():
    from sglang_fl.models.glm_53_flash.modelslim import GlmW8A8MoEMethod

    layer = SimpleNamespace(w13_weight_offset=torch.ones(1))
    with pytest.raises(ValueError, match="symmetric weights"):
        GlmW8A8MoEMethod().process_weights_after_loading(layer)
