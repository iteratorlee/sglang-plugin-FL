"""Fused mHC/native RMS against the materialized BF16 intermediate."""
import pytest
import torch


@pytest.mark.parametrize("tokens", [1, 16, 128, 4096])
@pytest.mark.parametrize("magnitude", [0.0, 1.0, 10.0])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_mhc_native_norm_numerics_and_graph(tokens, magnitude, weight_dtype):
    import torch_npu
    from sglang_fl.models.glm_53_flash.mhc import hc_pre

    torch.npu.set_device(0)
    torch.manual_seed(20260913)
    x = (torch.randn(tokens, 16384, device="npu") * magnitude).bfloat16()
    projection = torch.randn(24, 16384, device="npu") * 0.01
    scale = torch.tensor([0.1, 0.2, 0.3], device="npu")
    base = torch.randn(24, device="npu") * 2
    weight = torch.randn(4096, device="npu", dtype=weight_dtype)

    def reference():
        mixed, comb, post, fused = hc_pre(
            x, projection, scale, base, 4, 1e-5, 1e-6, 20
        )
        assert not fused
        z = mixed.float()
        normalized = (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5) * weight).bfloat16()
        return normalized, comb, post

    def run():
        result = hc_pre(
            x, projection, scale, base, 4, 1e-5, 1e-6, 20,
            out_norm_weight=weight, out_norm_eps=1e-5,
        )
        assert result[3]
        return result[:3]

    actual, expected = run(), reference()
    for a, b in zip(actual, expected):
        assert torch.isfinite(a).all()
        if a.dtype == torch.float32:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            error = ((a.float() - b.float()).square().mean() / b.float().square().mean().clamp_min(1e-20)).sqrt()
            assert error < 0.0003, error.item()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    x.mul_(0.75)
    base.add_(0.25)
    weight.mul_(0.5)
    changed = run()
    graph.replay()
    torch.npu.synchronize()
    for a, b in zip(captured, changed):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_norm_fusion_respects_runtime_formula():
    from types import SimpleNamespace
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang_fl.models.glm_53_flash.mhc_communicator import _State

    norm = SimpleNamespace(
        weight=torch.ones(4096), variance_epsilon=1e-5,
        cast_x_before_out_mul=False, override_orig_dtype=None,
        variance_size_override=None,
    )
    norm._forward_method = RMSNorm.forward_native.__get__(norm)
    weight, eps = _State._norm_args(norm)
    assert weight.data_ptr() == norm.weight.data_ptr() and eps == 1e-5
    norm._forward_method = RMSNorm.forward_npu.__get__(norm)
    assert _State._norm_args(norm) == (None, None)
    norm._forward_method = RMSNorm.forward_native.__get__(norm)
    norm.cast_x_before_out_mul = True
    assert _State._norm_args(norm) == (None, None)
