"""Fused GLM mHC versus vendor-equation PyTorch reference and graph replay."""

import pytest
import torch


@pytest.mark.parametrize("tokens", [0, 1, 16, 128, 4096])
@pytest.mark.parametrize("magnitude", [0.0, 1.0, 10.0])
def test_fused_mhc_numerics_and_graph(tokens, magnitude):
    import torch_npu
    from sglang_fl.models.glm_53_flash.mhc import (
        _hc_post_reference,
        _hc_pre_reference,
        hc_post,
        hc_pre,
    )

    torch.npu.set_device(0)
    torch.manual_seed(20260912)
    x = (torch.randn(tokens, 16384, device="npu") * magnitude).bfloat16()
    projection = torch.randn(24, 16384, device="npu") * 0.01
    scale = torch.tensor([0.1, 0.2, 0.3], device="npu")
    base = torch.randn(24, device="npu") * 2
    # Use an independent layer output to catch residual-mixing transpose errors.
    layer_output = torch.randn(tokens, 4096, device="npu", dtype=torch.bfloat16)

    def run(pre, post):
        inp, comb, gates, fused_norm = pre(
            x, projection, scale, base, 4, 1e-5, 1e-6, 20
        )
        assert fused_norm is False
        out = post(layer_output, x, gates, comb, 4)
        return inp, comb, gates, out

    expected = run(_hc_pre_reference, _hc_post_reference)
    actual = run(hc_pre, hc_post)
    for a, b in zip(actual, expected):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.isfinite(a).all()
        if not a.numel():
            continue
        if a.dtype == torch.float32:
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        else:
            error = (
                (a.float() - b.float()).square().mean()
                / b.float().square().mean().clamp_min(1e-20)
            ).sqrt()
            assert error < 0.0003, error.item()
    if tokens == 0:
        return
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run(hc_pre, hc_post)
    # Every coefficient/input remains a device read on graph replay.
    x.mul_(0.75)
    base.add_(0.25)
    layer_output.mul_(0.5)
    changed = run(hc_pre, hc_post)
    graph.replay()
    torch.npu.synchronize()
    for a, b in zip(captured, changed):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
