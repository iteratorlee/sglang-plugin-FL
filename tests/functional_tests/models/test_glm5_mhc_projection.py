"""Small mHC projections retain FP32 weights and support changing graph data."""

import pytest
import torch


@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("magnitude", [0.0, 1.0, 10.0])
def test_mhc_projection_fp32_graph(tokens, magnitude):
    import torch_npu
    from sglang_fl.models.glm_53_flash.mhc_npu import project_mhc

    torch.npu.set_device(0)
    torch.manual_seed(20260913)
    x = (torch.randn(tokens, 16384, device="npu") * magnitude).bfloat16()
    weight = torch.randn(24, 16384, device="npu", dtype=torch.float32) * 0.01

    def check(actual):
        expected = torch.nn.functional.linear(x.float(), weight)
        error = (
            (actual - expected).square().mean()
            / expected.square().mean().clamp_min(1e-20)
        ).sqrt()
        assert actual.dtype == weight.dtype == torch.float32
        assert torch.isfinite(actual).all()
        assert error < 1e-5, error.item()

    check(project_mhc(x, weight))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = project_mhc(x, weight)
    for factor in (0.0, 0.75, 2.0):
        x.normal_().mul_(factor)
        weight.add_(0.0001)
        graph.replay()
        torch.npu.synchronize()
        check(captured)
        torch.testing.assert_close(captured, project_mhc(x, weight), rtol=0, atol=0)
