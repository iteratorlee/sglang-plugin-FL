"""CPU rolling-window oracle over changing decode graph slots."""

import pytest
import torch


@pytest.mark.parametrize("batch", [1, 16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("magnitude", [0.0, 1.0, 10.0])
def test_causal_conv_multistep_graph(batch, weight_dtype, magnitude):
    import torch_npu
    from sglang_fl.models.glm_53_flash.causal_conv_npu import (
        causal_conv_step,
        supports_causal_conv_step,
    )

    torch.npu.set_device(0)
    torch.manual_seed(20260913)
    dim = 1536
    weight_cpu = (torch.randn(dim, 4) * 0.2).to(weight_dtype)
    state_cpu = (torch.randn(40, dim, 3) * magnitude).bfloat16()
    x = torch.zeros(batch, dim, device="npu", dtype=torch.bfloat16)
    state = state_cpu.npu()
    weight = weight_cpu.npu()
    indices = torch.zeros(batch, device="npu", dtype=torch.int32)
    assert supports_causal_conv_step(x, state, weight, None, indices)
    assert not supports_causal_conv_step(x, state, weight, weight[0], indices)
    causal_conv_step(x, state, weight, indices)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        causal_conv_step(x, state, weight, indices)
    for step in range(8):
        input_cpu = (torch.randn(batch, dim) * magnitude).bfloat16()
        # Include all-padding and fully occupied steps, plus moving live slots.
        slots = [
            0 if step == 0 or (step % 3 == 0 and i % 2 == 0)
            else i + 1 + (step % 2) * batch
            for i in range(batch)
        ]
        expected = input_cpu.clone()
        for token, slot in enumerate(slots):
            if not slot:
                continue
            window = torch.cat(
                [state_cpu[slot], input_cpu[token, :, None]], dim=-1
            ).float()
            products = window * weight_cpu.float()
            acc = ((products[:, 0] + products[:, 1]) + products[:, 2]) + products[:, 3]
            expected[token] = (acc / (1 + torch.exp(-acc))).bfloat16()
            state_cpu[slot] = window[:, 1:].bfloat16()
        x.copy_(input_cpu)
        indices.copy_(torch.tensor(slots, dtype=torch.int32))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(state.cpu(), state_cpu, rtol=0, atol=0)
        error = (
            (x.cpu().float() - expected.float()).square().mean()
            / expected.float().square().mean().clamp_min(1e-20)
        ).sqrt()
        assert error < 0.0003, (step, error.item())
