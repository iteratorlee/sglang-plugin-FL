"""Exact FP32-scale output dequantization with changing EP16 routing."""

import pytest
import torch


@pytest.mark.parametrize(
    "rows,channels", [(256, 257), (257, 4096), (1024, 4096), (16384, 4096)]
)
def test_dequantization_padding_and_graph(rows, channels):
    import torch_npu
    from sglang_fl.models.glm_53_flash.quant_ops import dequantize_grouped_int32

    torch.npu.set_device(0)
    gen = torch.Generator().manual_seed(20260912)
    accum_cpu = torch.randint(
        -1000000, 1000000, (rows, channels), generator=gen, dtype=torch.int32
    )
    ws_cpu = torch.rand(18, channels, generator=gen) * 0.01
    ts_cpu = torch.rand(rows, generator=gen) * 0.03
    accum, ws, ts = accum_cpu.npu(), ws_cpu.npu(), ts_cpu.npu()
    counts = torch.zeros(18, device="npu", dtype=torch.int64)

    def run():
        return dequantize_grouped_int32(accum, ws, ts, counts, torch.bfloat16)

    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    for hist in (
        [0] * 18,
        [rows] + [0] * 17,
        [0] * 17 + [rows],
        [3, 0, 29] + [0] * 14 + [rows - 40],
        [rows // 18] * 18,
    ):
        counts.copy_(torch.tensor(hist, dtype=torch.int64, device="npu"))
        expected = torch.zeros(rows, channels, dtype=torch.bfloat16)
        start = 0
        for expert, count in enumerate(hist):
            end = start + count
            expected[start:end] = (
                accum_cpu[start:end].float() * ts_cpu[start:end, None] * ws_cpu[expert]
            ).bfloat16()
            start = end
        eager = run()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(eager.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
