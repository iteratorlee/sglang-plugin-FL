"""Page boundaries and changing real/padded requests in NPU graph replay."""

import pytest
import torch


@pytest.mark.parametrize("bs", [1, 2, 5, 16])
@pytest.mark.parametrize("columns", [1028, 2053])
def test_kpool_metadata_graph(bs, columns):
    import torch_npu
    from sglang_fl.models.glm_53_flash.kpool_metadata_npu import decode_metadata

    torch.npu.set_device(0)
    req = torch.ones(bs, device="npu", dtype=torch.int64)
    seq = torch.ones(bs, device="npu", dtype=torch.int32)
    pos = torch.zeros(bs, device="npu", dtype=torch.int64)
    table = torch.ones(bs, columns, device="npu", dtype=torch.int32)
    decode_metadata(req, seq, pos, table)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = decode_metadata(req, seq, pos, table)
    for replay, position in enumerate([3, 4, 63, 64, 255, 256, 65535]):
        r = (torch.arange(bs, dtype=torch.int64) + replay) % 3
        p = torch.full((bs,), position, dtype=torch.int64)
        s = p.int() + 1
        s[(torch.arange(bs) + replay) % 4 == 0] = 0
        tb = torch.randint(-1, 2048, (bs, columns), dtype=torch.int32)
        req.copy_(r); seq.copy_(s); pos.copy_(p); table.copy_(tb)
        graph.replay()
        torch.npu.synchronize()
        valid = (r > 0) & (s > 0)
        r = torch.where(valid, r, 0)
        slot = p % 4
        pool = p // 4
        closing = (slot == 3) & valid
        page = tb[torch.arange(bs), (pool // 64) * 4].clamp_min(0)
        expected = (r, r * 4 + slot,
            torch.where(closing, page.long() * 64 + pool % 64, 0),
            closing, s // 4, torch.arange(1, bs + 1, dtype=torch.int32),
            tb[:, ::4].contiguous().clamp_min(0))
        for actual, reference in zip(output, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
