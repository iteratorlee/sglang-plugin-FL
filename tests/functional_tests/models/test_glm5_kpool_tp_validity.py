"""KPool request validity is independent of each rank's sparse-MoE slice."""
from types import SimpleNamespace
import pytest
import torch


@pytest.mark.parametrize('batch', [1, 4])
@pytest.mark.parametrize('moe_tokens', [0, 1])
def test_kpool_graph_updates_all_attention_ranks(monkeypatch, batch, moe_tokens):
    import torch_npu
    from sglang_fl.models.glm_53_flash import kpool_indexer as mod
    torch.npu.set_device(0)
    indexer = mod.IndexerKPool.__new__(mod.IndexerKPool)
    torch.nn.Module.__init__(indexer)
    indexer.index_kpool, indexer.head_dim, indexer.index_topk = 4, 128, 4
    indexer._kpool_padding_req = 0
    indexer._kpool_tail_k = torch.randn(3, 4, 128, device='npu', dtype=torch.bfloat16)
    indexer._kpool_tail_score = torch.randn(3, 4, 128, device='npu')
    indexer.index_kpool_compress_ape = torch.zeros(4, 128, device='npu')
    cache = torch.zeros(128, 128, device='npu', dtype=torch.bfloat16)
    q = torch.randn(batch, 1, 128, device='npu', dtype=torch.bfloat16)
    k = torch.randn(batch, 128, device='npu', dtype=torch.bfloat16)
    w = torch.ones(batch, 1, device='npu', dtype=torch.float32)
    gate = torch.randn(batch, 128, device='npu', dtype=torch.float32)
    positions = torch.tensor([3]+[0]*(batch-1), device='npu')
    ids = torch.tensor([1]+[0]*(batch-1), device='npu')
    seq = torch.tensor([4]+[0]*(batch-1), dtype=torch.int32, device='npu')
    table = torch.ones(batch, 4, dtype=torch.int32, device='npu')
    metadata = SimpleNamespace(seq_lens=seq)
    fb = SimpleNamespace(req_pool_indices=ids, num_token_non_padded=torch.tensor([moe_tokens], device='npu'))
    monkeypatch.setattr(mod, '_get_full_attn_metadata', lambda _: metadata)
    monkeypatch.setattr(mod, '_get_index_k_buffer', lambda *_: cache)
    # Isolate cache validity from LightningIndexer ranking, which is unchanged.
    monkeypatch.setattr(mod.torch_npu, 'npu_lightning_indexer', lambda **_: (torch.zeros(batch,1,1,device='npu',dtype=torch.int32),None))
    def run(): return indexer._decode_topk(q,k,w,gate,positions,fb,table,0)
    run();torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph): run()
    for real_req in [2, 1]:
        ids[0] = real_req
        k.normal_();gate.normal_();cache.zero_()
        before_k = indexer._kpool_tail_k.cpu()
        before_score = indexer._kpool_tail_score.cpu()
        graph.replay();torch.npu.synchronize()
        torch.testing.assert_close(indexer._kpool_tail_k[real_req,3], k[0], rtol=0, atol=0)
        torch.testing.assert_close(indexer._kpool_tail_score[real_req,3], gate[0], rtol=0, atol=0)
        unused = 3-real_req
        torch.testing.assert_close(indexer._kpool_tail_k[unused].cpu(),before_k[unused],rtol=0,atol=0)
        torch.testing.assert_close(indexer._kpool_tail_score[unused].cpu(),before_score[unused],rtol=0,atol=0)
        expected = indexer._compress(indexer._kpool_tail_k[real_req:real_req+1],indexer._kpool_tail_score[real_req:real_req+1])
        torch.testing.assert_close(cache[64],expected[0],rtol=0,atol=0)
