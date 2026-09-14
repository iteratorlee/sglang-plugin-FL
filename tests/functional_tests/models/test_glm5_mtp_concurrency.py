"""Packed accepted lengths must never write rejected/padded MTP cache slots."""
from types import SimpleNamespace
import pytest
import torch

@pytest.mark.parametrize('bs', [4, 8, 16, 24, 32, 36, 40])
@pytest.mark.parametrize('steps', [3, 4])
def test_many_request_accepted_kpool_graph(monkeypatch, bs, steps):
    import torch_npu
    from sglang_fl.models.glm_53_flash import kpool_indexer as mod
    torch.npu.set_device(0)
    idx=mod.IndexerKPool.__new__(mod.IndexerKPool)
    torch.nn.Module.__init__(idx)
    idx.index_kpool,idx.head_dim,idx.index_topk=4,128,4
    idx._kpool_padding_req=0
    idx._glm53_draft_graph_steps=steps
    idx._kpool_tail_k=torch.randn(bs+1,4,128,device='npu',dtype=torch.bfloat16)
    idx._kpool_tail_score=torch.randn(bs+1,4,128,device='npu')
    idx.index_kpool_compress_ape=torch.zeros(4,128,device='npu')
    n=bs*steps
    cache=torch.zeros((bs+1)*128,128,device='npu',dtype=torch.bfloat16)
    q=torch.randn(n,1,128,device='npu',dtype=torch.bfloat16)
    key=torch.randn(n,128,device='npu',dtype=torch.bfloat16)
    score=torch.randn(n,128,device='npu')
    weight=torch.ones(n,1,device='npu')
    lens=torch.full((bs,),steps,device='npu',dtype=torch.int32)
    req=torch.arange(1,bs+1,device='npu')
    pos=torch.zeros(n,device='npu',dtype=torch.int64)
    table=torch.tensor([[i]*4 for i in range(1,bs+1)],device='npu',dtype=torch.int32)
    meta=SimpleNamespace(seq_lens=torch.zeros(bs,device='npu',dtype=torch.int32))
    fb=SimpleNamespace(batch_size=bs,extend_seq_lens=lens,req_pool_indices=req,
        attn_backend=SimpleNamespace(forward_metadata=meta))
    monkeypatch.setattr(mod,'_get_index_k_buffer',lambda *_:cache)
    monkeypatch.setattr(mod.torch_npu,'npu_lightning_indexer',lambda **kw:
        (torch.where(kw['actual_seq_lengths_key'][:,None,None]>0,
                     torch.zeros(bs,1,1,device='npu',dtype=torch.int32),-1),None))
    def run(): return idx._draft_extend_topk(q,key,weight,score,pos,fb,table,0)
    run();torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph): result=run()
    for i in range(2*steps):
        lengths=[1+(i+j)%steps for j in range(bs)]
        slots=list(range(1,bs+1))
        if i>=steps: slots.reverse()
        if i%2: slots[-1]=0
        lens.copy_(torch.tensor(lengths,device='npu',dtype=torch.int32))
        req.copy_(torch.tensor(slots,device='npu'))
        table.copy_(torch.tensor([[v]*4 for v in slots],device='npu',dtype=torch.int32))
        positions=[]
        for j,length in enumerate(lengths): positions.extend(i+j+k for k in range(length))
        positions+= [0]*(n-len(positions))
        pos.copy_(torch.tensor(positions,device='npu'))
        key.normal_();score.normal_()
        expected_k,expected_s=idx._kpool_tail_k.cpu(),idx._kpool_tail_score.cpu()
        keys,scores=key.cpu(),score.cpu()
        offset=0
        for slot,length in zip(slots,lengths):
            if slot:
                for t in range(offset,offset+length):
                    expected_k[slot,positions[t]%4]=keys[t]
                    expected_s[slot,positions[t]%4]=scores[t]
            offset+=length
        graph.replay();torch.npu.synchronize()
        torch.testing.assert_close(idx._kpool_tail_k[1:].cpu(),expected_k[1:],rtol=0,atol=0)
        torch.testing.assert_close(idx._kpool_tail_score[1:].cpu(),expected_s[1:],rtol=0,atol=0)



@pytest.mark.parametrize("requests,tokens,wanted", [
    (1,4,(16,4)), (2,3,(16,16)), (5,4,(16,8)), (17,4,(32,20)),
    (31,3,(32,32)), (32,4,(32,32)), (33,4,(48,36)), (33,5,(48,48)),
])
def test_capture_capacity_boundaries(requests,tokens,wanted):
    from sglang_fl.models.glm_53_flash.mtp_draft_graph import capture_counts
    assert capture_counts(requests,16,tokens)==wanted


@pytest.mark.parametrize("requests,tp,tokens,draft,extend", [
    (1,16,4,[16],[4]),
    (4,16,4,[16],[4]),
    (8,16,4,[16],[4,8]),
    (17,16,4,[16,32],[4,8,12,16,20]),
    (33,16,3,[16,32,48],[16,32,48]),
    (33,16,5,[16,32,48],[16,32,48]),
    (33,32,4,[32,64],[8,16,24,32,40]),
])
def test_dense_mtp_capture_selects_smallest_legal_graph(requests,tp,tokens,draft,extend):
    import bisect
    import math
    from sglang_fl.models.glm_53_flash.mtp_draft_graph import capture_batch_sizes
    assert capture_batch_sizes(requests,tp,tokens,dense=True)==(draft,extend)
    assert capture_batch_sizes(requests,tp,tokens)==([draft[-1]],[extend[-1]])
    for real in range(1,requests+1):
        d=draft[bisect.bisect_left(draft,real)]
        e=extend[bisect.bisect_left(extend,real)]
        assert real<=d<real+tp and d%tp==0
        unit=tp//math.gcd(tp,tokens)
        assert real<=e<real+unit and e*tokens%tp==0
