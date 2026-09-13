"""Packed accepted lengths must never write rejected/padded MTP cache slots."""
from types import SimpleNamespace
import pytest
import torch

@pytest.mark.parametrize('bs', [1, 3, 16])
@pytest.mark.parametrize('steps', [3, 4, 5])
def test_multistep_accepted_kpool_graph(monkeypatch, bs, steps):
    import torch_npu
    from sglang_fl.models.glm_53_flash import kpool_indexer as mod
    torch.npu.set_device(0)
    idx=mod.IndexerKPool.__new__(mod.IndexerKPool)
    torch.nn.Module.__init__(idx)
    idx.index_kpool,idx.head_dim,idx.index_topk=4,128,4
    idx._kpool_padding_req=0
    idx._glm53_draft_graph_steps=steps
    idx._kpool_tail_k=torch.randn(3,4,128,device='npu',dtype=torch.bfloat16)
    idx._kpool_tail_score=torch.randn(3,4,128,device='npu')
    idx.index_kpool_compress_ape=torch.zeros(4,128,device='npu')
    n=bs*steps
    cache=torch.zeros(384,128,device='npu',dtype=torch.bfloat16)
    q=torch.randn(n,1,128,device='npu',dtype=torch.bfloat16)
    key=torch.randn(n,128,device='npu',dtype=torch.bfloat16)
    score=torch.randn(n,128,device='npu')
    weight=torch.ones(n,1,device='npu')
    lens=torch.full((bs,),steps,device='npu',dtype=torch.int32)
    req=torch.tensor(([1,2]+[0]*bs)[:bs],device='npu')
    pos=torch.zeros(n,device='npu',dtype=torch.int64)
    table=torch.tensor([[i]*4 for i in ([1,2]+[0]*bs)[:bs]],device='npu',dtype=torch.int32)
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
        lengths=([1+i%steps,1+(i//2)%steps]+[steps]*bs)[:bs]
        slots=([1,2]+[0]*bs)[:bs]
        if i>=steps: slots=([2,1]+[0]*bs)[:bs]
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


@pytest.mark.parametrize("raises", [False, True])
def test_draft_restores_kpool_transaction(raises):
    from sglang_fl.models.glm_53_flash.mtp_draft_graph import wrap_draft_transaction
    indexer=SimpleNamespace(_kpool_tail_k=torch.randn(3,4,128), _kpool_tail_score=torch.randn(3,4,128))
    model=type("Glm5NextForConditionalGenerationNextN", (), {})()
    model.model=SimpleNamespace(decoder=SimpleNamespace(self_attn=SimpleNamespace(indexer=indexer)))
    worker=SimpleNamespace(draft_model_runner=SimpleNamespace(model=model), speculative_num_steps=2)
    before=[indexer._kpool_tail_k.clone(), indexer._kpool_tail_score.clone()]
    def original(self, batch):
        indexer._kpool_tail_k.add_(2)
        indexer._kpool_tail_score.zero_()
        if raises: raise RuntimeError("draft failed")
        return batch
    wrapped=wrap_draft_transaction(original)
    if raises:
        with pytest.raises(RuntimeError, match="draft failed"): wrapped(worker, "output")
    else: assert wrapped(worker, "output")=="output"
    torch.testing.assert_close(indexer._kpool_tail_k,before[0],rtol=0,atol=0)
    torch.testing.assert_close(indexer._kpool_tail_score,before[1],rtol=0,atol=0)
