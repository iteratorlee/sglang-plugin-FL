"""MTP KPool tails must survive rejected tokens across four-token boundaries."""
from copy import copy
from types import SimpleNamespace
import pytest
import torch


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('steps', [2, 5])
def test_mtp_kpool_commit_after_graph_verify(monkeypatch, batch, steps):
    import torch_npu
    from sglang_fl.models.glm_53_flash import kpool_indexer as mod
    from sglang_fl.models.glm_53_flash.mtp_state_npu import commit_state
    torch.npu.set_device(0)
    indexer = mod.IndexerKPool.__new__(mod.IndexerKPool)
    torch.nn.Module.__init__(indexer)
    indexer.index_kpool, indexer.head_dim, indexer.index_topk = 4, 128, 4
    indexer._kpool_padding_req = 0
    indexer._kpool_tail_k = torch.randn(3,4,128,device='npu',dtype=torch.bfloat16)
    indexer._kpool_tail_score = torch.randn(3,4,128,device='npu')
    indexer._kpool_mtp_tail_k = torch.zeros(batch+1,steps,4,128,device='npu',dtype=torch.bfloat16)
    indexer._kpool_mtp_tail_score = torch.zeros(batch+1,steps,4,128,device='npu')
    indexer.index_kpool_compress_ape = torch.zeros(4,128,device='npu')
    cache = torch.zeros(384,128,device='npu',dtype=torch.bfloat16)
    q = torch.randn(batch*steps,1,128,device='npu',dtype=torch.bfloat16)
    key = torch.randn(batch*steps,128,device='npu',dtype=torch.bfloat16)
    weights = torch.ones(batch*steps,1,device='npu',dtype=torch.float32)
    score = torch.randn(batch*steps,128,device='npu')
    ids = torch.tensor([1,2,0][:batch],device='npu')
    positions = torch.tensor([p+i for p in [1,3,0][:batch] for i in range(steps)],device='npu')
    table = torch.tensor([[1]*4,[2]*4,[0]*4][:batch],device='npu',dtype=torch.int32)
    metadata = SimpleNamespace(seq_lens=torch.tensor([steps+1,steps+3,0][:batch],device='npu'))
    fb = SimpleNamespace(req_pool_indices=ids,batch_size=batch,
        spec_info=SimpleNamespace(draft_token_num=steps),
        attn_backend=SimpleNamespace(forward_metadata=metadata),
        num_token_non_padded=torch.tensor([0],device='npu'))
    accepted = torch.tensor([0,steps-1,-1][:batch],device='npu')
    monkeypatch.setattr(mod, '_get_index_k_buffer', lambda *_: cache)
    # Ranking is not changed by this patch; expose one legal closed pool.
    def index(**kw):
        return torch.where(kw['actual_seq_lengths_key'][:,None,None]>0,
            torch.zeros(batch,1,1,device='npu',dtype=torch.int32),-1), None
    monkeypatch.setattr(mod.torch_npu,'npu_lightning_indexer',index)
    def verify(): return indexer._verify_topk(q,key,weights,score,positions,fb,table,0)
    def commit():
        commit_state(indexer._kpool_tail_k[None],indexer._kpool_mtp_tail_k[None],ids,accepted)
        commit_state(indexer._kpool_tail_score[None],indexer._kpool_mtp_tail_score[None],ids,accepted)
    verify();commit();torch.npu.synchronize()
    vg,cg = torch.npu.NPUGraph(),torch.npu.NPUGraph()
    with torch.npu.graph(vg): result = verify()
    with torch.npu.graph(cg): commit()
    for iteration in range(2):
        key.normal_();score.normal_()
        if iteration:
            ids.copy_(torch.tensor([2,1,0][:batch],device='npu'))
            accepted.copy_(torch.tensor([steps//2,0,-1][:batch],device='npu'))
        before_k,before_s = indexer._kpool_tail_k.cpu(),indexer._kpool_tail_score.cpu()
        kc,sc,pc = key.cpu(),score.cpu(),positions.cpu().tolist()
        snapshots = {}
        for req,slot in enumerate(ids.cpu().tolist()):
            if slot == 0: continue
            k,s = before_k[slot].clone(),before_s[slot].clone()
            snapshots[req] = []
            for step in range(steps):
                token=req*steps+step
                k[pc[token]%4]=kc[token]
                s[pc[token]%4]=sc[token]
                snapshots[req].append((k.clone(),s.clone()))
        vg.replay();torch.npu.synchronize()
        for req in snapshots:
            for step,(k,s) in enumerate(snapshots[req]):
                torch.testing.assert_close(indexer._kpool_mtp_tail_k[req,step].cpu(),k,rtol=0,atol=0)
                torch.testing.assert_close(indexer._kpool_mtp_tail_score[req,step].cpu(),s,rtol=0,atol=0)
        for req,(slot,step) in enumerate(zip(ids.cpu().tolist(),accepted.cpu().tolist())):
            if slot and step>=0:
                before_k[slot],before_s[slot]=snapshots[req][step]
        cg.replay();torch.npu.synchronize()
        torch.testing.assert_close(indexer._kpool_tail_k[1:].cpu(),before_k[1:],rtol=0,atol=0)
        torch.testing.assert_close(indexer._kpool_tail_score[1:].cpu(),before_s[1:],rtol=0,atol=0)
