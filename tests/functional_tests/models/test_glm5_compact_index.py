"""Allocation geometry and logical compressed-history equivalence under reuse."""
import pytest
import torch


def test_allocation_mode_only_index():
    from sglang_fl.models.glm_53_flash.compact_index import make_layout, IndexAllocationMode
    layout=make_layout(2,512,4096)
    with IndexAllocationMode(layout) as mode:
        k=torch.zeros((2,65,64,1,512))
        v=torch.zeros((2,65,64,1,64))
        index=torch.zeros((2,65,64,1,128))
        state=torch.zeros((2,65,64,128))
    assert mode.redirected==1 and index.shape[1]==layout.pages
    assert k.shape[1]==v.shape[1]==state.shape[1]==65
    assert make_layout(32,1048576,196608) is None
    with pytest.raises(ValueError):make_layout(1,512,4096,page_size=128)


@pytest.mark.parametrize('requests,context',[(1,65536),(4,131072),(40,66624)])
@pytest.mark.parametrize('cols',[1,4,7,1024,2080])
def test_dynamic_slot_table_and_history(requests,context,cols):
    import torch_npu
    from sglang_fl.models.glm_53_flash.compact_index import make_layout
    from sglang_fl.models.glm_53_flash.compact_index_npu import request_block_table
    torch.npu.set_device(0)
    layout=make_layout(requests,context,requests*(context+2048))
    rows=requests+2
    req=torch.zeros(rows*2,device='npu',dtype=torch.int64)[::2]
    original=torch.zeros(rows,cols,device='npu',dtype=torch.int32)
    request_block_table(req,original,layout);torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):output=request_block_table(req,original,layout)
    for replay in range(4):
        slots=list(range(1,requests+1));slots=slots[replay%requests:]+slots[:replay%requests]
        if replay&1:slots.reverse()
        slots += [0,requests+1]
        req.copy_(torch.tensor(slots,device='npu'))
        graph.replay();torch.npu.synchronize()
        actual=output.cpu()
        for row,slot in enumerate(slots):
            if slot==0 or slot>requests:
                assert not actual[row].any();continue
            # Independently enumerate the owned pages. Each compressed page
            # represents four ordinary pages and never aliases another request.
            owned=list(range(1+(slot-1)*layout.pages_per_request,
                             1+slot*layout.pages_per_request))
            wanted=[page for page in owned for _ in range(4)]
            wanted=(wanted+[owned[-1]]*cols)[:cols]
            assert actual[row].tolist()==wanted
        assert actual.min()>=0 and actual.max()<layout.pages


@pytest.mark.parametrize('phase',['prefill','decode','verify','accepted'])
def test_fragmented_index_history_equivalence(phase):
    import torch_npu
    from sglang_fl.models.glm_53_flash.compact_index import make_layout
    from sglang_fl.models.glm_53_flash.compact_index_npu import request_block_table
    from sglang_fl.models.glm_53_flash.kpool_indexer import IndexerKPool
    from sglang_fl.models.glm_53_flash.graph_ops import scatter_rows_
    torch.npu.set_device(0)
    slots=[3,1,4,2];layout=make_layout(4,2048,32768)
    # Fragment the original MLA allocation: physical order differs from both
    # logical token order and request slot order. Only every fourth page stores
    # compressed keys; compare the entire logical history after incremental writes.
    gen=torch.Generator().manual_seed(8801)
    original=(torch.randperm(512,generator=gen)[:4*32]+1).reshape(4,32).to('npu').int()
    compact=request_block_table(torch.tensor(slots,device='npu'),original,layout)
    full_cache=torch.zeros(513*64,128,device='npu',dtype=torch.bfloat16)
    small_cache=torch.zeros(layout.pages*64,128,device='npu',dtype=torch.bfloat16)
    idx=IndexerKPool.__new__(IndexerKPool);torch.nn.Module.__init__(idx);idx.index_kpool=4
    histories=[torch.randn(512,128,generator=gen).to(torch.bfloat16) for _ in slots]
    step={'prefill':256,'decode':1,'verify':4,'accepted':3}[phase]
    starts=[0,63,64,127,255,508]
    for row in range(4):
        for start in starts:
            end=min(512,start+step)
            pools=torch.arange(start,end,device='npu',dtype=torch.int64)
            values=histories[row][start:end].to('npu')
            a=idx._pooled_write_locs(original[row],pools)
            b=idx._pooled_write_locs(compact[row],pools)
            scatter_rows_(full_cache,a,values);scatter_rows_(small_cache,b,values)
        pools=torch.arange(512,device='npu',dtype=torch.int64)
        a=idx._pooled_write_locs(original[row],pools)
        b=idx._pooled_write_locs(compact[row],pools)
        torch.testing.assert_close(full_cache[a].cpu(),small_cache[b].cpu(),rtol=0,atol=0)
