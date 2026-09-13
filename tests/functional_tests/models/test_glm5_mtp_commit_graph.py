"""Captured state copies must track new slots, accepted lengths and scratch values."""
import pytest
import torch

@pytest.mark.parametrize('bs',[1,2,3])
@pytest.mark.parametrize('steps',[2,4,5])
def test_accepted_state_graph_dynamic_inputs(bs,steps):
    import torch_npu
    from sglang_fl.models.glm_53_flash.mtp_commit_graph import AcceptedStateGraph
    torch.npu.set_device(0)
    ops=[]
    for layers,shape,dtype,kind in [(2,(2,8,16),torch.float32,0),(2,(96,3),torch.bfloat16,0),
                                   (1,(4,128),torch.bfloat16,1),(1,(4,128),torch.float32,1)]:
        dst=torch.randn(layers,8,*shape,device='npu',dtype=dtype)
        src=torch.randn(layers,bs+1,steps,*shape,device='npu',dtype=dtype)
        ops.append((dst,src,kind))
    indices=torch.tensor([2,5,0][:bs],device='npu',dtype=torch.int32)
    req=torch.tensor([5,2,0][:bs],device='npu',dtype=torch.int64)
    accepted=torch.tensor([1,steps-1,-1][:bs],device='npu',dtype=torch.int64)
    def expected_copy():
        expected=[]
        for dst,src,kind in ops:
            result=dst.cpu();source=src.cpu();slots=(req if kind else indices).cpu().tolist()
            for i,(slot,step) in enumerate(zip(slots,accepted.cpu().tolist())):
                if slot>0 and 0<=step<steps:result[:,slot]=source[:,i,step]
            expected.append(result)
        return expected
    expected=expected_copy()
    graph=AcceptedStateGraph(ops,indices,req,accepted);torch.npu.synchronize()
    for (dst,_,_),wanted in zip(ops,expected):torch.testing.assert_close(dst.cpu(),wanted,rtol=0,atol=0)
    for i in range(8):
        for _,src,_ in ops:src.normal_()
        indices.copy_(torch.tensor(([5,2,0] if i%2 else [2,5,0])[:bs],device='npu',dtype=torch.int32))
        req.copy_(torch.tensor(([2,5,0] if i%2 else [5,2,0])[:bs],device='npu'))
        accepted.copy_(torch.tensor([i%(steps+2)-1,(i+1)%(steps+2)-1,-1][:bs],device='npu'))
        expected=expected_copy();graph.replay(indices,req,accepted);torch.npu.synchronize()
        for (dst,_,_),wanted in zip(ops,expected):torch.testing.assert_close(dst.cpu(),wanted,rtol=0,atol=0)
