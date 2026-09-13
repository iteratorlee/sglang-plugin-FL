"""CPU token-list oracle, noncontiguous metadata and changing NPU graph inputs."""
import pytest
import torch

@pytest.mark.parametrize('bs',[1,2,16,32,129])
@pytest.mark.parametrize('pools',[1,7,1024])
@pytest.mark.parametrize('dtype',[torch.int32,torch.int64])
def test_kpool_expand_dynamic_graph(bs,pools,dtype):
    import torch_npu
    from sglang_fl.models.glm_53_flash.kpool_expand_npu import expand_with_tail
    torch.npu.set_device(0)
    indices=torch.zeros(bs,pools*2,device='npu',dtype=dtype)[:,::2]
    positions=torch.zeros(bs*2,device='npu',dtype=torch.int64)[::2]
    expand_with_tail(indices,positions);torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):actual=expand_with_tail(indices,positions)
    lengths=[1,2,3,4,5,63,64,65,256,65536,131072,1048576]
    gen=torch.Generator().manual_seed(381+bs+pools)
    for replay in range(6):
        source=torch.randint(-2,32770,(bs,pools),generator=gen,dtype=dtype)
        seq=[lengths[(replay+j)%len(lengths)] for j in range(bs)]
        expected=[]
        for row,length in zip(source.tolist(),seq):
            tokens=[4*pool+offset if pool>=0 else -1 for pool in row for offset in range(4)]
            history=min(length-length%4,len(tokens));tokens.extend([-1]*3)
            tail_start=length-length%4
            for i in range(3):tokens[history+i]=tail_start+i if i<length%4 else -1
            expected.append([v if v>=0 else length for v in tokens])
        indices.copy_(source.to('npu'));positions.copy_(torch.tensor(seq,device='npu')-1)
        graph.replay();torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(),torch.tensor(expected,dtype=dtype),rtol=0,atol=0)
