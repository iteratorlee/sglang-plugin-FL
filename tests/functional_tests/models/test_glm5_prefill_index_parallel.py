"""Partitioning preserves each original causal GEMM tile and request order."""
from types import SimpleNamespace
import pytest
import torch


@pytest.mark.parametrize('world',[1,2,4,16])
@pytest.mark.parametrize('lengths,ends',[
    ([4096],[4096]), ([4096],[131072]), ([257,129,0,333],[65535,129,0,131071]),
    ([3],[3]), ([0,0],[4,128]), ([127,128,129],[256,256,258]),
])
def test_partition_preserves_original_tiles_and_rows(world,lengths,ends):
    from sglang_fl.models.glm_53_flash.prefill_index_parallel import partition_queries
    partitions,capacity=partition_queries(lengths,ends,world)
    expected=[];offset=0
    for req,(length,end) in enumerate(zip(lengths,ends)):
        first=end-length
        # Independently enumerate each token, grouping by absolute causal tile.
        current=[]
        for i in range(length):
            if current and (first+i)//128!=(first+current[-1])//128:
                expected.append((req,offset+current[0],len(current),first+current[-1]+1));current=[]
            current.append(i)
        if current:expected.append((req,offset+current[0],len(current),first+current[-1]+1))
        offset+=length
    actual=[];rows=[]
    for fragments in partitions:
        assert sum(f.length for f in fragments)<=capacity
        for f in fragments:
            rows.extend(range(f.offset,f.offset+f.length))
            first=f.sequence_end-f.length
            sub=[]
            for i in range(f.length):
                if sub and (first+i)//128!=(first+sub[-1])//128:
                    actual.append((f.request,f.offset+sub[0],len(sub),first+sub[-1]+1));sub=[]
                sub.append(i)
            if sub:actual.append((f.request,f.offset+sub[0],len(sub),first+sub[-1]+1))
    assert actual==expected
    assert rows==list(range(sum(lengths)))


@pytest.mark.parametrize('lengths,ends,world',[([-1],[0],16),([3],[2],16),([1],[],16),([1],[1],0)])
def test_partition_rejects_invalid_metadata(lengths,ends,world):
    from sglang_fl.models.glm_53_flash.prefill_index_parallel import partition_queries
    with pytest.raises(ValueError):partition_queries(lengths,ends,world)


def test_disabled_policy_needs_no_distributed_state(monkeypatch):
    from sglang_fl.models.glm_53_flash.prefill_index_parallel import enabled
    monkeypatch.delenv('SGLANG_FL_GLM53_PREFILL_INDEX_TP',raising=False)
    assert not enabled(torch.empty(4096,32,128),SimpleNamespace(extend_seq_lens_cpu=[4096]))
