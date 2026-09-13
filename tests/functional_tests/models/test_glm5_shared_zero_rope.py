"""A layer-contiguous shared placeholder must remain zero under actual KV writes."""
import pytest
import torch


@pytest.mark.parametrize('layers',[1,3,11])
def test_allocation_geometry(layers):
    from sglang_fl.models.glm_53_flash.shared_zero_rope import SharedZeroMode
    with SharedZeroMode(9) as mode:
        k=torch.zeros((layers,9,64,1,512),dtype=torch.bfloat16)
        v=torch.zeros((layers,9,64,1,64),dtype=torch.bfloat16)
        index=torch.zeros((layers,9,64,1,128),dtype=torch.bfloat16)
    assert mode.redirected==int(layers>1)
    assert all(v[i].is_contiguous() for i in range(layers))
    assert v.untyped_storage().nbytes()==9*64*64*2
    assert k.untyped_storage().nbytes()==layers*9*64*512*2
    assert index.untyped_storage().nbytes()==layers*9*64*128*2


@pytest.mark.parametrize('layers',[3,11])
@pytest.mark.parametrize('compact',[False,True])
def test_real_cache_setter_preserves_shared_zero(layers,compact):
    import torch_npu
    from contextlib import nullcontext
    from types import SimpleNamespace
    from sglang_fl.models.glm_53_flash import register
    from sglang_fl.models.glm_53_flash.shared_zero_rope import _SCOPE,patch_shared_zero_rope
    from sglang_fl.models.glm_53_flash.compact_index import IndexAllocationMode,make_layout
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    torch.npu.set_device(0)
    register.patch_npu_mla_zero_rope_cache()
    patch_shared_zero_rope()
    token=register._GLM_POOL_INDEX_HEAD_DIM.set(128);shared=_SCOPE.set(True)
    try:
        ctx=IndexAllocationMode(make_layout(1,512,4096)) if compact else nullcontext()
        with ctx:
            pool=NPUMLATokenToKVPool(size=4096,page_size=64,dtype=torch.bfloat16,
                kv_lora_rank=512,qk_rope_head_dim=0,index_head_dim=128,
                layer_num=layers,device='npu',enable_memory_saver=False,start_layer=0,end_layer=layers)
    finally:
        _SCOPE.reset(shared);register._GLM_POOL_INDEX_HEAD_DIM.reset(token)
    values=[torch.randn(4,512,device='npu',dtype=torch.bfloat16) for _ in range(layers)]
    loc=torch.tensor([0,63,127,4095],device='npu',dtype=torch.int32)
    def write():
        for i in range(layers):pool.set_kv_buffer(SimpleNamespace(layer_id=i),loc,values[i],None)
    write();torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):write()
    for rep in range(4):
        loc.copy_(torch.tensor([1+rep,64+rep,256+rep,2048+rep],device='npu',dtype=torch.int32))
        for v in values:v.normal_()
        graph.replay();torch.npu.synchronize()
        for i in range(layers):
            torch.testing.assert_close(pool.get_key_buffer(i).view(-1,512)[loc.long()].cpu(),values[i].cpu(),rtol=0,atol=0)
            assert pool.get_value_buffer(i).is_contiguous()
            assert pool.get_value_buffer(i).data_ptr()==pool.get_value_buffer(0).data_ptr()
        assert not pool.v_buffer[0].cpu().any()
    physical=sum(x.untyped_storage().nbytes() for x in (pool.k_buffer,pool.v_buffer,pool.index_k_buffer))
    assert pool.get_kv_size_bytes()==physical
