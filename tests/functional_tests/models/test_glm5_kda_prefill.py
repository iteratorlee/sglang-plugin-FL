"""Prefill head partitioning must preserve recurrent values and live state."""
import pytest
import torch


def tensors(length, scale=.4):
    import torch_npu
    torch.npu.set_device(0)
    torch.manual_seed(31415+length)
    sample=lambda *shape:(torch.randn(*shape,device='npu')*scale).bfloat16()
    return dict(q=sample(1,length,4,128), k=sample(1,length,4,128),
        v=sample(1,length,4,128), a=sample(1,length,4,128), b=sample(1,length,4),
        A_log=torch.linspace(-2.,.8,4,device='npu'),
        dt_bias=torch.randn(4,128,device='npu')*.5,
        initial_state_indices=torch.tensor([3],device='npu',dtype=torch.int32),
        cu_seqlens=torch.tensor([0,length],device='npu',dtype=torch.int32), lower_bound=-5.)


@pytest.mark.parametrize('length',[128,511,4096])
@pytest.mark.parametrize('scale',[.05,2.])
def test_prefill_matches_original_across_continuations(length,scale):
    from sglang_fl.models.glm_53_flash.kda_recurrent_npu import glm_kda_varlen_recurrent_npu as run
    args=tensors(length,scale)
    old=torch.randn(6,4,128,128,device='npu')*.2
    new=old.clone()
    for _ in range(4):
        for name in ('q','k','v','a','b'):
            args[name].copy_(torch.randn_like(args[name])*scale)
        expected=run(**args,initial_state_source=old,prefill=False)
        actual=run(**args,initial_state_source=new,prefill=True)
        torch.npu.synchronize()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(new,old,rtol=0,atol=0)


def test_prefill_dynamic_graph_slot_switch_and_padding():
    from sglang_fl.models.glm_53_flash.kda_recurrent_npu import glm_kda_varlen_recurrent_npu as run
    args=tensors(256)
    old=torch.randn(6,4,128,128,device='npu')*.2
    new=old.clone()
    run(**args,initial_state_source=new,prefill=True)
    torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):actual=run(**args,initial_state_source=new,prefill=True)
    for slot in (3,1,0,5,3):
        new.copy_(old)
        args['initial_state_indices'].fill_(slot)
        for name in ('q','k','v','a','b'):args[name].copy_(torch.randn_like(args[name])*.4)
        expected=run(**args,initial_state_source=old,prefill=False)
        graph.replay();torch.npu.synchronize()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(new,old,rtol=0,atol=0)


@pytest.mark.parametrize('length',[1,4,16,127])
def test_short_shapes_keep_existing_result(length):
    from sglang_fl.models.glm_53_flash.kda_recurrent_npu import glm_kda_varlen_recurrent_npu as run
    args=tensors(length)
    state=torch.randn(6,4,128,128,device='npu')*.2
    old,new=state.clone(),state.clone()
    expected=run(**args,initial_state_source=old)
    actual=run(**args,initial_state_source=new,prefill=True)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(new,old,rtol=0,atol=0)


def test_explicit_prefill_optout(monkeypatch):
    from sglang_fl.models.glm_53_flash.kda_recurrent_npu import glm_kda_varlen_recurrent_npu as run
    args=tensors(256)
    state=torch.randn(6,4,128,128,device='npu')*.2
    monkeypatch.setenv('SGLANG_FL_GLM53_KDA_PREFILL_HEADS','0')
    old,new=state.clone(),state.clone()
    torch.testing.assert_close(run(**args,initial_state_source=old),
        run(**args,initial_state_source=new,prefill=True),rtol=0,atol=0)
    torch.testing.assert_close(new,old,rtol=0,atol=0)
