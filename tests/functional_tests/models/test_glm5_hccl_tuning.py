import json
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize('world', [16,32])
@pytest.mark.parametrize('group', ['tp','moe_ep','dp',None])
@pytest.mark.parametrize('value', [None,'512','4096'])
def test_buffer_scope(tmp_path,monkeypatch,world,group,value):
    from sglang_fl.models.glm_53_flash.hccl_tuning import selected_config
    (tmp_path/'config.json').write_text(json.dumps({'architectures':['Glm5NextForConditionalGeneration']}))
    args=SimpleNamespace(device='npu',tp_size=world,ep_size=world,nnodes=world//16,
        pp_size=1,enable_dp_attention=False,enable_two_batch_overlap=False,
        quantization='modelslim',model_path=str(tmp_path))
    monkeypatch.delenv('SGLANG_FL_GLM53_TP_AIV',raising=False)
    monkeypatch.delenv('SGLANG_FL_GLM53_HCCL_BUFFER_MB',raising=False)
    if value is not None: monkeypatch.setenv('SGLANG_FL_GLM53_HCCL_BUFFER_MB',value)
    expected={'hccl_buffer_size':int(value)} if value and group in ('tp','moe_ep') else {}
    assert selected_config(args,group)==expected
    for key,replacement in [('device','cpu'),('tp_size',8),('nnodes',3),('pp_size',2),
        ('enable_dp_attention',True),('enable_two_batch_overlap',True),('quantization','fp8')]:
        changed=SimpleNamespace(**vars(args));setattr(changed,key,replacement)
        assert selected_config(changed,group)=={}

@pytest.mark.parametrize('value',['0','-1','4097','abc'])
def test_invalid_buffer_fails(tmp_path,monkeypatch,value):
    from sglang_fl.models.glm_53_flash.hccl_tuning import selected_config
    (tmp_path/'config.json').write_text(json.dumps({'architectures':['Glm5NextForConditionalGeneration']}))
    args=SimpleNamespace(device='npu',tp_size=16,ep_size=16,nnodes=1,pp_size=1,
        enable_dp_attention=False,enable_two_batch_overlap=False,quantization='modelslim',model_path=str(tmp_path))
    monkeypatch.setenv('SGLANG_FL_GLM53_HCCL_BUFFER_MB',value)
    with pytest.raises(ValueError):selected_config(args,'tp')

@pytest.mark.parametrize('group',['tp','moe_ep'])
@pytest.mark.parametrize('existing',[False,True])
def test_options_preserve_config_and_align_environment(tmp_path,monkeypatch,group,existing):
    import os
    from sglang_fl.models.glm_53_flash import hccl_tuning as m
    from sglang.srt.distributed import parallel_state
    from sglang.srt import server_args
    (tmp_path/'config.json').write_text(json.dumps({'architectures':['Glm5NextForConditionalGeneration']}))
    args=SimpleNamespace(device='npu',tp_size=16,ep_size=16,nnodes=1,pp_size=1,
        enable_dp_attention=False,enable_two_batch_overlap=False,quantization='modelslim',model_path=str(tmp_path))
    value=SimpleNamespace(hccl_config={'existing':42}) if existing else None
    calls=[]
    def original(name):calls.append(name);return value
    monkeypatch.setattr(parallel_state,'get_torch_distributed_pg_options',original)
    monkeypatch.setattr(parallel_state,'get_world_group',lambda:SimpleNamespace(rank=7))
    monkeypatch.setattr(server_args,'get_global_server_args',lambda:args)
    monkeypatch.setattr(m,'_PATCHED',False)
    monkeypatch.setenv('SGLANG_FL_GLM53_HCCL_BUFFER_MB','512')
    monkeypatch.setenv('HCCL_BUFFSIZE','4096')
    monkeypatch.setenv('SGLANG_FL_GLM53_AUDIT_DIR',str(tmp_path/'quant-audit'))
    m.patch_hccl_options();patched=parallel_state.get_torch_distributed_pg_options
    m.patch_hccl_options();assert parallel_state.get_torch_distributed_pg_options is patched
    result=patched(group)
    expected={'existing':42,'hccl_buffer_size':512} if existing else {'hccl_buffer_size':512}
    assert result.hccl_config==expected
    assert os.environ['HCCL_BUFFSIZE']=='512' and calls==[group]
    log=next((tmp_path/'hccl-options').glob('*.jsonl'))
    assert json.loads(log.read_text())['hccl_config']==expected
