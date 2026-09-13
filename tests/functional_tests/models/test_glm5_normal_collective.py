"""The alternate prefill transport must remain inside its supported GLM scope."""
import json
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize('override', [{},{'device':'cpu'},{'tp_size':8},{'ep_size':8},
    {'nnodes':2},{'pp_size':2},{'enable_dp_attention':True},{'enable_two_batch_overlap':True},
    {'enable_eplb':True},{'ep_num_redundant_experts':1},{'quantization':'fp8'},
    {'architecture':'UnrelatedModel'},{'environment':'0'}])
def test_normal_collective_scope(tmp_path,monkeypatch,override):
    from sglang_fl.models.glm_53_flash.normal_collective import enabled
    args=dict(device='npu',tp_size=16,ep_size=16,nnodes=1,pp_size=1,
              enable_dp_attention=False,enable_two_batch_overlap=False,enable_eplb=False,
              ep_num_redundant_experts=0,quantization='modelslim',model_path=str(tmp_path))
    arch=override.get('architecture','Glm5NextForConditionalGeneration')
    (tmp_path/'config.json').write_text(json.dumps({'architectures':[arch]}))
    monkeypatch.setenv('SGLANG_FL_GLM53_NORMAL_HCCL',override.get('environment','1'))
    args.update({k:v for k,v in override.items() if k not in ('architecture','environment')})
    assert enabled(SimpleNamespace(**args)) == (not override)
