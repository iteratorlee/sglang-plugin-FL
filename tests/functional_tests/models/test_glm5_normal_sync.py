"""The installed-runtime workaround must not affect other models or LL decode."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest

@pytest.fixture
def mod():
    p=Path(__file__).parents[3]/'sglang_fl/models/glm_53_flash/normal_sync.py'
    spec=importlib.util.spec_from_file_location('glm53_normal_sync_test',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.fixture
def args(tmp_path,monkeypatch):
    (tmp_path/'config.json').write_text(json.dumps({'architectures':['Glm5NextForConditionalGeneration']}))
    monkeypatch.setenv('SGLANG_FL_GLM53_DEEPEP_SYNC','1')
    return SimpleNamespace(device='npu',tp_size=16,ep_size=16,nnodes=1,pp_size=1,
                           enable_dp_attention=False,quantization='modelslim',model_path=str(tmp_path))

@pytest.mark.parametrize('field,value', [('device','cuda'),('tp_size',8),('ep_size',8),('nnodes',2),('pp_size',2),('enable_dp_attention',True),('quantization','bf16')])
def test_other_runtime_untouched(mod,args,field,value):
    setattr(args,field,value);assert not mod.enabled(args)

def test_explicit_checkpoint_scope(mod,args,monkeypatch,tmp_path):
    assert mod.enabled(args)
    monkeypatch.delenv('SGLANG_FL_GLM53_DEEPEP_SYNC');assert not mod.enabled(args)
    monkeypatch.setenv('SGLANG_FL_GLM53_DEEPEP_SYNC','1')
    other=tmp_path/'other';other.mkdir();(other/'config.json').write_text('{"architectures":["DeepseekV3ForCausalLM"]}')
    args.model_path=str(other);assert not mod.enabled(args)

@pytest.mark.parametrize('active', [True,False])
def test_preserves_dispatch_result_and_arguments(mod,active):
    events=[];result=object();instance=object()
    def original(self,x,*,handle):
        assert self is instance and x==7 and handle==9
        events.append('dispatch');return result
    wrapped=mod.wrap_dispatch(original,lambda:active,lambda:events.append('sync'))
    assert wrapped(instance,7,handle=9) is result
    assert events==(['sync','dispatch','sync'] if active else ['dispatch'])

def test_failure_is_propagated_without_repeating_collective(mod):
    events=[]
    def original(*args,**kwargs):events.append('dispatch');raise RuntimeError('failure')
    wrapped=mod.wrap_dispatch(original,lambda:True,lambda:events.append('sync'))
    with pytest.raises(RuntimeError,match='failure'):wrapped(None)
    assert events==['sync','dispatch']
