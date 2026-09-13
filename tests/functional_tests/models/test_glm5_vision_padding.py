from types import SimpleNamespace
import pytest,torch
@pytest.mark.parametrize('dummy',[0,16,48])
@pytest.mark.parametrize('kind',['q_norm','k_norm','qkv_proj','proj'])
def test_dummy_heads_preserve_per_head_norm(dummy,kind):
 from sglang_fl.models.glm_53_flash.vision_weights import pad_vision_weight
 cfg=SimpleNamespace(vision_config=SimpleNamespace(num_dummy_heads=dummy,head_dim=64))
 shape=(64,) if 'norm' in kind else (3072,16) if kind=='qkv_proj' else (16,1024)
 x=torch.arange(torch.tensor(shape).prod()).float().reshape(shape)
 y=pad_vision_weight(cfg,'visual.blocks.0.attn.'+kind+'.weight',x)
 if 'norm' in kind:assert y is x
 elif kind=='qkv_proj':
  a=y.reshape(3,1024+dummy*64,16);b=x.reshape(3,1024,16)
  torch.testing.assert_close(a[:,:1024],b,rtol=0,atol=0);assert torch.count_nonzero(a[:,1024:])==0
 else:
  torch.testing.assert_close(y[:,:1024],x,rtol=0,atol=0);assert torch.count_nonzero(y[:,1024:])==0
