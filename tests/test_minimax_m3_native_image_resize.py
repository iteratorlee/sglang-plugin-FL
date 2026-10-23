# Author: liyan <yan.l@pku.edu.cn>
"""Native image geometry and processor boundaries without an NPU dependency."""
import asyncio
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from PIL import Image

PATH = Path(__file__).parents[1] / 'sglang_fl/dispatch/backends/vendor/ascend/patches/minimax_m3_image.py'
SPEC = importlib.util.spec_from_file_location('native_image_patch_under_test', PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
FLAG = 'SGLANG_FL_ASCEND_MINIMAX_NATIVE_IMAGE_RESIZE'


class GeometryTests(unittest.TestCase):
    def test_native_usage_vectors(self):
        # Independently observed NV image-token counts, not nearest-grid resize.
        for h,w,tokens in [(256,320,120), (180,320,84), (300,400,165),
                           (1102,1469,2120), (1080,1920,2691), (150,200,63)]:
            hh,ww = MOD._native_image_size(h,w,28,2016)
            self.assertEqual((hh//28)*(ww//28), tokens)

    def test_cap_and_limits(self):
        self.assertEqual(MOD._native_image_size(1000,2000,28,300),(140,280))
        self.assertEqual(MOD._native_image_limit({'detail':'low'}),672)
        self.assertEqual(MOD._native_image_limit(SimpleNamespace(detail='high')),3584)
        self.assertEqual(MOD._native_image_limit('data:image/png;base64,...'),2016)
        self.assertEqual(MOD._native_image_limit({'detail':'low','max_long_side_pixel':448}),448)

    def test_invalid_geometry_is_explicit(self):
        for params in [(0,200,28,2016), (1,1000,28,2016), (200,200,28,True),
                       (200,200,28,10), (200,200,28,2016.5)]:
            with self.assertRaises(ValueError):
                MOD._native_image_size(*params)


class ProcessorTests(unittest.TestCase):
    def setUp(self):
        class Base:
            def _fast_image_processor_device(self, processor):
                return 'npu'

        class Processor(Base):
            def __init__(self,hf_config,server_args,_processor,*args,**kwargs):
                self._processor = _processor
                self.image_config = {'custom_option':7}
                self.mm_preprocess_cache = SimpleNamespace(enabled=bool(server_args.cache))
                self.uses_media_artifacts_without_cache = False
                self.processor_fingerprint = 'before'
                self.spatial_merge_size = 2
                self.loaded_images = []
                self.videos = object()
                self.io_executor = ThreadPoolExecutor(max_workers=1)

            async def load_mm_data(self,prompt,image_data=None,video_data=None,**kwargs):
                return SimpleNamespace(images=list(self.loaded_images), videos=self.videos)

            def resolve_image_token_counts(self,images):
                return [-1]

            def process_mm_data(self,input_text,images=None,videos=None,audios=None,**kwargs):
                return self._fast_image_processor_device(self._processor)

        modules = {}
        for name in ('sglang','sglang.srt','sglang.srt.multimodal','sglang.srt.multimodal.processors'):
            module = ModuleType(name)
            module.__path__ = []
            modules[name] = module
        base = ModuleType('sglang.srt.multimodal.processors.base_processor')
        base.BaseMultimodalProcessor = Base
        base.build_processor_fingerprint = lambda obj,cfg,**kw: dict(obj.image_config,extra=kw.get('extra'))
        minimax = ModuleType('sglang.srt.multimodal.processors.minimax_m3_vl')
        minimax.MiniMaxM3VLProcessor = Processor
        modules[base.__name__] = base
        modules[minimax.__name__] = minimax
        self.modules = patch.dict(sys.modules,modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.flag = patch.dict(os.environ,{FLAG:'1'})
        self.flag.start()
        self.addCleanup(self.flag.stop)
        self.cls = Processor
        self.base = Base
        hf_type = type('MiniMaxVLProcessor',(),{})
        self.hf = hf_type()
        self.hf.image_processor = SimpleNamespace(patch_size=14)

    def instance(self,cache=False):
        obj = self.cls(hf_config=object(),server_args=SimpleNamespace(cache=cache),_processor=self.hf)
        self.addCleanup(obj.io_executor.shutdown,wait=True)
        return obj

    def test_idempotence_keyword_init_and_cache_fingerprint(self):
        MOD.patch_minimax_m3_image_preprocess_device()
        functions = self.cls.__init__,self.cls.load_mm_data,self.cls.resolve_image_token_counts
        MOD.patch_minimax_m3_image_preprocess_device()
        self.assertEqual(functions,(self.cls.__init__,self.cls.load_mm_data,self.cls.resolve_image_token_counts))
        obj = self.instance(cache=True)
        self.assertEqual(obj.image_config,{'custom_option':7,'do_resize':False})
        self.assertEqual(obj.processor_fingerprint['extra'],{'sglang_fl_minimax_native_image_resize':'v1'})
        hf_type = type('MiniMaxVLProcessor',(),{})
        self.assertEqual(obj._fast_image_processor_device(hf_type()),'npu')
        self.assertEqual(obj._fast_image_processor_device(object()),'npu')
        self.assertEqual(obj.process_mm_data('prompt',images=[object()]),'cpu')
        self.assertEqual(obj.process_mm_data('prompt',videos=[object()]),'npu')
        self.assertEqual(obj.process_mm_data('prompt',images=[object()],videos=[object()]),'npu')
        self.assertEqual(obj.process_mm_data('prompt',images=[object()],audios=[object()]),'npu')
        self.assertFalse(MOD._native_image_only_cpu.get())

    def test_images_dicts_videos_and_counts(self):
        MOD.patch_minimax_m3_image_preprocess_device()
        obj = self.instance()
        precomputed = {'format':'precomputed_embedding','feature':object()}
        obj.loaded_images = [Image.new('RGB',(320,256)),precomputed]
        result = asyncio.run(obj.load_mm_data('prompt',image_data=['data',{}],video_data='video'))
        self.assertEqual(result.images[0].size,(336,280))
        self.assertIs(result.images[1],precomputed)
        self.assertIs(result.videos,obj.videos)
        self.assertEqual(obj.resolve_image_token_counts([result.images[0]]),[120])
        with self.assertRaises(ValueError):
            obj.resolve_image_token_counts([Image.new('RGB',(300,300))])

    def test_existing_and_disabled_instances_retain_original_behavior(self):
        obj_before = self.instance()
        MOD.patch_minimax_m3_image_preprocess_device()
        os.environ[FLAG] = '0'
        obj_after = self.instance()
        image = Image.new('RGB',(320,256))
        for obj in (obj_before,obj_after):
            obj.loaded_images = [image]
            result = asyncio.run(obj.load_mm_data('prompt',image_data=['data']))
            self.assertIs(result.images[0],image)
            self.assertEqual(obj.resolve_image_token_counts([image]),[-1])

    def test_mismatched_metadata_fails_without_swapping_images(self):
        MOD.patch_minimax_m3_image_preprocess_device()
        obj = self.instance()
        obj.loaded_images = [Image.new('RGB',(320,256)),Image.new('RGB',(400,300))]
        with self.assertRaises(ValueError):
            asyncio.run(obj.load_mm_data('prompt',image_data=['one']))


if __name__ == '__main__':
    unittest.main()
