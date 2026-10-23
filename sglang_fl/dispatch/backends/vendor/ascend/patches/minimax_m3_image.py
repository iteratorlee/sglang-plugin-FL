# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Keep the current MiniMax-M3.1 image processor on CPU on Ascend.

SGLang's NPU preprocessing hook recognizes ``MiniMaxVLProcessor``. The
checkpoint's Transformers processor is named ``MiniMaxM3VLProcessor`` instead,
so SGLang sends its unpatched 10-dimensional patch extraction to the NPU,
which supports at most eight dimensions. The upstream processor succeeds on
CPU; its output is transferred to the model device by the normal MM path.
"""

import hashlib
import json
import logging
import math
import os
from contextvars import ContextVar
from functools import wraps

logger = logging.getLogger(__name__)

_NATIVE_RESIZE_ENV = "SGLANG_FL_ASCEND_MINIMAX_NATIVE_IMAGE_RESIZE"
_native_image_only_cpu = ContextVar("fl_minimax_native_image_only_cpu", default=False)


def _native_image_resize_enabled() -> bool:
    return os.environ.get(_NATIVE_RESIZE_ENV, "0") == "1"


def _native_image_size(height: int, width: int, factor: int, max_long_side: int):
    """Native MiniMax image geometry, independently checked against NV usage.

    Preserve the two ceil-based minimum-size steps before grid alignment.
    Nearest-grid rounding or an area cap of 672**2 changes the visual tokens.
    """
    if min(height, width, factor) <= 0:
        raise ValueError("MiniMax image dimensions and alignment must be positive")
    if isinstance(max_long_side, bool) or not isinstance(max_long_side, int):
        raise ValueError("max_long_side_pixel must be a positive integer")
    cap = max_long_side - max_long_side % factor
    if cap < factor:
        raise ValueError("max_long_side_pixel is smaller than the image alignment")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("MiniMax image aspect ratio must not exceed 200")
    if height * width < 168**2:
        scale = math.sqrt(168**2 / (height * width))
        height, width = math.ceil(height * scale), math.ceil(width * scale)
    if min(height, width) < 168:
        scale = 168 / min(height, width)
        height, width = math.ceil(height * scale), math.ceil(width * scale)
    h = ((height + factor - 1) // factor) * factor
    w = ((width + factor - 1) // factor) * factor
    if h > cap or w > cap:
        scaled_w = min(w * cap // w, w * cap // h)
        scaled_h = min(h * cap // w, h * cap // h)
        if min(scaled_h, scaled_w) <= 0:
            raise ValueError("MiniMax image cap is too small for its aspect ratio")
        h = ((scaled_h + factor - 1) // factor) * factor
        w = ((scaled_w + factor - 1) // factor) * factor
    if min(h, w) <= 0 or max(h, w) > cap:
        raise ValueError("Invalid aligned MiniMax image dimensions")
    return h, w


def _native_image_limit(image_data) -> int:
    def get(name):
        return image_data.get(name) if isinstance(image_data, dict) else getattr(image_data, name, None)

    explicit = get("max_long_side_pixel")
    if explicit is not None:
        return explicit
    return {"low": 672, "high": 3584}.get(get("detail"), 2016)


def patch_minimax_m3_native_image_resize() -> None:
    """Opt into native image sizing without changing video/embedding inputs."""
    if not _native_image_resize_enabled():
        return
    import inspect
    import asyncio

    from PIL import Image

    from sglang.srt.multimodal.processors.base_processor import build_processor_fingerprint
    from sglang.srt.multimodal.processors.minimax_m3_vl import MiniMaxM3VLProcessor

    cls = MiniMaxM3VLProcessor
    if getattr(cls, "_sglang_fl_minimax_m3_native_resize_patch", False):
        return
    original_init = cls.__init__
    original_load = cls.load_mm_data
    original_counts = cls.resolve_image_token_counts
    original_process = cls.process_mm_data
    load_signature = inspect.signature(original_load)
    process_signature = inspect.signature(original_process)
    if "image_data" not in load_signature.parameters:
        raise RuntimeError("Unsupported MiniMax load_mm_data signature")

    @wraps(original_init)
    def patched_init(self, hf_config, server_args, _processor, *args, **kwargs):
        original_init(self, hf_config, server_args, _processor, *args, **kwargs)
        self._sglang_fl_native_image_resize = _native_image_resize_enabled()
        if self._sglang_fl_native_image_resize:
            self.image_config = dict(self.image_config, do_resize=False)
            if self.mm_preprocess_cache.enabled or self.uses_media_artifacts_without_cache:
                self.processor_fingerprint = build_processor_fingerprint(
                    self, hf_config, extra={"sglang_fl_minimax_native_image_resize": "v1"}
                )

    @wraps(original_load)
    async def patched_load(self, *args, **kwargs):
        output = await original_load(self, *args, **kwargs)
        if not getattr(self, "_sglang_fl_native_image_resize", False) or not output.images:
            return output
        image_data = load_signature.bind(self, *args, **kwargs).arguments.get("image_data") or []
        if len(output.images) != len(image_data):
            raise ValueError("MiniMax loaded image count does not match image metadata")
        factor = self._processor.image_processor.patch_size * self.spatial_merge_size
        def resize_images():
            images = []
            for image, descriptor in zip(output.images, image_data):
                # Keep preprocessed dictionaries and their feature ownership.
                if isinstance(image, dict):
                    images.append(image)
                    continue
                if not isinstance(image, Image.Image):
                    raise TypeError("Native MiniMax image resizing requires a PIL image")
                h, w = _native_image_size(
                    image.height, image.width, factor, _native_image_limit(descriptor)
                )
                images.append(
                    image if (w, h) == image.size
                    else image.resize((w, h), Image.Resampling.BICUBIC)
                )
            return images

        # The loader's existing thread pool also handles PIL decode; avoid
        # blocking the tokenizer event loop or pickling closures to its CPU pool.
        output.images = await asyncio.get_running_loop().run_in_executor(
            self.io_executor, resize_images
        )
        return output

    @wraps(original_process)
    def patched_process(self, *args, **kwargs):
        bound = process_signature.bind(self, *args, **kwargs).arguments
        image_only = (
            getattr(self, "_sglang_fl_native_image_resize", False)
            and bool(bound.get("images"))
            and not bound.get("videos")
            and not bound.get("audios")
        )
        token = _native_image_only_cpu.set(image_only)
        try:
            return original_process(self, *args, **kwargs)
        finally:
            _native_image_only_cpu.reset(token)

    @wraps(original_counts)
    def patched_counts(self, images):
        if not getattr(self, "_sglang_fl_native_image_resize", False):
            return original_counts(self, images)
        patch = self._processor.image_processor.patch_size
        factor = patch * self.spatial_merge_size
        counts = []
        for image in images:
            if not isinstance(image, Image.Image) or image.width % factor or image.height % factor:
                raise ValueError("MiniMax image token counting requires aligned PIL images")
            counts.append((image.height // patch) * (image.width // patch) // self.spatial_merge_size**2)
        return counts

    cls.__init__ = patched_init
    cls.load_mm_data = patched_load
    cls.resolve_image_token_counts = patched_counts
    cls.process_mm_data = patched_process
    cls._sglang_fl_minimax_m3_native_resize_patch = True
    logger.info("Enabled native MiniMax image sizing and aligned token counts")


def patch_minimax_m3_image_preprocess_device() -> None:
    patch_minimax_m3_native_image_resize()
    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

    original = BaseMultimodalProcessor._fast_image_processor_device
    if getattr(original, "_sglang_fl_minimax_m3_native_cpu_patch", False):
        return

    @wraps(original)
    def patched(self, processor):
        name = type(processor).__name__
        if name == "MiniMaxM3VLProcessor" or (
            name == "MiniMaxVLProcessor" and _native_image_only_cpu.get()
        ):
            if not getattr(self, "_sglang_fl_minimax_m3_image_cpu_logged", False):
                logger.info("Using CPU preprocessing for %s on NPU", name)
                self._sglang_fl_minimax_m3_image_cpu_logged = True
            return "cpu"
        return original(self, processor)

    patched._sglang_fl_minimax_m3_image_cpu_patch = True
    patched._sglang_fl_minimax_m3_native_cpu_patch = True
    BaseMultimodalProcessor._fast_image_processor_device = patched


def patch_minimax_m3_vision_rope_compile() -> None:
    """Use eager MiniMax vision RoPE with FlagTree's Triton on Ascend.

    The upstream function is torch.compile-decorated. Torch NPU's Inductor
    cache imports ``triton_key``, which FlagTree's Triton module does not
    provide. The undecorated function performs the same RoPE operations and
    can run in the normal vision forward pass without Inductor.
    """
    from sglang.srt.models import minimax_vl_common

    compiled = minimax_vl_common._minimax_rope_applier
    eager = getattr(compiled, "__wrapped__", None)
    if eager is None:
        return
    if getattr(eager, "_sglang_fl_minimax_m3_vision_rope_patch", False):
        return

    eager._sglang_fl_minimax_m3_vision_rope_patch = True
    minimax_vl_common._minimax_rope_applier = eager
    logger.info("Using eager MiniMax vision RoPE on Ascend")


def patch_minimax_m3_image_token_boundaries() -> None:
    """Preserve the HF image framing when expanding native token prompts.

    The generic retokenization avoidance path repeats only the image token.
    MiniMax's HF processor also inserts a start/end token around every image;
    without those tokens adjacent images collapse into a single MM span.
    Keep the caller's text tokens verbatim and restore only missing framing.
    """
    from sglang.srt.multimodal.processors.minimax_m3_vl import MiniMaxM3VLProcessor

    original = MiniMaxM3VLProcessor._expand_input_ids
    if getattr(original, "_sglang_fl_minimax_m3_image_boundaries_patch", False):
        return

    @wraps(original)
    def patched(self, original_ids, counts, placeholder_token_id):
        if placeholder_token_id != self.IM_TOKEN_ID:
            return original(original_ids, counts, placeholder_token_id)
        # Retain the upstream validation (including placeholder/count mismatch).
        original(original_ids, counts, placeholder_token_id)
        rebuilt = []
        image_index = 0
        for index, token_id in enumerate(original_ids):
            if token_id != placeholder_token_id:
                rebuilt.append(token_id)
                continue
            if index == 0 or original_ids[index - 1] != self.IM_START_TOKEN_ID:
                rebuilt.append(self.IM_START_TOKEN_ID)
            rebuilt.extend([placeholder_token_id] * counts[image_index])
            if (
                index + 1 == len(original_ids)
                or original_ids[index + 1] != self.IM_END_TOKEN_ID
            ):
                rebuilt.append(self.IM_END_TOKEN_ID)
            image_index += 1
        return rebuilt

    patched._sglang_fl_minimax_m3_image_boundaries_patch = True
    MiniMaxM3VLProcessor._expand_input_ids = patched
    logger.info("Preserving MiniMax image boundaries in native token prompts")


def patch_minimax_m3_image_grid_cache_key() -> None:
    """Scope raw image cache identities to the grid used by vision RoPE.

    Equal flattened patches can describe different spatial grids. Feature-only
    hashes alias these images even when their placeholder counts match.
    """
    import torch

    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalInputFormat,
    )
    from sglang.srt.multimodal.processors.minimax_m3_vl import MiniMaxM3VLProcessor

    original = MiniMaxM3VLProcessor._postprocess_mm_items_before_transport
    if getattr(original, "_sglang_fl_minimax_m3_grid_cache_patch", False):
        return
    state_key = "_sglang_fl_minimax_m3_grid_cache"

    @wraps(original)
    def patched(self, mm_items, *, images):
        items = original(self, mm_items, images=images)
        for item in items:
            feature = item.feature
            grid = item.model_specific_data.get("image_grid_thw")
            if (
                item.modality != Modality.IMAGE
                or item.format != MultimodalInputFormat.NORMAL
                or item.precomputed_embeddings is not None
                or not isinstance(feature, torch.Tensor)
                or not feature.is_cpu
                or not isinstance(grid, torch.Tensor)
                or not grid.is_cpu
                or grid.ndim != 2
                or grid.shape[1] != 3
            ):
                continue
            item.set_pad_value()
            if item.hash is None:
                continue
            previous = item.model_specific_data.get(state_key)
            base_hash = (
                previous["base"]
                if isinstance(previous, dict) and item.hash == previous.get("scoped")
                else item.hash
            )
            payload = {
                "version": "minimax-m3-image-grid-v1",
                "feature_hash": base_hash,
                "feature_dtype": str(feature.dtype),
                "feature_shape": list(feature.shape),
                "image_grid_thw": grid.tolist(),
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).digest()
            scoped_hash = int.from_bytes(digest[:8], "big")
            item.set_hash(scoped_hash)
            item.model_specific_data[state_key] = {
                "base": base_hash,
                "scoped": scoped_hash,
            }
        return items

    patched._sglang_fl_minimax_m3_grid_cache_patch = True
    MiniMaxM3VLProcessor._postprocess_mm_items_before_transport = patched
    logger.info("Including MiniMax image grids in multimodal cache identities")
