"""Enable the audited SG startup warmup without importing SG or vendor kernels."""

import ast
import hashlib
import importlib.util
import os
from pathlib import Path

FLAG = "SGLANG_GLM53_PD_WARM_ALLOCATOR"
# Audited SG reference aa232279da28c4b8827b52ec43b544d79e03cec8.
# These two files own scratch safety, variant coverage, and the vendor call ABI.
SOURCE_HASHES = {
    "srt/hardware_backend/npu/attention/glm53/pd_allocator_warmup.py": (
        "c9ac20cf3e756f2ee0a57c371c4572fb925c09c66cb05076b4ac6b6d3d789912"
    ),
    "srt/hardware_backend/npu/allocator_npu.py": (
        "de63fdd354c691ef07f59b946746da948179182f4c1668fec44996330211036d"
    ),
}
_HOOK_PREFIX = """
if (
    _is_npu
    and not self.is_draft_worker
    and get_disagg().disaggregation_mode == "decode"
    and os.getenv("SGLANG_GLM53_PD_WARM_ALLOCATOR", "0") == "1"
):
    from sglang.srt.hardware_backend.npu.attention.glm53.pd_allocator_warmup import maybe_warm_pd_allocators
    maybe_warm_pd_allocators(self)
capture = capture_cuda_graphs(
    model_runner=self, capture_decode_cuda_graph=capture_decode_cuda_graph
)
"""


def _source_root():
    # Top-level find_spec does not execute the package or its submodules.
    spec = importlib.util.find_spec("sglang")
    roots = list(spec.submodule_search_locations or ()) if spec else []
    if len(roots) != 1:
        raise RuntimeError("Allocator warmup requires one inspectable SG source root")
    return Path(roots[0]).resolve(strict=True)


def _validate_source(root):
    hashes = {}
    for relative, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Unaudited allocator warmup source: {relative}: {actual}"
            )
        hashes[relative] = actual
    tree = ast.parse((root / "srt/model_executor/model_runner.py").read_text())
    methods = [
        method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "ModelRunner"
        for method in cls.body
        if isinstance(method, ast.FunctionDef) and method.name == "init_cuda_graphs"
    ]
    expected = [ast.dump(node) for node in ast.parse(_HOOK_PREFIX).body]
    if len(methods) != 1 or [ast.dump(n) for n in methods[0].body[:2]] != expected:
        raise RuntimeError("Audited allocator warmup hook must precede graph capture")
    return hashes


def configure():
    """Default on only in native mode, after FlagTree validation by the caller.

    Explicit 0 opts out, including on an incompatible SG tree. Enabled mode
    fails before changing the environment if source compatibility is unknown.
    This requests startup work; it does not claim any kernel has been warmed.
    """
    value = os.environ.get(FLAG, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG} must be exactly 0 or 1")
    proof = {
        "env": FLAG,
        "value": value,
        "defaulted_in_this_process": FLAG not in os.environ,
        "status": "disabled" if value == "0" else "requested",
    }
    if value == "0":
        return proof
    root = _source_root()
    proof["source_sha256"] = _validate_source(root)
    proof["sglang_source"] = str(root)
    proof["hook"] = "ModelRunner.init_cuda_graphs:before_capture"
    os.environ.setdefault(FLAG, "1")
    return proof
