"""Add the witnessed single-server GLM decode topology to the native gate."""

import hashlib
import importlib
import inspect
import os
from pathlib import Path

FLAG = "SGLANG_FL_GLM53_DP2TP4_PREFIX_COMPAT"
TARGET = "sglang.srt.disaggregation.glm53_decode_prefix"
OLD = ((16, 8, 16), (8, 4, 8))
NEW = OLD + ((8, 2, 8),)


def install(module):
    """Change one immutable topology constant; retain all other native guards.

    Updating the existing function's code preserves aliases imported before the
    plugin. No wrapper executes per request, and no tensor operation changes.
    """
    fn = module.glm53_pd_decode_prefix_profile
    if not inspect.isfunction(fn) or tuple(inspect.signature(fn).parameters) != (
        "cfg",
        "model_config",
    ):
        raise RuntimeError("Unknown GLM prefix-profile callable")
    constants = fn.__code__.co_consts
    if constants.count(NEW) == 1 and constants.count(OLD) == 0:
        return "already_supported"
    if constants.count(OLD) != 1 or constants.count(NEW) != 0:
        raise RuntimeError("Unknown GLM prefix topology gate; review before patching")
    constants = tuple(NEW if value == OLD else value for value in constants)
    fn.__code__ = fn.__code__.replace(co_consts=constants)
    return "patched"


def configure():
    value = os.environ.get(FLAG, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG} must be exactly 0 or 1")
    if value == "0":
        return {"status": "disabled", "env": FLAG}
    module = importlib.import_module(TARGET)
    source = Path(inspect.getsourcefile(module))
    return {
        "status": install(module),
        "env": FLAG,
        "topologies": NEW,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "policy": "only topology constant extended; original model/state/MTP guards retained",
    }
