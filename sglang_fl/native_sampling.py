"""Fix the audited target-only sampler's U=0 acceptance boundary in memory.

This sampler bug occurs under both compilers; GPQA attribution is unproven.
Install during native plugin loading, before warmup or graph capture. No tensor
work or launch wrapper is added; unknown source/ABI fails closed.
"""

import ast
import hashlib
import importlib
import inspect
import linecache
import threading
from pathlib import Path

TARGET = "sgl_kernel_npu.sample.tree_speculative_sampling_target_only"
KERNEL = "_tree_target_only_accept_kernel"
ENTRY = "tree_speculative_sampling_target_only"
SOURCE_SHA256 = "e6a36161730dbf47641d5a3b5a8ce3e3f787ea6a271a1aa35d955e2c81912831"
_LOCK = threading.Lock()


def _strict_source(source):
    tree = ast.parse(source)
    comparisons = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, ast.LtE) for op in node.ops)
    ]
    expression = "coin <= prob_acc / threshold_acc"
    if (
        len(comparisons) != 1
        or ast.get_source_segment(source, comparisons[0]) != expression
        or source.count(expression) != 1
    ):
        raise RuntimeError("Unsupported target-only acceptance comparison")
    comparisons[0].ops = [ast.Lt()]
    patched = source.replace(expression, "coin < prob_acc / threshold_acc", 1)
    if ast.dump(ast.parse(patched)) != ast.dump(tree):
        raise RuntimeError("Target-only patch changed more than one comparison")
    return patched


def install(module=None):
    """Replace only the audited module's accept JIT; preserve public aliases."""
    with _LOCK:
        return _install(
            module if module is not None else importlib.import_module(TARGET)
        )


def _install(module):
    import triton

    try:
        raw = Path(inspect.getsourcefile(module)).read_bytes()
    except (OSError, TypeError) as exc:
        raise RuntimeError("Cannot verify target-only sampling source") from exc
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Unsupported target-only sampling source SHA256")
    source = raw.decode("utf-8")
    definitions = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    kernel = getattr(module, KERNEL, None)
    entry = getattr(module, ENTRY, None)
    if (
        not isinstance(kernel, triton.JITFunction)
        or not inspect.isfunction(kernel.fn)
        or kernel.fn.__globals__ is not vars(module)
    ):
        raise RuntimeError("Unsupported target-only accept JIT")
    original = definitions[KERNEL]
    patched = _strict_source(original)
    current = kernel.src.strip()
    installed = getattr(kernel, "_fl_native_sampling", None) == SOURCE_SHA256
    if current != (patched if installed else original):
        raise RuntimeError("Unsupported target-only accept JIT source")
    if (
        not inspect.isfunction(entry)
        or entry.__globals__ is not vars(module)
        or inspect.getsource(entry).strip() != definitions[ENTRY]
    ):
        raise RuntimeError("Unsupported target-only sampling entry point")

    # Keep formatting/comments and all globals. Separate exec locals prevent
    # publishing anything until every guard passes. linecache lets public jit()
    # inspect the real replacement source and compute a fresh matching hash.
    virtual_source = patched + "\n\n" + definitions[ENTRY] + "\n"
    digest = hashlib.sha256(virtual_source.encode()).hexdigest()
    filename = f"<sglang_fl.native_sampling.{digest}>"
    linecache.cache[filename] = (
        len(virtual_source),
        None,
        virtual_source.splitlines(keepends=True),
        filename,
    )
    namespace = {}
    exec(
        compile(virtual_source, filename, "exec", dont_inherit=True),
        vars(module),
        namespace,
    )
    fn = namespace[KERNEL]
    signature = inspect.signature(fn)
    constexprs = [
        i
        for i, param in enumerate(signature.parameters.values())
        if param.annotation is module.tl.constexpr
    ]
    if (
        inspect.signature(kernel.fn) != signature
        or kernel.signature != signature
        or tuple(kernel.arg_names) != tuple(signature.parameters)
        or list(kernel.constexprs) != constexprs
        or inspect.signature(entry) != inspect.signature(namespace[ENTRY])
    ):
        raise RuntimeError("Unsupported target-only sampling function signature/ABI")
    if installed:
        return "already_installed"
    # Use the actual (verified) JIT source, without private source setters or
    # reusing a compiled object's cache. Public jit() retains default options
    # from the audited bare @triton.jit decorator.
    if _strict_source(kernel.src.strip()) != patched:
        raise RuntimeError("Target-only accept JIT source changed during installation")
    replacement = triton.jit(fn)
    if (
        replacement.src.strip() != patched
        or replacement.signature != signature
        or tuple(replacement.arg_names) != tuple(kernel.arg_names)
        or list(replacement.constexprs) != constexprs
    ):
        raise RuntimeError("Replacement target-only JIT source/ABI mismatch")
    replacement._fl_native_sampling = SOURCE_SHA256
    setattr(module, KERNEL, replacement)
    return "patched"


def configure():
    return {"status": install(), "source_sha256": SOURCE_SHA256, "target": TARGET}
