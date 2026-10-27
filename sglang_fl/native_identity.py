"""Read-only identity checks for pinned FlagTree wheels and editable builds."""

import base64
import hashlib
import importlib
import json
import os
import subprocess
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlsplit

FLAGTREE_COMMIT = "2b5b81320c18dbf4008cf6ad14d3c246bd74f7ff"
FLAGTREE_WHEEL_VERSION = "0.7.0+ascend3.5"
TRITON_VERSION = "3.5.1"


def _git(root, *args):
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_editable(dist, direct_url, module_path, extension_path):
    expected_version = os.environ.get("SGLANG_FL_EXPECT_FLAGTREE_VERSION", "").strip()
    if not expected_version:
        raise RuntimeError(
            "Set SGLANG_FL_EXPECT_FLAGTREE_VERSION to the exact metadata version "
            "reported by the completed, pinned editable build"
        )
    expected_commit = os.environ.get(
        "SGLANG_FL_EXPECT_FLAGTREE_COMMIT", FLAGTREE_COMMIT
    )
    if expected_commit != FLAGTREE_COMMIT:
        raise RuntimeError(
            f"Only the audited FlagTree commit {FLAGTREE_COMMIT} is supported"
        )
    root = Path(os.environ.get("SGLANG_FL_FLAGTREE_SOURCE", "/sgl-workspace/FlagTree"))
    if not root.is_absolute():
        raise RuntimeError("SGLANG_FL_FLAGTREE_SOURCE must be an absolute source path")
    root = root.resolve(strict=True)
    if dist.version != expected_version:
        raise RuntimeError(
            f"Expected flagtree metadata {expected_version!r}, got {dist.version!r}"
        )
    url = urlsplit(direct_url.get("url", ""))
    if (
        url.scheme != "file"
        or url.netloc not in ("", "localhost")
        or url.query
        or url.fragment
    ):
        raise RuntimeError(
            "FlagTree editable URL must identify a local source directory"
        )
    if Path(unquote(url.path)).resolve(strict=True) != root:
        raise RuntimeError(
            "FlagTree editable direct_url does not match the expected source"
        )
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise RuntimeError(
            "FlagTree editable source must be the pinned repository root"
        )
    commit = _git(root, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError(f"FlagTree commit mismatch: {commit} != {expected_commit}")
    # Include staged and unstaged tracked changes. Build outputs may be untracked.
    if _git(root, "diff", "HEAD", "--name-only", "--no-ext-diff"):
        raise RuntimeError("FlagTree tracked source differs from the pinned commit")

    if not module_path.is_relative_to(root) or module_path != (
        root / "python/triton/__init__.py"
    ).resolve(strict=True):
        raise RuntimeError(
            f"Triton module is shadowed outside the pinned editable: {module_path}"
        )
    if not extension_path.is_relative_to(root):
        raise RuntimeError(
            f"Triton compiled extension is outside the pinned source/build: {extension_path}"
        )
    return {
        "flagtree_install": "editable",
        "flagtree_commit": commit,
        "flagtree_source": str(root),
    }


def _validate_wheel(dist, module_path, extension_path):
    expected = os.environ.get(
        "SGLANG_FL_EXPECT_FLAGTREE_VERSION", FLAGTREE_WHEEL_VERSION
    )
    if expected != FLAGTREE_WHEEL_VERSION or dist.version != FLAGTREE_WHEEL_VERSION:
        raise RuntimeError(
            f"Expected flagtree wheel exactly {FLAGTREE_WHEEL_VERSION}, got {dist.version}"
        )
    # Verify both loaded files belong to this wheel, not an old PYTHONPATH overlay.
    # RECORD digest verification also detects overwritten package/extension files.
    for path in (module_path, extension_path):
        matches = [
            entry
            for entry in (dist.files or ())
            if Path(dist.locate_file(entry)).resolve() == path
        ]
        if (
            len(matches) != 1
            or matches[0].hash is None
            or matches[0].hash.mode != "sha256"
        ):
            raise RuntimeError(
                f"Loaded Triton file is not owned by the FlagTree wheel RECORD: {path}"
            )
        digest = (
            base64.urlsafe_b64encode(bytes.fromhex(_sha256(path))).decode().rstrip("=")
        )
        if digest != matches[0].hash.value:
            raise RuntimeError(f"FlagTree wheel RECORD hash mismatch: {path}")
    return {"flagtree_install": "wheel"}


def validate_flagtree():
    """Exact wheel pin, or exact observed editable version plus commit/path pin."""
    dist = metadata.distribution("flagtree")
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    triton = importlib.import_module("triton")
    if triton.__version__ != TRITON_VERSION:
        raise RuntimeError(
            f"Expected triton module {TRITON_VERSION}, got {triton.__version__}"
        )
    module_path = Path(triton.__file__).resolve(strict=True)
    extension = importlib.import_module("triton._C.libtriton")
    extension_path = Path(extension.__file__).resolve(strict=True)
    if direct_url.get("dir_info", {}).get("editable") is True:
        identity = _validate_editable(dist, direct_url, module_path, extension_path)
    else:
        identity = _validate_wheel(dist, module_path, extension_path)
    return {
        **identity,
        "flagtree": dist.version,
        "flagtree_direct_url": direct_url,
        "triton_module": triton.__version__,
        "triton_path": str(module_path),
        "triton_extension_path": str(extension_path),
        "triton_extension_sha256": _sha256(extension_path),
    }
