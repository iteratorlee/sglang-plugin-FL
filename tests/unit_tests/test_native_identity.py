"""CPU validation of wheel and editable provenance, without importing Triton."""

import base64
import hashlib
import json
import sys
from importlib import metadata
from types import SimpleNamespace

import pytest

from sglang_fl import native_identity as identity


@pytest.fixture
def installation(tmp_path, monkeypatch):
    for key in (
        "SGLANG_FL_EXPECT_FLAGTREE_VERSION",
        "SGLANG_FL_EXPECT_FLAGTREE_COMMIT",
        "SGLANG_FL_FLAGTREE_SOURCE",
    ):
        monkeypatch.delenv(key, raising=False)

    def make(mode="editable"):
        source = tmp_path / "FlagTree"
        site = tmp_path / "site"
        root = source / "python" if mode == "editable" else site
        module = root / "triton/__init__.py"
        extension = root / "triton/_C/libtriton.so"
        for path in (module, extension):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"identity test fixture, not executable")
        info = site / "flagtree.dist-info"
        info.mkdir(parents=True)
        # Fixture suffix is deliberately NOT a claim about the parent's real build.
        version = (
            "0.7.0+fixture.git2b5b813.ascend"
            if mode == "editable"
            else "0.7.0+ascend3.5"
        )
        (info / "METADATA").write_text(f"Name: flagtree\nVersion: {version}\n")
        (info / "direct_url.json").write_text(
            json.dumps({"url": source.as_uri(), "dir_info": {"editable": True}})
            if mode == "editable"
            else "{}"
        )
        if mode == "wheel":
            records = []
            for path in (module, extension):
                sha = (
                    base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
                    .decode()
                    .rstrip("=")
                )
                records.append(
                    f"{path.relative_to(site)},sha256={sha},{path.stat().st_size}\n"
                )
            (info / "RECORD").write_text("".join(records))
        else:
            monkeypatch.setenv("SGLANG_FL_EXPECT_FLAGTREE_VERSION", version)
            monkeypatch.setenv("SGLANG_FL_FLAGTREE_SOURCE", str(source))
        dist = metadata.Distribution.at(info)
        monkeypatch.setattr(identity.metadata, "distribution", lambda name: dist)
        triton = SimpleNamespace(__version__="3.5.1", __file__=str(module))
        lib = SimpleNamespace(__file__=str(extension))
        monkeypatch.setitem(sys.modules, "triton", triton)
        monkeypatch.setitem(sys.modules, "triton._C.libtriton", lib)
        git = {"commit": identity.FLAGTREE_COMMIT, "diff": ""}

        def run_git(root, *args):
            if args == ("rev-parse", "--show-toplevel"):
                return str(source)
            if args == ("rev-parse", "HEAD"):
                return git["commit"]
            assert args == ("diff", "HEAD", "--name-only", "--no-ext-diff")
            return git["diff"]

        monkeypatch.setattr(identity, "_git", run_git)
        return SimpleNamespace(
            source=source,
            module=module,
            extension=extension,
            info=info,
            git=git,
            triton=triton,
            lib=lib,
            version=version,
        )

    return make


@pytest.mark.parametrize("mode", ["editable", "wheel"])
def test_accepts_exact_supported_installation(installation, mode):
    env = installation(mode)
    result = identity.validate_flagtree()
    assert result["flagtree"] == env.version
    assert result["flagtree_install"] == mode
    assert (
        result["triton_extension_sha256"]
        == hashlib.sha256(env.extension.read_bytes()).hexdigest()
    )
    if mode == "editable":
        assert result["flagtree_commit"] == "2b5b81320c18dbf4008cf6ad14d3c246bd74f7ff"


@pytest.mark.parametrize(
    "problem",
    [
        "missing_version",
        "wrong_version",
        "commit",
        "commit_override",
        "dirty",
        "directurl",
        "not_editable",
        "module",
        "extension",
        "triton",
    ],
)
def test_editable_rejects_each_identity_mismatch(
    installation, monkeypatch, tmp_path, problem
):
    env = installation()
    foreign = tmp_path / "foreign.py"
    foreign.write_text("foreign")
    if problem == "missing_version":
        monkeypatch.delenv("SGLANG_FL_EXPECT_FLAGTREE_VERSION")
    elif problem == "wrong_version":
        monkeypatch.setenv("SGLANG_FL_EXPECT_FLAGTREE_VERSION", "0.7.0+different")
    elif problem == "commit":
        env.git["commit"] = "0" * 40
    elif problem == "commit_override":
        monkeypatch.setenv("SGLANG_FL_EXPECT_FLAGTREE_COMMIT", "0" * 40)
    elif problem == "dirty":
        env.git["diff"] = "python/triton/runtime/jit.py"
    elif problem in ("directurl", "not_editable"):
        (env.info / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": tmp_path.as_uri(),
                    "dir_info": {"editable": problem != "not_editable"},
                }
            )
        )
    elif problem == "module":
        env.triton.__file__ = str(foreign)
    elif problem == "extension":
        env.lib.__file__ = str(foreign)
    elif problem == "triton":
        env.triton.__version__ = "3.2.0"
    with pytest.raises(RuntimeError):
        identity.validate_flagtree()


@pytest.mark.parametrize(
    "problem", ["version", "override", "shadow", "overwrite", "record"]
)
def test_wheel_does_not_accept_nearby_versions_or_overlays(
    installation, monkeypatch, tmp_path, problem
):
    env = installation("wheel")
    if problem == "version":
        (env.info / "METADATA").write_text("Name: flagtree\nVersion: 0.7.1a1\n")
    elif problem == "override":
        monkeypatch.setenv("SGLANG_FL_EXPECT_FLAGTREE_VERSION", "0.7.1a1")
    elif problem == "shadow":
        foreign = tmp_path / "foreign.so"
        foreign.write_bytes(env.extension.read_bytes())
        env.lib.__file__ = str(foreign)
    elif problem == "overwrite":
        env.extension.write_bytes(b"wrong extension")
    elif problem == "record":
        (env.info / "RECORD").unlink()
    with pytest.raises(RuntimeError):
        identity.validate_flagtree()
