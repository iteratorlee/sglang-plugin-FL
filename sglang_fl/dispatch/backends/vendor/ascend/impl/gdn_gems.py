# Copyright 2026 FlagOS Contributors
"""Call the installed FlagGems operator, never an external candidate module."""

import os

_audited_pids = set()


def fused_sigmoid_gating_delta_rule_update_gems(*args, **kwargs):
    from flag_gems.runtime.backend._ascend.fla import (
        fused_sigmoid_gating_delta_rule_update,
    )

    result = fused_sigmoid_gating_delta_rule_update(*args, **kwargs)
    directory = os.environ.get("SGLANG_FL_GEMS_GDN_AUDIT_DIR")
    if directory and os.getpid() not in _audited_pids:
        import json
        from pathlib import Path
        import subprocess
        import sys
        import torch
        import flag_gems

        root = Path(flag_gems.__file__).resolve().parents[2]
        module = fused_sigmoid_gating_delta_rule_update.__module__
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        )
        record = {
            "route": "sglang_fl.dispatch/vendor.ascend.gems",
            "rank": rank,
            "library": "flag_gems",
            "module": module,
            "module_file": str(Path(sys.modules[module].__file__).resolve()),
            "root": str(root),
            "revision": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "candidate_sha256": getattr(
                fused_sigmoid_gating_delta_rule_update,
                "__kg_gems_candidate_sha256__",
                None,
            ),
        }
        (path / f"rank-{rank}-pid-{os.getpid()}.json").write_text(
            json.dumps(record, indent=2)
        )
        _audited_pids.add(os.getpid())
    return result
