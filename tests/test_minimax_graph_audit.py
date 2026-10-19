# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
# SPDX-License-Identifier: Apache-2.0
"""Test the prospective evidence rule without importing a device framework."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "sglang_fl/dispatch/backends/vendor/ascend/patches/minimax_m3_graph_audit.py"
spec = importlib.util.spec_from_file_location("audit_target", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def evidence(modes=(2, 2, 2, 2), tokens=(16, 16, 16, 16), rank=0):
    rows = [[n, n, int(m in (2, 4)), int(m in (1, 3)), 1, m, 0, 0]
            for m, n in zip(modes, tokens)]
    vote = all(row[2] for row in rows)
    mode = modes[rank // 4]
    actual = vote and mode in (2, 4)
    return dict(matrix=rows, tokens=tokens, rank=rank, mode=mode,
                vote=vote, extend=any(row[3] for row in rows),
                scope_ok=True, association_ok=True, completed=True,
                actual=actual, eligible=int(actual) if mode in (2, 4) else -1,
                calls=int(mode in (2, 4)), replays=int(actual),
                batch_size=tokens[rank // 4] if mode != 4 else 0,
                bucket=16 if actual else 0, seq=1, proof_seq=1, wall_ns=1)


class GraphAuditTest(unittest.TestCase):
    def test_global_decode_requires_actual_replay(self):
        state = evidence()
        self.assertEqual(audit.classify(state), 1)
        state.update(actual=False, eligible=0, replays=0)
        self.assertEqual(audit.classify(state), 4)

    def test_same_round_peer_prefill_is_required(self):
        state = evidence((2, 1, 4, 2), (16, 8192, 0, 8))
        self.assertEqual(audit.classify(state), 2)
        for field, wrong in (("association_ok", False), ("tokens", (16, 16, 0, 8)),
                             ("vote", True), ("extend", False), ("calls", 2),
                             ("replays", 1), ("completed", False)):
            bad = copy.deepcopy(state)
            bad[field] = wrong
            self.assertEqual(audit.classify(bad), 4, field)

    def test_prefill_and_idle_are_distinct(self):
        state = evidence((1, 2, 4, 2), (8192, 16, 0, 16))
        self.assertEqual(audit.classify(state), 3)
        idle = evidence((2, 2, 4, 2), (16, 8, 0, 4), rank=8)
        self.assertEqual(audit.classify(idle), 1)
        idle["tokens"] = (16, 8, 1, 4)
        self.assertEqual(audit.classify(idle), 4)

    def test_missing_and_tampered_proofs_rejected(self):
        for matrix in ((), [[0] * 7] * 4, [[0] * 8] * 4):
            state = evidence()
            state["matrix"] = matrix
            self.assertEqual(audit.classify(state), 4)

        for field, value in (("scope_ok", False), ("bucket", 12), ("eligible", 0)):
            state = evidence()
            state[field] = value
            self.assertEqual(audit.classify(state), 4)

    def test_json_roundtrip_retains_prefill_and_mixed_proof(self):
        for state, category in ((evidence((1, 2, 4, 2), (8192, 16, 0, 16)), 3),
                                (evidence((2, 1, 4, 2), (16, 8192, 0, 8)), 2)):
            self.assertEqual(audit.classify(json.loads(json.dumps(state))), category)

    def test_record_is_closed_and_unique(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SGLANG_FL_ASCEND_GRAPH_AUDIT_DIR": directory}):
            audit._writer = None
            try:
                audit._record(evidence())
                raw = (Path(directory) / "rank-00.bin").read_bytes()
                self.assertEqual(len(raw), audit.RECORD.size)
                self.assertEqual(audit.RECORD.unpack(raw)[8], 1)
                self.assertEqual((Path(directory) / "rank-00.jsonl").read_bytes(), b"")
                for fd in audit._writer[2:]:
                    os.close(fd)
                audit._writer = None
                with self.assertRaises(FileExistsError):
                    audit._record(evidence())
            finally:
                audit._writer = None

    def test_disabled_install_imports_no_framework(self):
        with patch.dict(os.environ, {"SGLANG_FL_ASCEND_GRAPH_AUDIT": "0"}):
            audit.patch_minimax_m3_graph_audit()
        self.assertFalse(audit._installed)


if __name__ == "__main__":
    unittest.main()
