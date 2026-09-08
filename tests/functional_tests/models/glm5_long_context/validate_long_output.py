#!/usr/bin/env python3
"""Long input plus a >=512-token exact ordered-copy answer.

Complements sparse retrieval/reasoning cases: requires 64 authoritative records
from three separated archive regions, a complete JSON answer and normal stop.
Defaults: 32K prompt, max 2048 generated tokens, two identical greedy runs.
"""

import random
import sys

import validate_long_context as suite


def main():
    rng = random.Random(20260905)
    names = ["cedar", "quartz", "amber", "silver", "bronze", "orchid"]
    codes = [
        f"ledger-{index:03d}-{rng.choice(names)}-item-{rng.randrange(1000, 9999)}"
        for index in range(1, 65)
    ]
    needles = []
    for group, (start, stop) in enumerate(((0, 21), (21, 43), (43, 64)), 1):
        entries = "; ".join(
            f"entry {index + 1:03d} = {codes[index]}" for index in range(start, stop)
        )
        needles.append(f"AUTHORITATIVE RECORD: ledger group {group}: {entries}.")
    suite.CASES = {
        "ordered_record_copy": {
            "needles": needles,
            "question": (
                "Copy all 64 ledger codes from authoritative groups 1, 2 and 3 "
                "in ascending entry order 001 through 064. Output only one JSON "
                "object with the sole key codes and an array of all 64 exact "
                "code strings. Preserve every digit and hyphen; do not omit "
                "entries, abbreviate, use ellipses or include the entry labels."
            ),
            "expected": {"codes": codes},
            "minimum_completion_tokens": 512,
        }
    }
    if "--max-tokens" not in sys.argv:
        sys.argv.extend(("--max-tokens", "2048"))
    if "--lengths" not in sys.argv:
        sys.argv.extend(("--lengths", "32768"))
    return suite.main()


if __name__ == "__main__":
    raise SystemExit(main())
