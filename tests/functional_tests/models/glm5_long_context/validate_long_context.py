#!/usr/bin/env python3
"""Measured long-prompt semantic tests, with durable per-request evidence.

Use the checkpoint tokenizer/template to size complete chat requests. The HTTP
server's usage is independently checked; merely submitting a large string does
not count as long-context coverage. Graph use must additionally be evidenced by
the corresponding server log. This script does not change server configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


CASES = {
    "three_position_retrieval": {
        "needles": [
            "AUTHORITATIVE RECORD: key ALPHA has access code amber-falcon-1847.",
            "AUTHORITATIVE RECORD: key BETA has access code silver-orchid-5923.",
            "AUTHORITATIVE RECORD: key GAMMA has access code cobalt-otter-7601.",
        ],
        "question": (
            "Return the three authoritative access codes. Output only one JSON "
            "object with exactly the string keys alpha, beta, gamma. Copy each "
            "code exactly, including the hyphens. Do not use unrelated records."
        ),
        "expected": {
            "alpha": "amber-falcon-1847",
            "beta": "silver-orchid-5923",
            "gamma": "cobalt-otter-7601",
        },
    },
    "cross_record_join": {
        "needles": [
            "AUTHORITATIVE RECORD: project Juniper's invoice ID is INV-8307.",
            "AUTHORITATIVE RECORD: invoice INV-8307 has a gross amount of 125 credits.",
            "AUTHORITATIVE RECORD: invoice INV-8307 receives a rebate of 17 credits.",
        ],
        "question": (
            "Find project Juniper's invoice and subtract its rebate from its "
            "gross amount. Output only JSON with exactly these keys: project "
            "(string), invoice (string), gross (integer), rebate (integer), net "
            "(integer). Use the authoritative records, not unrelated records."
        ),
        "expected": {
            "project": "Juniper",
            "invoice": "INV-8307",
            "gross": 125,
            "rebate": 17,
            "net": 108,
        },
    },
}


def make_noise(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    materials = ["quartz", "copper", "granite", "cotton", "cedar", "bronze"]
    sites = ["north", "south", "east", "west", "central", "coastal"]
    states = ["stored", "checked", "archived", "sorted", "received", "counted"]
    return [
        f"Unrelated record {index:06d}: {rng.choice(materials)} samples at the "
        f"{rng.choice(sites)} depot were {rng.choice(states)}; "
        f"batch {rng.randrange(10000, 99999)} contained {rng.randrange(20, 900)} units."
        for index in range(count)
    ]


def messages_for(noise: list[str], count: int, case: dict) -> list[dict]:
    lines = list(noise[:count])
    # Insert backwards so each location is relative to the unchanged corpus.
    for fraction, needle in reversed(list(zip((0.05, 0.5, 0.95), case["needles"]))):
        lines.insert(int(count * fraction), needle)
    text = (
        "Read the complete archive below. Most records are unrelated filler. "
        "The three AUTHORITATIVE RECORD lines contain the facts needed to "
        "answer the question after the archive.\n<archive>\n"
        + "\n".join(lines)
        + "\n</archive>\nQuestion: "
        + case["question"]
    )
    return [{"role": "user", "content": text}]


def sized_request(tokenizer, noise, target, case, template_kwargs):
    low, high = 0, len(noise)
    best = None
    while low <= high:
        middle = (low + high) // 2
        messages = messages_for(noise, middle, case)
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        if not isinstance(ids, list) or not all(
            isinstance(token, int) for token in ids
        ):
            raise TypeError("Chat template must return one flat token ID list")
        if len(ids) <= target:
            best = (messages, ids, middle)
            low = middle + 1
        else:
            high = middle - 1
    if best is None or len(best[1]) < target * 0.99:
        raise ValueError(f"Could not construct a prompt within 1% of {target} tokens")
    messages, ids, line_count = best
    content = messages[0]["content"]
    positions = []
    for needle in case["needles"]:
        preceding = content[: content.index(needle)]
        position = len(tokenizer.encode(preceding, add_special_tokens=False))
        positions.append({"text": needle, "token_offset_approx": position})
    return messages, {
        "target_prompt_tokens": target,
        "local_prompt_tokens": len(ids),
        "prompt_ids_sha256": hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "noise_records": line_count,
        "needle_positions": positions,
    }


def post_json(url, body, timeout):
    request = urllib.request.Request(
        url,
        json.dumps(body, ensure_ascii=False).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_answer(content):
    content = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        content = match.group(1)
    return json.loads(content)


def save(path, result):
    # Atomic replacement makes results available after each completed request,
    # including when a later request fails or the run is interrupted.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--model-path", default="/models/GLM-5.3-Flash-BF16")
    parser.add_argument("--model-kind", choices=["hy4", "glm"], default="glm")
    parser.add_argument("--lengths", type=int, nargs="+", default=[32768, 131072])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.lengths) < 512 or args.repeats < 1:
        parser.error("lengths must be >=512 and repeats >=1")
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer_config = json.loads(
        (Path(args.model_path) / "tokenizer_config.json").read_text()
    )
    if tokenizer_config.get("tokenizer_class") in {
        "TokenizersBackend",
        "PreTrainedTokenizerFast",
    }:
        # Both checkpoints carry a complete tokenizer.json. AutoTokenizer in
        # the image tries to parse their new model config before selecting this
        # generic fast tokenizer; direct loading avoids any model registration
        # or NPU initialization in this independent CPU-only test client.
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
    template_path = Path(args.model_path) / "chat_template.jinja"
    if template_path.exists():
        tokenizer.chat_template = template_path.read_text()
    template_kwargs = {
        "reasoning_effort": "no_think" if args.model_kind == "hy4" else "low"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Each record is comfortably longer than 16 tokens for either checkpoint;
    # avoid initially tokenizing a corpus dozens of times larger than needed.
    noise = make_noise(max(args.lengths) // 16 + 256, args.seed)
    result = {
        "model": args.model,
        "model_path": args.model_path,
        "base_url": args.base_url,
        "dry_run": args.dry_run,
        "passed": False,
        "graph_evidence": "Correlate these request IDs/times with the server decode graph log.",
        "cases": [],
    }
    save(args.output, result)
    for length in args.lengths:
        for name in args.cases:
            case = CASES[name]
            messages, metadata = sized_request(
                tokenizer, noise, length, case, template_kwargs
            )
            body = {
                "model": args.model,
                "messages": messages,
                "temperature": 0,
                "top_p": 1,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "chat_template_kwargs": template_kwargs,
            }
            payload_path = (
                args.output.parent / f"{args.output.stem}_{length}_{name}_request.json"
            )
            save(payload_path, body)
            row = {
                "name": name,
                **metadata,
                "payload_path": str(payload_path),
                "payload_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                "expected": case["expected"],
                "runs": [],
            }
            result["cases"].append(row)
            save(args.output, result)
            if args.dry_run:
                print(json.dumps({"name": name, **metadata}), flush=True)
                continue
            for repeat in range(args.repeats):
                started = time.time()
                run = {"repeat": repeat, "started_unix": started, "pass": False}
                try:
                    response = post_json(
                        args.base_url.rstrip("/") + "/v1/chat/completions",
                        body,
                        args.timeout,
                    )
                    run["response"] = response
                    choice = response["choices"][0]
                    content = choice["message"].get("content") or ""
                    try:
                        answer = parse_answer(content)
                    except (ValueError, TypeError):
                        answer = None
                    usage = response.get("usage", {})
                    actual = usage.get("prompt_tokens", 0)
                    completed = usage.get("completion_tokens", 0)
                    run.update(
                        answer=answer,
                        semantic=answer == case["expected"],
                        measured_long_prompt=(length * 0.99 <= actual <= length + 64),
                        actual_prompt_tokens=actual,
                        continuous_decode=completed
                        >= case.get("minimum_completion_tokens", 16),
                        completion_tokens=completed,
                        normal_stop=choice.get("finish_reason") == "stop",
                    )
                    run["pass"] = all(
                        run[key]
                        for key in (
                            "semantic",
                            "measured_long_prompt",
                            "continuous_decode",
                            "normal_stop",
                        )
                    )
                except Exception as exc:
                    run["error"] = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, urllib.error.HTTPError):
                        run["error_body"] = exc.read().decode(errors="replace")
                finally:
                    run["elapsed_seconds"] = time.time() - started
                    row["runs"].append(run)
                    save(args.output, result)
                    print(
                        json.dumps({"length": length, "case": name, **run}), flush=True
                    )
                if not run["pass"]:
                    return 1
            contents = [
                run["response"]["choices"][0]["message"]["content"]
                for run in row["runs"]
            ]
            row["deterministic_text"] = len(set(contents)) == 1
            row["pass"] = (
                all(run["pass"] for run in row["runs"]) and row["deterministic_text"]
            )
            save(args.output, result)
            if not row["pass"]:
                return 1
    result["passed"] = not args.dry_run and all(row["pass"] for row in result["cases"])
    save(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
