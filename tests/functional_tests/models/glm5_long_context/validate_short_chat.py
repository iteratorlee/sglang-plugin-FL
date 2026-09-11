#!/usr/bin/env python3
"""Repeat short official-template chats and save full request/response evidence."""

import argparse
import json
import time
import urllib.request
from pathlib import Path


CASES = (
    ("17_plus_25", "What is 17 plus 25?", "42"),
    ("6_times_7", "What is 6 times 7?", "42"),
    ("10_minus_4", "What is 10 minus 4?", "6"),
    (
        "capital_france",
        "What is the capital of France?",
        "Paris",
    ),
)


def post(url, body, timeout):
    request = urllib.request.Request(
        url,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.load(response)


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31012")
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--reasoning-effort", choices=["low", "high", "max"], default="low"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "requested_runtime": {
            "cuda_graph": True,
            "cuda_graph_bs": [16],
            "use_flaggems": 1,
            "hccl_deterministic": True,
            "reasoning_parser": "glm45",
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": 512,
            "seed": 20260904,
        },
        "cases": [],
        "all_correct": False,
        "all_exact_content_replay": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, result)
    for name, prompt, expected in CASES:
        prompt += (
            " Reply with only the integer or city name in plain text."
            " Do not add Markdown, punctuation, or explanation."
        )
        case = {"name": name, "prompt": prompt, "expected": expected, "runs": []}
        result["cases"].append(case)
        for repetition in range(args.repeats):
            started = time.time()
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 512,
                "seed": 20260904,
                "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort},
            }
            status, response = post(
                args.base_url.rstrip("/") + "/v1/chat/completions",
                body,
                args.timeout,
            )
            choice = response["choices"][0]
            content = (choice["message"].get("content") or "").strip()
            row = {
                "repetition": repetition,
                "http_status": status,
                "finish_reason": choice.get("finish_reason"),
                "content": content,
                "reasoning_content": choice["message"].get("reasoning_content") or "",
                "correct": content == expected
                and choice.get("finish_reason") == "stop",
                "request": body,
                "elapsed_seconds": time.time() - started,
                "full_response": response,
            }
            case["runs"].append(row)
            save(args.output, result)
        case["all_correct"] = all(row["correct"] for row in case["runs"])
        case["exact_content_replay"] = (
            len({row["content"] for row in case["runs"]}) == 1
        )
        save(args.output, result)
    result["all_correct"] = all(case["all_correct"] for case in result["cases"])
    result["all_exact_content_replay"] = all(
        case["exact_content_replay"] for case in result["cases"]
    )
    save(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["all_correct"] and result["all_exact_content_replay"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
