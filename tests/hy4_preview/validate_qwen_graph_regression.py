"""Run a small deterministic Qwen graph-mode regression against an HTTP server."""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path


def _post(base_url, endpoint, payload):
    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.load(response)
        status = response.status
    return {
        "status": status,
        "elapsed_seconds": time.monotonic() - started,
        "response": body,
    }


def _raw_text(response):
    return response["text"]


def _raw_token_ids(response):
    records = response.get("meta_info", {}).get("output_token_logprobs") or []
    return [record[1] for record in records]


def _chat_text(response):
    return response["choices"][0]["message"]["content"]


def _semantic_ok(name, text):
    if "france" in name:
        return "paris" in text.lower()
    if name == "chat_addition":
        return re.search(r"(?<!\d)5(?!\d)", text) is not None
    raise ValueError(f"unknown case {name}")


def _run_case(base_url, case, repeats):
    rows = []
    for replay in range(repeats):
        result = _post(base_url, case["endpoint"], case["payload"])
        response = result["response"]
        text = case["extract_text"](response)
        rows.append(
            {
                "replay": replay,
                "status": result["status"],
                "elapsed_seconds": result["elapsed_seconds"],
                "text": text,
                "token_ids": (
                    _raw_token_ids(response)
                    if case["endpoint"] == "/generate"
                    else None
                ),
                "semantic_ok": _semantic_ok(case["name"], text),
                "response": response,
            }
        )
    texts = [row["text"] for row in rows]
    token_ids = [row["token_ids"] for row in rows]
    deterministic_text = len(set(texts)) == 1
    deterministic_token_ids = (
        len({tuple(value) for value in token_ids}) == 1
        if token_ids[0]
        else None
    )
    passed = (
        all(row["status"] == 200 and row["semantic_ok"] for row in rows)
        and deterministic_text
        and deterministic_token_ids is not False
    )
    return {
        "name": case["name"],
        "endpoint": case["endpoint"],
        "repeats": repeats,
        "deterministic_text": deterministic_text,
        "deterministic_token_ids": deterministic_token_ids,
        "passed": passed,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31962")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_sampling = {
        "temperature": 0,
        "max_new_tokens": 16,
    }
    chat_sampling = {
        "temperature": 0,
        "max_tokens": 32,
    }
    cases = [
        {
            "name": "raw_france",
            "endpoint": "/generate",
            "payload": {
                "text": "The capital of France is",
                "sampling_params": raw_sampling,
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 0,
            },
            "extract_text": _raw_text,
        },
        {
            "name": "chat_addition",
            "endpoint": "/v1/chat/completions",
            "payload": {
                "model": "/models/Qwen3-0.6B",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 2 + 3? Reply with only the final number.",
                    }
                ],
                "chat_template_kwargs": {"enable_thinking": False},
                **chat_sampling,
            },
            "extract_text": _chat_text,
        },
        {
            "name": "chat_france",
            "endpoint": "/v1/chat/completions",
            "payload": {
                "model": "/models/Qwen3-0.6B",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "What is the capital of France? "
                            "Reply with only the city name."
                        ),
                    }
                ],
                "chat_template_kwargs": {"enable_thinking": False},
                **chat_sampling,
            },
            "extract_text": _chat_text,
        },
    ]
    results = [_run_case(args.base_url, case, args.repeats) for case in cases]
    server_log = args.server_log.read_text(errors="replace") if args.server_log else ""
    decode_graph_lines = [
        line
        for line in server_log.splitlines()
        if "Decode batch" in line and "cuda graph: True" in line
    ]
    metadata = json.loads(args.metadata.read_text()) if args.metadata else {}
    output = {
        "metadata": metadata,
        "decode_cuda_graph_true": bool(decode_graph_lines),
        "decode_cuda_graph_lines": decode_graph_lines,
        "cases": results,
    }
    output["passed"] = output["decode_cuda_graph_true"] and all(
        result["passed"] for result in results
    )
    rendered = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
