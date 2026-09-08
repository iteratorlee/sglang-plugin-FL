#!/usr/bin/env python3
"""Check first-token parity for identical requests in one packed prefill."""

from __future__ import annotations

import argparse
import json
import urllib.request


def post_json(url: str, body: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        json.dumps(body).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31952")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output")
    args = parser.parse_args()

    body = {
        # A single batch API request guarantees that all identical prompts are
        # packed into one extend forward.  Independent concurrent HTTP calls
        # may be scheduled serially and cannot validate packed-prefill parity.
        "text": ["The capital of France is"] * args.batch_size,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "max_new_tokens": args.max_new_tokens,
            "sampling_seed": 20260904,
        },
    }

    responses = post_json(f"{args.base_url}/generate", body, args.timeout)
    if not isinstance(responses, list) or len(responses) != args.batch_size:
        raise RuntimeError(
            f"batch API returned {type(responses).__name__} with "
            f"length {len(responses) if isinstance(responses, list) else 'n/a'}"
        )
    ids = [response["output_ids"] for response in responses]
    texts = [response["text"] for response in responses]
    result = {
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "passed": len({tuple(item) for item in ids}) == 1
        and len(set(texts)) == 1
        and all(item for item in ids),
        "output_ids": ids,
        "texts": texts,
        "responses": responses,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
