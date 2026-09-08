#!/usr/bin/env python3
"""Deterministic semantic smoke test for the Hy4-preview SGLang endpoint."""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request


# Keep a completion-path smoke test in addition to the model's instruct/chat
# path.  These prompts place a checkable answer in a short continuation.
CASES = (
    # The single-request graph service is intentionally slow. Keep each case
    # only as long as needed to expose its checkable answer, but repeat every
    # case three times so a one-off correct output cannot hide instability.
    # The first generated token is a leading space, so this case needs several
    # tokens to expose the integer rather than falsely failing on truncation.
    ("addition_1", "1 + 1 =", r"(?:^|\D)2(?:\D|$)", 4),
    ("addition_2", "Two plus two equals", r"(?:^|\W)four(?:\W|$)", 1),
    ("knowledge", "The capital of France is", r"Paris", 16),
    ("language", "床前明月光，", r"疑是地上霜", 5),
)
REPEAT_COUNT = 3
LONG_CONTEXT_LENGTHS = (33, 37, 100)
DIRECT_NO_THINK_PROMPT_IDS = (
    120000, 13251, 120001, 120039, 59785, 287, 62, 4138, 506, 25,
    3202, 32980, 1121, 120025, 120000, 3717, 120001, 3462, 341, 269,
    8778, 299, 12749, 30, 6144, 430, 269, 7370, 2046, 13, 120025,
    120000, 611, 10372, 120001, 120029, 120030,
)


def post_json(url: str, body: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        json.dumps(body, ensure_ascii=False).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31932")
    parser.add_argument("--model", default="Hy4-preview")
    parser.add_argument(
        "--model-path", default="/models/Hy4-preview-W8A8-linear-moe"
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output")
    args = parser.parse_args()

    # Exercise the native endpoint without depending on Transformers knowing
    # HYV4's new config class. SGLang 0.5.11 calls the seed field
    # ``sampling_seed``.
    chat_repeat_body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Answer with the city name.",
            }
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 4,
        "seed": 20260904,
        "chat_template_kwargs": {"reasoning_effort": "no_think"},
    }
    rows = []
    for case_id, prompt, pattern, max_new_tokens in CASES:
        body = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "top_p": 1,
                "max_new_tokens": max_new_tokens,
                "sampling_seed": 20260904,
            },
        }
        responses = [
            post_json(f"{args.base_url}/generate", body, args.timeout)
            for _ in range(REPEAT_COUNT)
        ]
        texts = [response["text"] for response in responses]
        ids = [response.get("output_ids") for response in responses]
        semantic = [
            re.search(pattern, text, re.IGNORECASE) is not None
            for text in texts
        ]
        rows.append(
            {
                "id": case_id,
                "prompt": prompt,
                "pattern": pattern,
                "max_new_tokens": max_new_tokens,
                "semantic": semantic,
                "pass": all(semantic),
                "deterministic": len(set(texts)) == 1
                and len({tuple(item) for item in ids}) == 1,
                "responses": responses,
            }
        )

    chat_repeat_responses = [
        post_json(
            f"{args.base_url}/v1/chat/completions",
            chat_repeat_body,
            args.timeout,
        )
        for _ in range(REPEAT_COUNT)
    ]
    chat_repeat_texts = [
        response["choices"][0]["message"]["content"].strip()
        for response in chat_repeat_responses
    ]
    chat_semantic = all(
        re.search(r"Paris", text, re.IGNORECASE) is not None
        for text in chat_repeat_texts
    )
    direct_body = {
        "input_ids": list(DIRECT_NO_THINK_PROMPT_IDS),
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "max_new_tokens": 4,
            "sampling_seed": 20260904,
        },
    }
    direct_responses = [
        post_json(f"{args.base_url}/generate", direct_body, args.timeout)
        for _ in range(REPEAT_COUNT)
    ]
    direct_texts = [response["text"] for response in direct_responses]
    direct_ids = [response.get("output_ids") for response in direct_responses]
    direct_semantic = all(
        re.search(r"Paris", text, re.IGNORECASE) is not None
        for text in direct_texts
    )
    direct_deterministic = len(set(direct_texts)) == 1 and len(
        {tuple(item) for item in direct_ids}
    ) == 1
    long_context = []
    for length in LONG_CONTEXT_LENGTHS:
        response = post_json(
            f"{args.base_url}/generate",
            {
                "input_ids": [220] * length,
                "sampling_params": {
                    "temperature": 0,
                    "top_p": 1,
                    "max_new_tokens": 1,
                    "sampling_seed": 20260904,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
            },
            args.timeout,
        )
        output_logprobs = response.get("meta_info", {}).get(
            "output_token_logprobs", []
        )
        score = output_logprobs[0][0] if output_logprobs else None
        long_context.append(
            {
                "prompt_tokens": length,
                "finite_output_logprob": isinstance(score, (int, float))
                and math.isfinite(score),
                "response": response,
            }
        )
    result = {
        "passed": sum(row["pass"] for row in rows),
        "total": len(rows),
        "deterministic": all(row["deterministic"] for row in rows),
        "chat_deterministic": len(set(chat_repeat_texts)) == 1,
        "chat_semantic": chat_semantic,
        "chat_repeat_responses": chat_repeat_responses,
        "direct_semantic": direct_semantic,
        "direct_deterministic": direct_deterministic,
        "direct_repeat_responses": direct_responses,
        "long_context": long_context,
        "cases": rows,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
    return (
        0
        if result["passed"] == result["total"]
        and result["deterministic"]
        and result["chat_deterministic"]
        and result["chat_semantic"]
        and result["direct_deterministic"]
        and result["direct_semantic"]
        and all(row["finite_output_logprob"] for row in result["long_context"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
