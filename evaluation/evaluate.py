import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evaluation" / "dataset.jsonl"
RESULTS_DIR = ROOT / "results"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000").rstrip("/")
API_URL = os.getenv("API_URL", "http://localhost:8080").rstrip("/")

DEFAULT_MODELS = {
    "ollama": os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b"),
    "vllm": os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
}
DEFAULT_OUTPUTS = {
    "ollama": "evaluation.json",
    "vllm": "evaluation-vllm.json",
    "gateway": "evaluation-gateway.json",
}


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def count_numbered_items(response):
    """Count lines beginning with a numbered-list marker, such as 1."""
    pattern = re.compile(r"^\s*\d+[.)]\s+\S")
    return sum(
        bool(pattern.match(line))
        for line in response.splitlines()
    )


def count_bullet_items(response):
    """Count lines beginning with -, * or •."""
    pattern = re.compile(r"^\s*[-*•]\s+\S")
    return sum(
        bool(pattern.match(line))
        for line in response.splitlines()
    )


def evaluate_case(test, response):
    text = normalize(response)
    failures = []

    # Every required phrase must appear.
    required = test.get("must_include_all", [])
    missing = [
        term for term in required
        if normalize(term) not in text
    ]
    if missing:
        failures.append(
            f"Missing required phrases: {missing}"
        )

    # At least one alternative phrase must appear.
    alternatives = test.get("must_include_any", [])
    if alternatives and not any(
        normalize(term) in text for term in alternatives
    ):
        failures.append(
            f"None of the alternative phrases found: {alternatives}"
        )

    numbered_count = count_numbered_items(response)
    bullet_count = count_bullet_items(response)

    min_numbered = test.get("min_numbered_items")
    max_numbered = test.get("max_numbered_items")

    if min_numbered is not None and numbered_count < min_numbered:
        failures.append(
            f"Expected at least {min_numbered} numbered items; "
            f"got {numbered_count}"
        )

    if max_numbered is not None and numbered_count > max_numbered:
        failures.append(
            f"Expected at most {max_numbered} numbered items; "
            f"got {numbered_count}"
        )

    min_bullets = test.get("min_bullets")
    max_bullets = test.get("max_bullets")

    if min_bullets is not None and bullet_count < min_bullets:
        failures.append(
            f"Expected at least {min_bullets} bullet items; "
            f"got {bullet_count}"
        )

    if max_bullets is not None and bullet_count > max_bullets:
        failures.append(
            f"Expected at most {max_bullets} bullet items; "
            f"got {bullet_count}"
        )

    return {
        "passed": not failures,
        "failures": failures,
        "numbered_item_count": numbered_count,
        "bullet_item_count": bullet_count,
    }


def build_request(backend, gateway_backend, model, prompt, max_tokens):
    if backend == "ollama":
        return (
            f"{OLLAMA_URL}/api/generate",
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "num_predict": max_tokens,
                },
            },
        )

    if backend == "vllm":
        return (
            f"{VLLM_URL}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "seed": 42,
                "max_tokens": max_tokens,
            },
        )

    # Exercise the gateway itself rather than the backend directly.
    return (
        f"{API_URL}/generate/{gateway_backend}",
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
    )


def parse_response(backend, data):
    if backend == "ollama":
        return data["response"], {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        }

    if backend == "vllm":
        return data["choices"][0]["message"]["content"], data.get(
            "usage", {}
        )

    return data["response"], data.get("usage", {})


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the LLMOps evaluation dataset against a backend.",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "vllm", "gateway"],
        default=os.getenv("EVAL_BACKEND", "ollama"),
        help="Where to send requests (default: ollama).",
    )
    parser.add_argument(
        "--gateway-backend",
        choices=["ollama", "vllm"],
        default=os.getenv("EVAL_GATEWAY_BACKEND", "ollama"),
        help="Upstream used when --backend gateway is selected.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name reported/sent to the backend.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=120,
        help="Maximum generated tokens per case (default: 120).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Report path (default: results/evaluation[-backend].json).",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help=(
            "Exit with status 1 when pass_rate is below this fraction "
            "(0-1). Useful as a CI regression gate."
        ),
    )
    args = parser.parse_args(argv)

    model = args.model or DEFAULT_MODELS.get(args.backend) or (
        DEFAULT_MODELS[args.gateway_backend]
    )

    output = args.out or (
        RESULTS_DIR / DEFAULT_OUTPUTS[args.backend]
    )
    if not output.is_absolute():
        output = ROOT / output

    with DATASET.open(encoding="utf-8") as file:
        tests = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    results = []

    print(
        f"Evaluating {len(tests)} cases via {args.backend} "
        f"(model={model})"
    )

    for test in tests:
        start = time.perf_counter()
        url, body = build_request(
            args.backend,
            args.gateway_backend,
            model,
            test["prompt"],
            args.max_tokens,
        )

        try:
            response = requests.post(url, json=body, timeout=120)
            response.raise_for_status()
            data = response.json()
            answer, usage = parse_response(args.backend, data)
            latency = time.perf_counter() - start
            checks = evaluate_case(test, answer)

            result = {
                "id": test["id"],
                "prompt": test["prompt"],
                "response": answer,
                "latency_seconds": round(latency, 4),
                "usage": usage,
                **checks,
            }

        except (
            requests.RequestException,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as error:
            result = {
                "id": test["id"],
                "prompt": test["prompt"],
                "response": None,
                "latency_seconds": round(
                    time.perf_counter() - start, 4
                ),
                "passed": False,
                "failures": [f"Request/response error: {error}"],
                "numbered_item_count": 0,
                "bullet_item_count": 0,
            }

        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{status}] {result['id']} "
            f"({result['latency_seconds']:.3f}s)"
        )

        for failure in result["failures"]:
            print(f"    {failure}")

        if result["response"] is not None:
            print(f"    Answer: {result['response'][:220]}")

        print()

    passed = sum(result["passed"] for result in results)
    total = len(results)
    pass_rate = round(passed / total, 4) if total else 0

    report = {
        "model": model,
        "backend": args.backend,
        "gateway_backend": (
            args.gateway_backend if args.backend == "gateway" else None
        ),
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": pass_rate,
        "results": results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 45)
    print(f"Model: {model}")
    print(f"Passed: {passed}/{total}")
    print(f"Pass rate: {report['pass_rate']:.1%}")
    print(f"Report saved to: {output.relative_to(ROOT)}")

    if args.fail_under is not None and pass_rate < args.fail_under:
        print(
            f"FAILED quality gate: pass rate {pass_rate:.1%} is below "
            f"{args.fail_under:.1%}"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
