
import csv
import os
import time
import statistics
import requests

URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:0.5b"
REPEATS = 3

PROMPTS = [
    "What is model inference? Answer in two sentences.",
    "Explain LLMOps in three bullet points.",
    "What is the difference between training and inference? Answer briefly.",
]


def print_section(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


os.makedirs("results", exist_ok=True)
rows = []

print_section("LLM LOCAL BENCHMARK")
print(f"Model: {MODEL} | Repeats per prompt: {REPEATS} | Endpoint: {URL}")

for prompt in PROMPTS:
    latencies = []
    prompt_rows = []

    print_section(f"PROMPT: {prompt}")

    for run in range(1, REPEATS + 1):
        start = time.perf_counter()

        response = requests.post(
            URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "num_predict": 60,
                },
            },
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()
        wall_seconds = time.perf_counter() - start
        latencies.append(wall_seconds)

        generated = result.get("eval_count", 0)
        generation_seconds = result.get("eval_duration", 0) / 1e9
        tokens_per_second = (
            generated / generation_seconds
            if generation_seconds > 0 else 0
        )
        model_output = result.get("response", "").strip()

        row = {
            "model": MODEL,
            "prompt": prompt,
            "run": run,
            "wall_seconds": round(wall_seconds, 4),
            "generated_tokens": generated,
            "tokens_per_second": round(tokens_per_second, 2),
            "load_seconds": round(
                result.get("load_duration", 0) / 1e9, 4
            ),
            "response": model_output,
        }
        rows.append(row)
        prompt_rows.append(row)

        print(
            f"  Run {run:>2}/{REPEATS}: "
            f"wall={wall_seconds:.3f}s | "
            f"tokens={generated} | "
            f"generation={tokens_per_second:.2f} tok/s"
        )
        print(f"    Output: {model_output}")

    median_latency = statistics.median(latencies)
    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    avg_tokens_per_second = statistics.mean(
        [item["tokens_per_second"] for item in prompt_rows]
    )

    print()
    print("  Summary")
    print(f"    median latency : {median_latency:.3f}s")
    print(f"    average latency: {avg_latency:.3f}s")
    print(f"    min / max     : {min_latency:.3f}s / {max_latency:.3f}s")
    print(f"    avg throughput: {avg_tokens_per_second:.2f} tok/s")

output_file = "results/benchmark.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print_section("RESULTS")
print(f"Saved {len(rows)} measurements to {output_file}")
print(f"CSV columns: {', '.join(rows[0].keys())}")
