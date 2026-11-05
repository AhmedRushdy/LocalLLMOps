import csv
import os
import time
import statistics
import requests

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REPEATS = 3
MAX_TOKENS = 60

PROMPTS = [
    "What is model inference? Answer in two sentences.",
    "Explain LLMOps in three bullet points.",
    "What is the difference between training and inference? Answer briefly.",
]


def request_model(prompt):
    response = requests.post(
        URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": MAX_TOKENS,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


os.makedirs("results", exist_ok=True)
rows = []

print("=" * 80)
print("vLLM BENCHMARK")
print(f"Model: {MODEL}")
print(f"Repeats per prompt: {REPEATS}")
print("Running one warm-up request; it will not be included in results.")

# Warm up the model before measuring
request_model(PROMPTS[0])
print("Warm-up complete.")

for prompt in PROMPTS:
    latencies = []
    prompt_rows = []

    print("\n" + "=" * 80)
    print(f"PROMPT: {prompt}")

    for run in range(1, REPEATS + 1):
        start = time.perf_counter()
        result = request_model(prompt)
        wall_seconds = time.perf_counter() - start

        answer = result["choices"][0]["message"]["content"].strip()
        usage = result.get("usage", {})
        generated = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)

        # Approximate end-to-end throughput, including HTTP overhead
        tokens_per_second = (
            generated / wall_seconds if wall_seconds > 0 else 0
        )

        row = {
            "engine": "vLLM",
            "model": MODEL,
            "prompt": prompt,
            "run": run,
            "wall_seconds": round(wall_seconds, 4),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated,
            "tokens_per_second_wall": round(tokens_per_second, 2),
            "response": answer,
        }
        rows.append(row)
        prompt_rows.append(row)
        latencies.append(wall_seconds)

        print(
            f"Run {run}/{REPEATS}: "
            f"latency={wall_seconds:.3f}s | "
            f"output_tokens={generated} | "
            f"HTTP throughput={tokens_per_second:.2f} tok/s"
        )
        print(f"  Output: {answer[:250]}")

    print("\nSummary")
    print(f"  Median latency: {statistics.median(latencies):.3f}s")
    print(f"  Average latency: {statistics.mean(latencies):.3f}s")
    print(
        "  Average HTTP throughput: "
        f"{statistics.mean(r['tokens_per_second_wall'] for r in prompt_rows):.2f} tok/s"
    )

output_file = "results/benchmark-vllm.csv"
with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 80)
print(f"Saved {len(rows)} measurements to {output_file}")
