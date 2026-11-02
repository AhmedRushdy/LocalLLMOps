import csv
import time
import statistics
import requests
from pathlib import Path

URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:0.5b"
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
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_predict": MAX_TOKENS,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()

Path("results").mkdir(exist_ok=True)
rows = []

print("Ollama benchmark")
print(f"Model: {MODEL} | Repeats: {REPEATS}")
print("Running warm-up request...")
request_model(PROMPTS[0])
print("Warm-up complete.")

for prompt in PROMPTS:
    latencies = []
    prompt_rows = []
    print(f"\nPROMPT: {prompt}")

    for run in range(1, REPEATS + 1):
        start = time.perf_counter()
        result = request_model(prompt)
        wall_seconds = time.perf_counter() - start

        generated = result.get("eval_count", 0)
        answer = result.get("response", "").strip()
        throughput = generated / wall_seconds if wall_seconds else 0

        row = {
            "engine": "Ollama",
            "model": MODEL,
            "prompt": prompt,
            "run": run,
            "wall_seconds": round(wall_seconds, 4),
            "prompt_tokens": result.get("prompt_eval_count", 0),
            "generated_tokens": generated,
            "tokens_per_second_wall": round(throughput, 2),
            "response": answer,
        }
        rows.append(row)
        prompt_rows.append(row)
        latencies.append(wall_seconds)

        print(
            f"Run {run}/{REPEATS}: latency={wall_seconds:.3f}s | "
            f"output_tokens={generated} | "
            f"HTTP throughput={throughput:.2f} tok/s"
        )

    print(f"Median latency: {statistics.median(latencies):.3f}s")
    print(f"Average latency: {statistics.mean(latencies):.3f}s")
    print(
        "Average HTTP throughput: "
        f"{statistics.mean(r['tokens_per_second_wall'] for r in prompt_rows):.2f} tok/s"
    )

output = "results/benchmark-ollama-fair.csv"
with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} measurements to {output}")
