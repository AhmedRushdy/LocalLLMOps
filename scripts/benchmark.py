import requests
import time

URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:0.5b"

prompts = [
    "What is model inference?",
    "Explain LLMOps in three bullet points.",
    "What is the difference between training and inference?",
]

for prompt in prompts:
    start = time.perf_counter()

    response = requests.post(
        URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()

    result = response.json()
    wall_time = time.perf_counter() - start

    generated_tokens = result.get("eval_count", 0)
    generation_seconds = result.get("eval_duration", 0) / 1e9

    tokens_per_second = (
        generated_tokens / generation_seconds
        if generation_seconds > 0
        else 0
    )

    print(f"\nPrompt: {prompt}")
    print(f"Response: {result['response'][:200]}...")
    print(f"Wall time: {wall_time:.3f} seconds")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Generation speed: {tokens_per_second:.2f} tokens/sec")