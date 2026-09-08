import requests
import json
import time

URL = "http://127.0.0.1:8001/v1/chat/completions"

test_cases = [
    {
        "name": "Math & Multi-step Calculation",
        "prompt": "A farmer has 15 cows and 25 chickens. How many total legs are on the farm? Show your step-by-step calculation.",
        "max_tokens": 200,
        "temperature": 0.0
    },
    {
        "name": "Logical Deduction",
        "prompt": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? Explain clearly.",
        "max_tokens": 150,
        "temperature": 0.0
    },
    {
        "name": "Code Analysis & Edge Case Handling",
        "prompt": "Look at this function:\n```python\ndef average(numbers):\n    return sum(numbers) / len(numbers)\n```\nWhat bug or exception happens if `numbers` is empty? Provide the fixed Python function.",
        "max_tokens": 200,
        "temperature": 0.0
    },
    {
        "name": "Structured JSON Extraction",
        "prompt": "Extract the key entities from this sentence: 'On October 14, 2024, Dr. Sarah Lin from DeepMind presented a keynote in Tokyo regarding reinforcement learning.' Return valid JSON with keys: 'person', 'organization', 'date', 'location', 'topic'. Do not output extra prose.",
        "max_tokens": 150,
        "temperature": 0.0
    },
    {
        "name": "Instruction Following / Haiku",
        "prompt": "Write a 5-7-5 syllable haiku about an NPU running AI on silicon.",
        "max_tokens": 100,
        "temperature": 0.3
    }
]

print("="*70)
print("RUNNING EXTENSIVE QUALITY EVALUATION ON MINICPM5-2B (AMD XDNA 2 NPU)")
print("="*70)

for idx, tc in enumerate(test_cases, 1):
    payload = {
        "model": "minicpm5:2b",
        "messages": [
            {"role": "user", "content": tc["prompt"]}
        ],
        "max_tokens": tc["max_tokens"],
        "temperature": tc["temperature"]
    }
    
    t0 = time.time()
    resp = requests.post(URL, json=payload, timeout=60)
    elapsed = time.time() - t0
    
    if resp.status_code != 200:
        print(f"[{idx}] {tc['name']} FAILED: {resp.status_code} {resp.text}")
        continue
        
    data = resp.json()
    choice = data["choices"][0]
    content = choice["message"]["content"]
    usage = data.get("usage", {})
    
    decode_speed = usage.get("decoding_speed_tps", 0.0)
    prefill_speed = usage.get("prefill_speed_tps", 0.0)
    ttft = usage.get("prefill_duration_ttft", 0.0)
    
    print(f"\n--- Test {idx}: {tc['name']} ---")
    print(f"Prompt: {tc['prompt']}")
    print(f"Output:\n{content.strip()}")
    print(f"\n[Metrics] TTFT: {ttft:.3f}s | Prefill: {prefill_speed:.1f} tok/s | Decode: {decode_speed:.1f} tok/s | Total Time: {elapsed:.2f}s")
    print("-" * 70)

print("\nQuality Evaluation Finished!")
