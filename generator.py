import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path
from tqdm import tqdm
import torch

# Ensure UTF-8 output on Windows consoles to prevent cp1252 charmap errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from loader import load_model_and_tokenizer, MODEL_ID
from prompts import SEED_TOPICS, GENERATOR_SYSTEM_PROMPT, GENERATOR_USER_TEMPLATE
from utils import extract_json, validate_alpaca_sample

def generate_sample(model, tokenizer, topic: str, max_new_tokens: int = 896, temperature: float = 0.7):
    """
    Generates a single synthetic instruction-response pair for the given topic.
    """
    user_prompt = GENERATOR_USER_TEMPLATE.format(topic=topic)
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
        # Disable thinking tokens for direct JSON generation
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode only the newly generated response tokens
    generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    parsed = extract_json(raw_response)
    if parsed and validate_alpaca_sample(parsed):
        parsed["seed_topic"] = topic
        return parsed, raw_response

    return None, raw_response

def load_existing_dedup_keys(output_path: Path) -> set:
    """
    Pre-populates deduplication cache from an existing dataset file.
    Deduplication key is normalized (instruction, input).
    """
    seen = set()
    if not output_path.exists():
        return seen

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                inst = entry.get("instruction", "").strip().lower()
                inp = entry.get("input", "").strip().lower()
                seen.add((inst, inp))
            except Exception:
                continue
    return seen

def run_generator(
    total_samples: int = 5,
    output_path: str = "synthetic_dataset.jsonl",
    max_new_tokens: int = 896,
    temperature: float = 0.7,
    model_id: str = MODEL_ID
):
    print("=== Starting Synthetic Dataset Generator ===")
    print(f"Target Samples: {total_samples}")
    print(f"Model ID:       {model_id}")
    print(f"Output File:    {output_path}")

    output_file = Path(output_path)
    
    # Load previously generated keys to prevent duplicates across multiple runs
    seen_keys = load_existing_dedup_keys(output_file)
    initial_existing_count = len(seen_keys)
    if initial_existing_count > 0:
        print(f"Found {initial_existing_count} existing unique samples in {output_path} (resuming)")

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_id=model_id)

    valid_count = 0
    attempt_count = 0
    duplicate_count = 0
    max_attempts = total_samples * 4

    # Shuffle seed topics pool
    topics_pool = SEED_TOPICS.copy()
    random.shuffle(topics_pool)

    progress_bar = tqdm(total=total_samples, desc="Generating Dataset")

    while valid_count < total_samples and attempt_count < max_attempts:
        topic = topics_pool[attempt_count % len(topics_pool)]
        attempt_count += 1

        sample, raw_output = generate_sample(
            model=model,
            tokenizer=tokenizer,
            topic=topic,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        if not sample:
            tqdm.write(f"[WARN] [Attempt {attempt_count}] Invalid format for '{topic[:25]}'. Retrying...")
            continue

        # Deduplication check based on normalized (instruction, input)
        inst_normalized = sample.get("instruction", "").strip().lower()
        inp_normalized = sample.get("input", "").strip().lower()
        dedup_key = (inst_normalized, inp_normalized)

        if dedup_key in seen_keys:
            duplicate_count += 1
            tqdm.write(f"[DEDUP] Duplicate instruction detected for '{topic[:25]}'. Skipping...")
            continue

        # Register key
        seen_keys.add(dedup_key)

        # Enrich sample with metadata: unique ID and generator model ID
        sample_id = f"synth_{uuid.uuid4().hex[:12]}"
        enriched_sample = {
            "id": sample_id,
            "model_id": model_id,
            "seed_topic": sample.get("seed_topic", topic),
            "instruction": sample["instruction"].strip(),
            "input": sample.get("input", "").strip(),
            "output": sample["output"].strip()
        }

        valid_count += 1

        # Append immediately to file so no progress is lost
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(enriched_sample, ensure_ascii=False) + "\n")

        progress_bar.update(1)
        progress_bar.set_postfix({
            "Topic": topic[:18],
            "Valid": f"{valid_count}/{attempt_count}",
            "Dupes": duplicate_count
        })

    progress_bar.close()

    print("\n=== Generation Summary ===")
    print(f"Total attempts:      {attempt_count}")
    print(f"New valid samples:   {valid_count}")
    print(f"Duplicates rejected: {duplicate_count}")
    print(f"Total in dataset:    {len(seen_keys)}")
    rate = (valid_count / attempt_count * 100) if attempt_count > 0 else 0
    print(f"Success rate:        {rate:.1f}%")
    print(f"Dataset location:    {output_file.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Instruction Dataset Generator")
    parser.add_argument("--samples", type=int, default=5, help="Number of valid samples to generate")
    parser.add_argument("--output", type=str, default="synthetic_dataset.jsonl", help="Output JSONL filepath")
    parser.add_argument("--max-tokens", type=int, default=896, help="Max new tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    args = parser.parse_args()

    run_generator(
        total_samples=args.samples,
        output_path=args.output,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
