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

from .loader import load_model_and_tokenizer, MODEL_ID
from .prompts import SEED_TOPICS, GENERATOR_SYSTEM_PROMPT, GENERATOR_USER_TEMPLATE
from ..qc.utils import extract_json
from ..qc.validator import QualityControlValidator

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

    generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    parsed = extract_json(raw_response)
    
    return parsed, raw_response

def run_generator(
    total_samples: int = 5,
    output_path: str = "data/synthetic_dataset.jsonl",
    rejections_path: str = "data/rejections.jsonl",
    max_new_tokens: int = 896,
    temperature: float = 0.7,
    model_id: str = MODEL_ID
):
    print("=== Starting Synthetic Dataset Generator with Quality Control ===")
    print(f"Target Samples: {total_samples}")
    print(f"Model ID:       {model_id}")
    print(f"Output File:    {output_path}")
    print(f"Rejection Log:  {rejections_path}")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize QC Validator and pre-populate deduplication cache
    validator = QualityControlValidator(rejection_log_path=rejections_path)
    validator.load_existing_dataset(str(output_file))
    existing_count = len(validator.seen_exact_keys)
    if existing_count > 0:
        print(f"Found {existing_count} existing unique samples in {output_path} (resuming)")

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_id=model_id)

    valid_count = 0
    attempt_count = 0
    rejection_reasons = {}
    max_attempts = total_samples * 5

    topics_pool = SEED_TOPICS.copy()
    random.shuffle(topics_pool)

    progress_bar = tqdm(total=total_samples, desc="Generating Dataset")

    while valid_count < total_samples and attempt_count < max_attempts:
        topic = topics_pool[attempt_count % len(topics_pool)]
        attempt_count += 1

        parsed, raw_output = generate_sample(
            model=model,
            tokenizer=tokenizer,
            topic=topic,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        if not parsed:
            reason = "Malformed JSON / unparseable output"
            validator.log_rejection(raw_output, reason, "json_parse_error", topic)
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            tqdm.write(f"[QC REJECT] Attempt {attempt_count}: {reason} for '{topic[:25]}'")
            continue

        is_valid, failure_reason = validator.validate(parsed, raw_output, topic=topic)

        if not is_valid:
            rejection_reasons[failure_reason] = rejection_reasons.get(failure_reason, 0) + 1
            tqdm.write(f"[QC REJECT] Attempt {attempt_count}: {failure_reason}")
            continue

        sample_id = f"synth_{uuid.uuid4().hex[:12]}"
        enriched_sample = {
            "id": sample_id,
            "model_id": model_id,
            "category": parsed.get("category", topic),
            "difficulty": parsed.get("difficulty", "intermediate"),
            "seed_topic": topic,
            "instruction": parsed["instruction"].strip(),
            "input": parsed.get("input", "").strip(),
            "output": parsed["output"].strip(),
            "review_status": "pending_review"
        }

        valid_count += 1

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(enriched_sample, ensure_ascii=False) + "\n")

        progress_bar.update(1)
        progress_bar.set_postfix({
            "Topic": topic[:18],
            "Valid": f"{valid_count}/{attempt_count}",
            "Rejects": sum(rejection_reasons.values())
        })

    progress_bar.close()

    total_rejected = sum(rejection_reasons.values())
    print("\n=== Generation & Quality Control Summary ===")
    print(f"Total attempts:      {attempt_count}")
    print(f"New valid samples:   {valid_count}")
    print(f"Rejected samples:    {total_rejected}")
    if rejection_reasons:
        print("\nRejection Breakdown:")
        for r, count in rejection_reasons.items():
            print(f"  - [{count}x] {r}")
    print(f"\nTotal unique in dataset: {len(validator.seen_exact_keys)}")
    rate = (valid_count / attempt_count * 100) if attempt_count > 0 else 0
    print(f"QC Pass Rate:            {rate:.1f}%")
    print(f"Dataset location:        {output_file.resolve()}")
    print(f"Rejections log:          {Path(rejections_path).resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Instruction Dataset Generator with Quality Control")
    parser.add_argument("--samples", type=int, default=5, help="Number of valid samples to generate")
    parser.add_argument("--output", type=str, default="data/synthetic_dataset.jsonl", help="Output JSONL filepath")
    parser.add_argument("--rejections", type=str, default="data/rejections.jsonl", help="Rejections log filepath")
    parser.add_argument("--max-tokens", type=int, default=896, help="Max new tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    args = parser.parse_args()

    run_generator(
        total_samples=args.samples,
        output_path=args.output,
        rejections_path=args.rejections,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
