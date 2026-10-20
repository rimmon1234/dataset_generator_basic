import json
import random
import sys
import uuid
from pathlib import Path
import pandas as pd
import gradio as gr
import torch

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from loader import load_model_and_tokenizer, MODEL_ID
from prompts import SEED_TOPICS, GENERATOR_SYSTEM_PROMPT, GENERATOR_USER_TEMPLATE
from utils import extract_json, validate_alpaca_sample

# Global state for cached model & tokenizer
CACHED_MODEL = None
CACHED_TOKENIZER = None

def get_model_and_tokenizer():
    global CACHED_MODEL, CACHED_TOKENIZER
    if CACHED_MODEL is None or CACHED_TOKENIZER is None:
        CACHED_MODEL, CACHED_TOKENIZER = load_model_and_tokenizer(MODEL_ID)
    return CACHED_MODEL, CACHED_TOKENIZER

def load_dataset_records(file_path: str):
    """Loads all records from a jsonl file into a list of dicts."""
    path = Path(file_path)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return records

def load_dataset_dataframe(file_path: str):
    records = load_dataset_records(file_path)
    if not records:
        return pd.DataFrame(columns=["ID", "Topic", "Instruction", "Input", "Output", "Model"])
    
    rows = []
    for r in records:
        rows.append({
            "ID": r.get("id", ""),
            "Topic": r.get("seed_topic", ""),
            "Instruction": r.get("instruction", ""),
            "Input": r.get("input", ""),
            "Output": r.get("output", ""),
            "Model": r.get("model_id", "")
        })
    return pd.DataFrame(rows)

def generate_sample_core(model, tokenizer, topic: str, max_tokens: int, temperature: float):
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
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    tokens = output_ids[0][inputs.input_ids.shape[1]:]
    raw_response = tokenizer.decode(tokens, skip_special_tokens=True)
    parsed = extract_json(raw_response)
    if parsed and validate_alpaca_sample(parsed):
        return parsed, raw_response
    return None, raw_response

def run_gradio_generation(
    topic_choice: str,
    custom_topic: str,
    num_samples: int,
    temperature: float,
    max_tokens: int,
    output_filename: str,
    progress=gr.Progress(track_tqdm=True)
):
    output_path = Path(output_filename.strip() or "synthetic_dataset.jsonl")
    
    # Load existing entries for deduplication
    existing_records = load_dataset_records(str(output_path))
    seen_keys = {
        (r.get("instruction", "").strip().lower(), r.get("input", "").strip().lower())
        for r in existing_records
    }

    # Determine topic source
    base_topic = custom_topic.strip() if custom_topic.strip() else topic_choice
    use_random_seed = (base_topic == "Random from Seed List" or not base_topic)

    model, tokenizer = get_model_and_tokenizer()

    valid_count = 0
    attempt = 0
    max_attempts = num_samples * 4
    topics_pool = SEED_TOPICS.copy()
    random.shuffle(topics_pool)

    status_log = f"Started generation of {num_samples} sample(s)...\n"
    yield (
        status_log,
        "",  # latest id
        "",  # latest topic
        "",  # latest instruction
        "",  # latest input
        "",  # latest output
        load_dataset_dataframe(str(output_path)),
        str(output_path) if output_path.exists() else None
    )

    while valid_count < num_samples and attempt < max_attempts:
        attempt += 1
        current_topic = topics_pool[attempt % len(topics_pool)] if use_random_seed else base_topic

        progress((valid_count) / num_samples, desc=f"Generating sample {valid_count + 1}/{num_samples} (Attempt {attempt})")
        status_log += f"Attempt {attempt}: Querying topic '{current_topic[:30]}'...\n"
        
        sample, raw_output = generate_sample_core(model, tokenizer, current_topic, int(max_tokens), float(temperature))

        if not sample:
            status_log += f"⚠️ Attempt {attempt}: Model response was not valid JSON. Retrying...\n"
            yield (
                status_log, "", "", "", "", "",
                load_dataset_dataframe(str(output_path)),
                str(output_path) if output_path.exists() else None
            )
            continue

        dedup_key = (
            sample.get("instruction", "").strip().lower(),
            sample.get("input", "").strip().lower()
        )

        if dedup_key in seen_keys:
            status_log += f"⚠️ Attempt {attempt}: Duplicate instruction detected. Skipping...\n"
            yield (
                status_log, "", "", "", "", "",
                load_dataset_dataframe(str(output_path)),
                str(output_path) if output_path.exists() else None
            )
            continue

        seen_keys.add(dedup_key)
        valid_count += 1
        sample_id = f"synth_{uuid.uuid4().hex[:12]}"

        enriched = {
            "id": sample_id,
            "model_id": MODEL_ID,
            "seed_topic": current_topic,
            "instruction": sample["instruction"].strip(),
            "input": sample.get("input", "").strip(),
            "output": sample["output"].strip()
        }

        # Save immediately to file
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

        status_log += f"✅ Success ({valid_count}/{num_samples}): Generated sample {sample_id}\n"

        yield (
            status_log,
            sample_id,
            current_topic,
            enriched["instruction"],
            enriched["input"],
            enriched["output"],
            load_dataset_dataframe(str(output_path)),
            str(output_path)
        )

    status_log += f"\n🎉 Finished! Generated {valid_count} unique sample(s). Saved to {output_path}.\n"
    yield (
        status_log,
        sample_id if valid_count > 0 else "",
        current_topic if valid_count > 0 else "",
        enriched["instruction"] if valid_count > 0 else "",
        enriched["input"] if valid_count > 0 else "",
        enriched["output"] if valid_count > 0 else "",
        load_dataset_dataframe(str(output_path)),
        str(output_path) if output_path.exists() else None
    )

def export_as_json(file_path: str):
    records = load_dataset_records(file_path)
    if not records:
        return None
    export_path = Path(file_path).with_suffix(".json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return str(export_path)

# Build the Gradio UI
def create_ui():
    topic_options = ["Random from Seed List"] + SEED_TOPICS

    with gr.Blocks(title="Synthetic Dataset Generator") as demo:
        gr.Markdown(
            f"""
            # 🧠 Synthetic Dataset Generator Studio
            Generate high-quality instruction-tuning datasets (Alpaca format) powered by **{MODEL_ID}** in 4-bit NF4 precision.
            """
        )

        with gr.Tabs():
            # TAB 1: GENERATE
            with gr.TabItem("⚡ Generate Dataset"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Generation Controls")
                        topic_dropdown = gr.Dropdown(
                            label="Seed Topic (Select or leave as Random)",
                            choices=topic_options,
                            value="Random from Seed List"
                        )
                        custom_topic_box = gr.Textbox(
                            label="Custom Topic (Overrides dropdown if entered)",
                            placeholder="e.g. Fine-tuning vision-language models with LoRA...",
                            lines=2
                        )
                        num_samples_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=2,
                            step=1,
                            label="Number of Samples to Generate"
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.2,
                            value=0.7,
                            step=0.05,
                            label="Temperature (Diversity)"
                        )
                        max_tokens_slider = gr.Slider(
                            minimum=256,
                            maximum=1536,
                            value=896,
                            step=64,
                            label="Max Output Tokens"
                        )
                        output_file_box = gr.Textbox(
                            label="Save Target (.jsonl)",
                            value="synthetic_dataset.jsonl"
                        )
                        gen_button = gr.Button("🚀 Generate Samples", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Latest Generated Example")
                        with gr.Group():
                            with gr.Row():
                                out_id = gr.Textbox(label="Sample ID", interactive=False, scale=1)
                                out_topic = gr.Textbox(label="Seed Topic", interactive=False, scale=2)
                            out_instruction = gr.Textbox(label="Instruction", lines=3, interactive=False)
                            out_input = gr.Textbox(label="Input Context (Optional)", lines=3, interactive=False)
                            out_response = gr.Textbox(label="Generated Output Response", lines=8, interactive=False)

                        log_box = gr.Textbox(label="Execution Logs", lines=5, interactive=False)

            # TAB 2: EXPLORE & EXPORT
            with gr.TabItem("📊 Dataset Explorer & Export"):
                gr.Markdown("### 🗃️ Stored Dataset Samples")
                with gr.Row():
                    explorer_file_box = gr.Textbox(
                        label="Dataset File to View",
                        value="synthetic_dataset.jsonl",
                        scale=3
                    )
                    refresh_btn = gr.Button("🔄 Refresh Table", scale=1)

                dataset_table = gr.Dataframe(
                    value=load_dataset_dataframe("synthetic_dataset.jsonl"),
                    headers=["ID", "Topic", "Instruction", "Input", "Output", "Model"],
                    wrap=True,
                    interactive=False
                )

                gr.Markdown("### 💾 Download & Export")
                with gr.Row():
                    download_jsonl_btn = gr.Button("📥 Download as JSONL")
                    download_json_btn = gr.Button("📥 Export as Formatted JSON")

                file_download_component = gr.File(label="Download File", interactive=False)

            # TAB 3: SYSTEM INFO
            with gr.TabItem("🖥️ System & Hardware"):
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Detected"
                vram_total = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB" if torch.cuda.is_available() else "N/A"
                gr.Markdown(
                    f"""
                    ### 💻 Hardware & Model Configuration
                    - **Active Model:** `{MODEL_ID}`
                    - **Quantization:** 4-bit NormalFloat4 (NF4) with Double Quant
                    - **Compute Precision:** `{'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'}`
                    - **Detected GPU:** `{gpu_name}`
                    - **Total VRAM:** `{vram_total}`
                    - **Deduplication:** Active on `(instruction, input)`
                    """
                )

        # Wire Events
        gen_button.click(
            fn=run_gradio_generation,
            inputs=[
                topic_dropdown,
                custom_topic_box,
                num_samples_slider,
                temperature_slider,
                max_tokens_slider,
                output_file_box
            ],
            outputs=[
                log_box,
                out_id,
                out_topic,
                out_instruction,
                out_input,
                out_response,
                dataset_table,
                file_download_component
            ]
        )

        refresh_btn.click(
            fn=load_dataset_dataframe,
            inputs=[explorer_file_box],
            outputs=[dataset_table]
        )

        download_jsonl_btn.click(
            fn=lambda f: f if Path(f).exists() else None,
            inputs=[explorer_file_box],
            outputs=[file_download_component]
        )

        download_json_btn.click(
            fn=export_as_json,
            inputs=[explorer_file_box],
            outputs=[file_download_component]
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
