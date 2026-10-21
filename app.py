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

# Ensure local package import
sys.path.insert(0, str(Path(__file__).parent))

from src.core.loader import load_model_and_tokenizer, MODEL_ID
from src.core.prompts import SEED_TOPICS, GENERATOR_SYSTEM_PROMPT, GENERATOR_USER_TEMPLATE
from src.qc.utils import extract_json
from src.qc.validator import QualityControlValidator
from src.tools.exporter import convert_jsonl_to_tabular

# Global state for cached model & tokenizer
CACHED_MODEL = None
CACHED_TOKENIZER = None

DEFAULT_DATASET_PATH = "data/synthetic_dataset.jsonl"
DEFAULT_REJECTIONS_PATH = "data/rejections.jsonl"

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

def save_all_records(file_path: str, records: list):
    """Overwrites jsonl file with updated list of records."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_dataset_dataframe(file_path: str):
    records = load_dataset_records(file_path)
    if not records:
        return pd.DataFrame(columns=["ID", "Category", "Difficulty", "Instruction", "Input", "Output", "Status"])
    
    rows = []
    for r in records:
        rows.append({
            "ID": r.get("id", ""),
            "Category": r.get("category", r.get("seed_topic", "")),
            "Difficulty": r.get("difficulty", "intermediate"),
            "Instruction": r.get("instruction", ""),
            "Input": r.get("input", ""),
            "Output": r.get("output", ""),
            "Status": r.get("review_status", "pending_review")
        })
    return pd.DataFrame(rows)

def load_rejections_dataframe(file_path: str = DEFAULT_REJECTIONS_PATH):
    records = load_dataset_records(file_path)
    if not records:
        return pd.DataFrame(columns=["Timestamp", "Topic", "Check Type", "Failure Reason"])
    
    rows = []
    for r in records:
        rows.append({
            "Timestamp": r.get("timestamp", "")[:19],
            "Topic": r.get("topic", ""),
            "Check Type": r.get("check_type", ""),
            "Failure Reason": r.get("reason", "")
        })
    return pd.DataFrame(rows)

def compute_dataset_stats(file_path: str):
    records = load_dataset_records(file_path)
    if not records:
        return (
            "No records found in dataset yet.",
            pd.DataFrame(columns=["Difficulty", "Count"]),
            pd.DataFrame(columns=["Category", "Count"])
        )
    
    total = len(records)
    approved = sum(1 for r in records if r.get("review_status") == "approved")
    rejected = sum(1 for r in records if r.get("review_status") == "rejected")
    pending = total - approved - rejected

    diff_counts = {}
    cat_counts = {}
    for r in records:
        d = r.get("difficulty", "intermediate")
        c = r.get("category", "General")
        diff_counts[d] = diff_counts.get(d, 0) + 1
        cat_counts[c] = cat_counts.get(c, 0) + 1

    summary_text = (
        f"### 📈 Dataset Summary\n"
        f"**Total Records:** {total} | "
        f"✅ **Approved:** {approved} | "
        f"⏳ **Pending Review:** {pending} | "
        f"❌ **Rejected:** {rejected}"
    )

    diff_df = pd.DataFrame(list(diff_counts.items()), columns=["Difficulty", "Count"])
    cat_df = pd.DataFrame(list(cat_counts.items()), columns=["Category", "Count"])
    return summary_text, diff_df, cat_df

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
    return parsed, raw_response

def run_gradio_generation(
    topic_choice: str,
    custom_topic: str,
    num_samples: int,
    temperature: float,
    max_tokens: int,
    output_filename: str,
    rejections_filename: str,
    progress=gr.Progress(track_tqdm=True)
):
    output_path = Path(output_filename.strip() or DEFAULT_DATASET_PATH)
    rejections_path = Path(rejections_filename.strip() or DEFAULT_REJECTIONS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejections_path.parent.mkdir(parents=True, exist_ok=True)

    validator = QualityControlValidator(rejection_log_path=str(rejections_path))
    validator.load_existing_dataset(str(output_path))

    base_topic = custom_topic.strip() if custom_topic.strip() else topic_choice
    use_random_seed = (base_topic == "Random from Seed List" or not base_topic)

    model, tokenizer = get_model_and_tokenizer()

    valid_count = 0
    attempt = 0
    max_attempts = num_samples * 5
    topics_pool = SEED_TOPICS.copy()
    random.shuffle(topics_pool)

    status_log = f"=== Starting QC-Enforced Generation ({num_samples} target samples) ===\n"
    yield (
        status_log, "", "", "", "", "", "",
        load_dataset_dataframe(str(output_path)),
        load_rejections_dataframe(str(rejections_path)),
        str(output_path) if output_path.exists() else None
    )

    while valid_count < num_samples and attempt < max_attempts:
        attempt += 1
        current_topic = topics_pool[attempt % len(topics_pool)] if use_random_seed else base_topic

        progress((valid_count) / num_samples, desc=f"Generating sample {valid_count + 1}/{num_samples} (Attempt {attempt})")
        status_log += f"Attempt {attempt}: Querying topic '{current_topic[:30]}'...\n"

        parsed, raw_output = generate_sample_core(model, tokenizer, current_topic, int(max_tokens), float(temperature))

        if not parsed:
            reason = "Malformed JSON / unparseable output"
            validator.log_rejection(raw_output, reason, "json_parse_error", current_topic)
            status_log += f"❌ QC Rejected (Attempt {attempt}): {reason}\n"
            yield (
                status_log, "", "", "", "", "", "",
                load_dataset_dataframe(str(output_path)),
                load_rejections_dataframe(str(rejections_path)),
                str(output_path) if output_path.exists() else None
            )
            continue

        is_valid, failure_reason = validator.validate(parsed, raw_output, topic=current_topic)

        if not is_valid:
            status_log += f"❌ QC Rejected (Attempt {attempt}): {failure_reason}\n"
            yield (
                status_log, "", "", "", "", "", "",
                load_dataset_dataframe(str(output_path)),
                load_rejections_dataframe(str(rejections_path)),
                str(output_path) if output_path.exists() else None
            )
            continue

        valid_count += 1
        sample_id = f"synth_{uuid.uuid4().hex[:12]}"

        enriched = {
            "id": sample_id,
            "model_id": MODEL_ID,
            "category": parsed.get("category", current_topic),
            "difficulty": parsed.get("difficulty", "intermediate"),
            "seed_topic": current_topic,
            "instruction": parsed["instruction"].strip(),
            "input": parsed.get("input", "").strip(),
            "output": parsed["output"].strip(),
            "review_status": "pending_review"
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

        status_log += f"✅ QC Passed ({valid_count}/{num_samples}): Generated {sample_id} [{enriched['category']} | {enriched['difficulty']}]\n"

        yield (
            status_log,
            sample_id,
            enriched["category"],
            enriched["difficulty"],
            enriched["instruction"],
            enriched["input"],
            enriched["output"],
            load_dataset_dataframe(str(output_path)),
            load_rejections_dataframe(str(rejections_path)),
            str(output_path)
        )

    status_log += f"\n🎉 Finished! Generated {valid_count} quality-verified sample(s). Saved to {output_path}.\n"
    yield (
        status_log,
        sample_id if valid_count > 0 else "",
        enriched["category"] if valid_count > 0 else "",
        enriched["difficulty"] if valid_count > 0 else "",
        enriched["instruction"] if valid_count > 0 else "",
        enriched["input"] if valid_count > 0 else "",
        enriched["output"] if valid_count > 0 else "",
        load_dataset_dataframe(str(output_path)),
        load_rejections_dataframe(str(rejections_path)),
        str(output_path) if output_path.exists() else None
    )

# Human Review Functions
def get_sample_for_review(file_path: str, index: int):
    records = load_dataset_records(file_path)
    if not records:
        return "No records found in dataset.", "", "", "", "", "", 0, "No records"
    
    idx = max(0, min(index, len(records) - 1))
    sample = records[idx]
    
    status_badge = f"Status: {sample.get('review_status', 'pending_review').upper()} (Sample {idx + 1} of {len(records)})"
    return (
        sample.get("id", ""),
        sample.get("category", ""),
        sample.get("difficulty", "intermediate"),
        sample.get("instruction", ""),
        sample.get("input", ""),
        sample.get("output", ""),
        idx,
        status_badge
    )

def update_sample_review(file_path: str, index: int, new_status: str, edited_inst: str, edited_inp: str, edited_out: str):
    records = load_dataset_records(file_path)
    if not records or index >= len(records):
        return "Failed to update: index out of range."
    
    records[index]["review_status"] = new_status
    records[index]["instruction"] = edited_inst.strip()
    records[index]["input"] = edited_inp.strip()
    records[index]["output"] = edited_out.strip()
    
    save_all_records(file_path, records)
    return f"Sample {records[index].get('id')} marked as '{new_status}' and updated."

def export_as_json(file_path: str):
    records = load_dataset_records(file_path)
    if not records:
        return None
    export_path = Path(file_path).with_suffix(".json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return str(export_path)

def export_as_csv(file_path: str):
    try:
        out = convert_jsonl_to_tabular(file_path, "csv")
        return str(out) if out else None
    except Exception as e:
        print(f"Error exporting CSV: {e}")
        return None

def create_ui():
    topic_options = ["Random from Seed List"] + SEED_TOPICS

    with gr.Blocks(title="Synthetic Dataset Generator Studio") as demo:
        gr.Markdown(
            f"""
            # 🧠 Synthetic Dataset Generator Studio
            *High-quality instruction dataset generation with strict multi-tier Quality Control (Schema, AST Code Syntax, Deduplication, and Balance Tracking).*
            """
        )

        with gr.Tabs():
            # TAB 1: GENERATE
            with gr.TabItem("⚡ Quality-Controlled Generator"):
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
                            placeholder="e.g. Distributed tracing in microservices...",
                            lines=2
                        )
                        num_samples_slider = gr.Slider(
                            minimum=1, maximum=20, value=2, step=1,
                            label="Number of Valid Samples to Generate"
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.1, maximum=1.2, value=0.7, step=0.05,
                            label="Temperature (Creativity)"
                        )
                        max_tokens_slider = gr.Slider(
                            minimum=256, maximum=1536, value=896, step=64,
                            label="Max Output Tokens"
                        )
                        with gr.Row():
                            output_file_box = gr.Textbox(label="Dataset Target (.jsonl)", value=DEFAULT_DATASET_PATH, scale=1)
                            rejections_file_box = gr.Textbox(label="Rejections Log (.jsonl)", value=DEFAULT_REJECTIONS_PATH, scale=1)
                        gen_button = gr.Button("🚀 Generate Validated Samples", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Latest Verified Sample")
                        with gr.Group():
                            with gr.Row():
                                out_id = gr.Textbox(label="Sample ID", interactive=False, scale=1)
                                out_category = gr.Textbox(label="Category", interactive=False, scale=1)
                                out_difficulty = gr.Textbox(label="Difficulty", interactive=False, scale=1)
                            out_instruction = gr.Textbox(label="Instruction", lines=3, interactive=False)
                            out_input = gr.Textbox(label="Input Context (Optional)", lines=3, interactive=False)
                            out_response = gr.Textbox(label="Generated Output Response", lines=8, interactive=False)

                        log_box = gr.Textbox(label="Execution & QC Logs", lines=5, interactive=False)

            # TAB 2: HUMAN REVIEW & QUALITY AUDIT
            with gr.TabItem("🕵️ Human Review & Quality Audit"):
                gr.Markdown(
                    """
                    ### 🔍 Human-in-the-Loop Review
                    *Plausible-sounding output can still have subtle bugs. Review, edit, or reject samples directly below.*
                    """
                )
                with gr.Row():
                    review_file_box = gr.Textbox(label="Dataset File to Review", value=DEFAULT_DATASET_PATH, scale=2)
                    review_index_slider = gr.Number(label="Sample Index (0-based)", value=0, precision=0, scale=1)
                    load_review_btn = gr.Button("📂 Load Sample", scale=1)

                review_status_badge = gr.Markdown("Status: PENDING_REVIEW")
                
                with gr.Group():
                    with gr.Row():
                        rev_id = gr.Textbox(label="Sample ID", interactive=False, scale=1)
                        rev_cat = gr.Textbox(label="Category", interactive=False, scale=1)
                        rev_diff = gr.Textbox(label="Difficulty", interactive=False, scale=1)
                    rev_inst = gr.Textbox(label="Instruction (Editable)", lines=3)
                    rev_inp = gr.Textbox(label="Input Context (Editable)", lines=3)
                    rev_out = gr.Textbox(label="Output Response (Editable)", lines=8)

                with gr.Row():
                    approve_btn = gr.Button("✅ Approve & Save", variant="primary", size="lg")
                    reject_btn = gr.Button("❌ Reject & Flag", variant="stop", size="lg")
                    save_edit_btn = gr.Button("💾 Save Edits (Keep Pending)", size="lg")

                review_feedback_box = gr.Markdown("")

            # TAB 3: DATASET EXPLORER, BALANCE & REJECTIONS
            with gr.TabItem("📊 Explorer, Balance & Rejections"):
                with gr.Row():
                    explorer_file_box = gr.Textbox(label="Dataset File", value=DEFAULT_DATASET_PATH, scale=3)
                    refresh_btn = gr.Button("🔄 Refresh All Data", scale=1)

                stats_text, initial_diff_df, initial_cat_df = compute_dataset_stats(DEFAULT_DATASET_PATH)
                stats_markdown = gr.Markdown(stats_text)
                with gr.Row():
                    difficulty_plot = gr.BarPlot(
                        value=initial_diff_df,
                        x="Difficulty",
                        y="Count",
                        title="Difficulty Balance",
                        tooltip=["Difficulty", "Count"]
                    )
                    category_plot = gr.BarPlot(
                        value=initial_cat_df,
                        x="Category",
                        y="Count",
                        title="Category Distribution",
                        tooltip=["Category", "Count"]
                    )

                gr.Markdown("### 🗃️ Accepted Dataset Samples")
                dataset_table = gr.Dataframe(
                    value=load_dataset_dataframe(DEFAULT_DATASET_PATH),
                    headers=["ID", "Category", "Difficulty", "Instruction", "Input", "Output", "Status"],
                    wrap=True,
                    interactive=False
                )

                gr.Markdown("### 🚫 Rejection Log (Failure Analysis)")
                rejections_table = gr.Dataframe(
                    value=load_rejections_dataframe(DEFAULT_REJECTIONS_PATH),
                    headers=["Timestamp", "Topic", "Check Type", "Failure Reason"],
                    wrap=True,
                    interactive=False
                )

                gr.Markdown("### 💾 Export")
                with gr.Row():
                    download_jsonl_btn = gr.Button("📥 Download as JSONL")
                    download_json_btn = gr.Button("📥 Export as Formatted JSON")
                    download_csv_btn = gr.Button("📊 Export as Tabular CSV")

                file_download_component = gr.File(label="Download File", interactive=False)

            # TAB 4: SYSTEM INFO
            with gr.TabItem("🖥️ System & Rules"):
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Detected"
                vram_total = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB" if torch.cuda.is_available() else "N/A"
                gr.Markdown(
                    f"""
                    ### 🛡️ Active Quality Control Rules
                    1. **Schema Validation:** Verifies required fields (`instruction`, `input`, `output`, `difficulty`, `category`).
                    2. **Code Syntax Verification:** Parses code blocks via `ast.parse()` to guarantee syntax error-free Python code.
                    3. **Deduplication:** Exact match and word-level Jaccard token overlap checks (>85% similarity rejected).
                    4. **Refusal Filtering:** Flags canned LLM disclaimers (*"As an AI model..."*).
                    5. **Degeneration Prevention:** Detects repeating loop artifacts in outputs.
                    6. **Diagnostic Rejection Log:** Logs all rejected generations into `data/rejections.jsonl` with failure reasons.

                    ### 💻 Hardware & Model Configuration
                    - **Model:** `{MODEL_ID}` (4-bit NF4 Quantization)
                    - **Detected GPU:** `{gpu_name}`
                    - **VRAM Total:** `{vram_total}`
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
                output_file_box,
                rejections_file_box
            ],
            outputs=[
                log_box,
                out_id,
                out_category,
                out_difficulty,
                out_instruction,
                out_input,
                out_response,
                dataset_table,
                rejections_table,
                file_download_component
            ]
        )

        load_review_btn.click(
            fn=get_sample_for_review,
            inputs=[review_file_box, review_index_slider],
            outputs=[rev_id, rev_cat, rev_diff, rev_inst, rev_inp, rev_out, review_index_slider, review_status_badge]
        )

        approve_btn.click(
            fn=lambda f, i, inst, inp, out: update_sample_review(f, int(i), "approved", inst, inp, out),
            inputs=[review_file_box, review_index_slider, rev_inst, rev_inp, rev_out],
            outputs=[review_feedback_box]
        ).then(
            fn=get_sample_for_review,
            inputs=[review_file_box, review_index_slider],
            outputs=[rev_id, rev_cat, rev_diff, rev_inst, rev_inp, rev_out, review_index_slider, review_status_badge]
        )

        reject_btn.click(
            fn=lambda f, i, inst, inp, out: update_sample_review(f, int(i), "rejected", inst, inp, out),
            inputs=[review_file_box, review_index_slider, rev_inst, rev_inp, rev_out],
            outputs=[review_feedback_box]
        ).then(
            fn=get_sample_for_review,
            inputs=[review_file_box, review_index_slider],
            outputs=[rev_id, rev_cat, rev_diff, rev_inst, rev_inp, rev_out, review_index_slider, review_status_badge]
        )

        save_edit_btn.click(
            fn=lambda f, i, inst, inp, out: update_sample_review(f, int(i), "pending_review", inst, inp, out),
            inputs=[review_file_box, review_index_slider, rev_inst, rev_inp, rev_out],
            outputs=[review_feedback_box]
        )

        def refresh_all(file_path):
            df = load_dataset_dataframe(file_path)
            rej_df = load_rejections_dataframe(DEFAULT_REJECTIONS_PATH)
            stats_text, diff_df, cat_df = compute_dataset_stats(file_path)
            return df, rej_df, stats_text, diff_df, cat_df

        refresh_btn.click(
            fn=refresh_all,
            inputs=[explorer_file_box],
            outputs=[dataset_table, rejections_table, stats_markdown, difficulty_plot, category_plot]
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

        download_csv_btn.click(
            fn=export_as_csv,
            inputs=[explorer_file_box],
            outputs=[file_download_component]
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
