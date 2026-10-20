# Synthetic Dataset Generator with Quality Control

An instruction-following synthetic dataset generator using **Qwen3-1.7B** with 4-bit NF4 quantization (`bitsandbytes`) for local GPU execution (e.g., RTX 3050/4050 mobile).

> **Core Philosophy:** *Valid JSON does not mean valid training data.* Generating 1,000 records doesn't help if 300 are inaccurate, repetitive, or broken.

---

## 🛡️ Multi-Tier Quality Control (QC)

Every generated sample passes through a rigorous automated validation pipeline before being accepted:

| Quality Check | What It Protects Against | Implementation |
| :--- | :--- | :--- |
| **Schema & Type Validation** | Missing keys or empty fields | Ensures `instruction`, `input`, `output`, `difficulty`, `category` are valid strings with minimum length thresholds. |
| **Code Syntax Validation** | Broken or hallucinated Python code | Extracts ```python ... ``` blocks and parses them with Python's Abstract Syntax Tree (`ast.parse`). Rejects syntax errors. |
| **Deduplication (Exact & Fuzzy)** | Repeated or redundant questions | Checks exact `(instruction, input)` hashes and rejects near-duplicates using Jaccard word-level similarity (>85% overlap). |
| **Refusal / Disclaimer Filtering** | Canned model non-answers | Flags and rejects outputs containing *"As an AI model..."* or *"I cannot assist..."*. |
| **Degeneration Check** | Infinite token repeating loops | Detects runaway line loops in generated outputs. |
| **Balance Tracking** | Unbalanced datasets | Categorizes records by domain and difficulty (`beginner`, `intermediate`, `advanced`) with distribution metrics. |
| **Rejection Logging** | Unclear failure sources | Logs all rejected attempts and reasons into `rejections.jsonl` to diagnose whether failures stem from prompts, models, or validators. |
| **Human Review / Audit** | Plausible-sounding but wrong answers | Interactive Gradio Human-in-the-Loop review UI to inspect, approve, edit, or reject samples. |

---

## Project Structure
```
├── app.py               # Gradio web studio (Generator + Human Review + Explorer + QC)
├── validator.py         # Multi-tier Quality Control & AST validation engine
├── loader.py            # Loads quantized Qwen3-1.7B model and tokenizer
├── prompts.py           # Seed topics, balanced formatting prompts, and system instructions
├── utils.py             # JSON extraction and repair heuristics
├── generator.py         # CLI batch generation loop with QC enforcement
├── inspect_dataset.py   # Terminal inspector for saved samples
├── requirements.txt     # Python dependencies
├── synthetic_dataset.jsonl # Verified output dataset
└── rejections.jsonl     # Diagnostic log of rejected generations
```

## Setup & Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   # Windows PowerShell:
   ./myenv/Scripts/activate
   ```

2. Install PyTorch with CUDA:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Web Studio (Recommended)
Launch the interactive web UI:
```bash
python app.py
```
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) to:
- Generate QC-verified samples in real-time.
- Audit samples with Human Review (`Approve`, `Reject`, `Edit`).
- Inspect Category & Difficulty distributions.
- Review the Rejection Log to analyze failed generations.
- Export as `.jsonl` or `.json`.

### 2. CLI Batch Generation
Run the automated generation loop with QC validation:
```bash
python generator.py --samples 20 --output synthetic_dataset.jsonl --rejections rejections.jsonl
```

### 3. Inspect in Terminal
```bash
python inspect_dataset.py synthetic_dataset.jsonl
```
