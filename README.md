# 🧠 Synthetic Dataset Generator Studio

A modular, instruction-following synthetic dataset generator using **Qwen3-1.7B** with 4-bit NF4 quantization (`bitsandbytes`) for local GPU execution (e.g., RTX 3050/4050 mobile).

> **Core Philosophy:** *Valid JSON does not mean valid training data.* Generating 1,000 records doesn't help if 300 are inaccurate, repetitive, or broken.

---

## 📁 Clean & Modular Architecture

```
synthetic_dataset_generator/
│
├── data/                            # Datasets & diagnostic logs
│   ├── synthetic_dataset.jsonl      # Validated instruction dataset
│   ├── synthetic_dataset.csv        # Tabular CSV export
│   └── rejections.jsonl             # Diagnostic log of rejected attempts
│
├── src/                             # Core Python package
│   ├── core/                        # LLM Engine & Generation
│   │   ├── loader.py                # 4-bit NF4 model & tokenizer loader
│   │   ├── generator.py             # Generation loop with QC checks
│   │   └── prompts.py               # Seed topics, schemas, and filler bans
│   │
│   ├── qc/                          # Quality Control & Validation
│   │   ├── validator.py             # 6-tier QC validation engine
│   │   └── utils.py                 # JSON extraction & filler cleaner
│   │
│   └── tools/                       # Analytics & Exporters
│       ├── exporter.py              # JSONL to CSV/Parquet converter
│       ├── visualizer.py            # Terminal ASCII distribution visualizer
│       └── inspector.py             # Formatted sample card viewer
│
├── app.py                           # Interactive Gradio Web Studio
├── main.py                          # Unified CLI entrypoint
├── requirements.txt                 # Pinned dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # Documentation
```

---

## 🛡️ Multi-Tier Quality Control (QC)

Every generated sample passes through an automated validation pipeline before being accepted:

| Quality Check | What It Protects Against | Implementation |
| :--- | :--- | :--- |
| **Schema & Type Validation** | Missing keys or empty fields | Ensures `instruction`, `input`, `output`, `difficulty`, `category` are valid strings with length thresholds. |
| **Code Syntax Validation** | Broken or hallucinated Python code | Extracts ```python ... ``` blocks and compiles them via Python's Abstract Syntax Tree (`ast.parse`). Rejects syntax errors. |
| **Deduplication (Exact & Fuzzy)** | Repeated or redundant questions | Checks exact `(instruction, input)` hashes and rejects near-duplicates using Jaccard word-level similarity (>85% overlap). |
| **Refusal & Disclaimer Filter** | Canned model non-answers | Flags and rejects outputs containing *"As an AI model..."* or *"I cannot assist..."*. |
| **Degeneration Check** | Infinite token repeating loops | Detects runaway line loops in generated outputs. |
| **Balance Tracking** | Unbalanced datasets | Categorizes records by domain and difficulty (`beginner`, `intermediate`, `advanced`) with distribution metrics. |
| **Rejection Logging** | Unclear failure sources | Logs all rejected attempts and reasons into `data/rejections.jsonl` to diagnose whether failures stem from prompts, models, or validators. |
| **Human Review / Audit** | Plausible-sounding but wrong answers | Interactive Gradio Human-in-the-Loop review UI to inspect, approve, edit, or reject samples. |

---

## 🚀 Getting Started

### 1. Installation
```bash
python -m venv myenv
# Windows PowerShell:
./myenv/Scripts/activate

# Install PyTorch with CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install dependencies:
pip install -r requirements.txt
```

---

## 💻 Usage

### Option A: Launch Interactive Web Studio (Recommended)
```bash
python main.py --ui
# or directly:
python app.py
```
Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** to:
- Generate samples with custom or random seed topics.
- Audit samples with Human Review (`Approve`, `Reject`, `Edit`).
- Inspect Category & Difficulty distributions with live interactive bar charts.
- Review the Rejection Log to analyze failed generations.
- Export to `.jsonl`, formatted `.json`, or tabular `.csv`.

---

### Option B: Unified Command-Line (CLI)

```bash
# Generate 10 verified samples
python main.py --samples 10

# Generate with a custom domain topic
python main.py --samples 5 --topic "Medical pharmacotherapy and drug interactions"

# Visualize dataset distributions & QC report
python main.py --visualize

# Inspect individual sample cards in terminal
python main.py --inspect

# Export dataset to tabular CSV
python main.py --export csv
```
