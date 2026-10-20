# Synthetic Dataset Generator

An instruction-following synthetic dataset generator using **Qwen3-1.7B** with 4-bit NF4 quantization (`bitsandbytes`) for local GPU execution (e.g., RTX 3050/4050 mobile).

Built as part of the **AI Engineer Core Track** (LLM Engineering, RAG, QLoRA, Agents).

## Features
- **Interactive Gradio UI:** Web interface to generate, preview, explore, and download datasets.
- **Local GPU Inference:** Runs on consumer GPUs using 4-bit NormalFloat (`NF4`) quantization via Hugging Face `transformers` and `bitsandbytes`.
- **Alpaca Format:** Generates standard instruction-tuning examples (`instruction`, `input`, `output`).
- **Deduplication:** Automatic deduplication on normalized `(instruction, input)` pairs.
- **Resumable Saving:** Pre-populates cache from existing `.jsonl` files so appending new samples never creates duplicates.
- **Robust JSON Extraction:** Strips model thinking blocks (`<think>`), handles nested code fences, and repairs unescaped characters.
- **Metadata Tracking:** Generates unique sample IDs (`synth_...`) and logs the generating `model_id`.

## Project Structure
```
├── app.py               # Gradio web user interface
├── loader.py            # Loads quantized Qwen3-1.7B model and tokenizer
├── prompts.py           # Seed topics and Alpaca formatting prompt templates
├── utils.py             # JSON extraction, repair heuristics, and schema validation
├── generator.py         # Batch generation CLI script with deduplication & progress bar
├── inspect_dataset.py   # Pretty-printer to inspect generated dataset samples
├── requirements.txt     # Python dependencies
└── synthetic_dataset.jsonl # Output dataset
```

## Setup & Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   # Windows PowerShell:
   ./myenv/Scripts/activate
   ```

2. Install PyTorch with CUDA (check your CUDA version):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Web Interface (Gradio UI)
Launch the interactive web studio:
```bash
python app.py
```
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser to:
- Generate samples with custom or random seed topics.
- Inspect latest outputs in real-time.
- View and browse all stored samples in an interactive table.
- Export and download your dataset as `.jsonl` or formatted `.json`.

### Option 2: CLI Batch Generator
```bash
python generator.py --samples 20 --output synthetic_dataset.jsonl
```

### Option 3: Terminal Inspector
Inspect saved samples in your terminal:
```bash
python inspect_dataset.py synthetic_dataset.jsonl
```
