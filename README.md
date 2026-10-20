# Synthetic Dataset Generator

An instruction-following synthetic dataset generator using **Qwen3-1.7B** with 4-bit NF4 quantization (`bitsandbytes`) for local GPU execution (e.g., RTX 3050/4050 mobile).

Built as part of the **AI Engineer Core Track** (LLM Engineering, RAG, QLoRA, Agents).

## Features
- **Local Inference:** Runs on consumer GPUs using 4-bit NormalFloat (`NF4`) quantization via Hugging Face `transformers` and `bitsandbytes`.
- **Alpaca Format:** Generates standard instruction-tuning examples (`instruction`, `input`, `output`).
- **Deduplication:** Automatic deduplication on normalized `(instruction, input)` pairs.
- **Resumable:** Pre-populates cache from existing `.jsonl` files so appending new samples never creates duplicates.
- **Robust JSON Extraction:** Strips model thinking blocks (`<think>`), handles nested code fences, and repairs unescaped characters.
- **Metadata Tracking:** Generates unique sample IDs (`synth_...`) and logs the generating `model_id`.

## Project Structure
```
├── loader.py            # Loads quantized Qwen3-1.7B model and tokenizer
├── prompts.py           # Seed topics and Alpaca formatting prompt templates
├── utils.py             # JSON extraction, repair heuristics, and schema validation
├── generator.py         # Main batch generation loop with deduplication & progress bar
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

### 1. Generate Synthetic Data
```bash
python generator.py --samples 20 --output synthetic_dataset.jsonl
```

Options:
- `--samples`: Target number of unique valid examples (default: `5`).
- `--output`: Filepath for output JSONL (default: `synthetic_dataset.jsonl`).
- `--temperature`: Creativity/diversity parameter (default: `0.7`).
- `--max-tokens`: Maximum new tokens generated (default: `896`).

### 2. Inspect Generated Examples
```bash
python inspect_dataset.py synthetic_dataset.jsonl
```
