import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def inspect(file_path: str = "data/synthetic_dataset.jsonl", max_show: int = 5):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: {file_path} not found.")
        return

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"\n=== Synthetic Dataset Inspection: {file_path} ===")
    print(f"Total entries: {len(lines)}\n")

    for idx, line in enumerate(lines[:max_show], 1):
        try:
            item = json.loads(line)
        except Exception as e:
            print(f"Error parsing line {idx}: {e}")
            continue

        sample_id = item.get("id", "N/A")
        model_id = item.get("model_id", "N/A")
        topic = item.get("seed_topic", "N/A")

        print(f"--- [Sample {idx}] ID: {sample_id} | Model: {model_id} ---")
        print(f"🏷️  TOPIC:       {topic}")
        print(f"📌 INSTRUCTION:\n{item.get('instruction', '')}\n")
        
        inp = item.get('input', '')
        if inp:
            print(f"📥 INPUT CONTEXT:\n{inp}\n")
            
        print(f"💡 OUTPUT RESPONSE:\n{item.get('output', '')}\n")
        print("=" * 65 + "\n")

if __name__ == "__main__":
    file_to_inspect = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_dataset.jsonl"
    inspect(file_to_inspect)
