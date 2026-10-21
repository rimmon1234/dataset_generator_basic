import json
import sys
from pathlib import Path
from collections import Counter

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def draw_bar(count: int, total: int, width: int = 30) -> str:
    if total == 0:
        return ""
    filled = int((count / total) * width)
    percentage = (count / total) * 100
    return f"{'█' * filled}{'░' * (width - filled)} {percentage:5.1f}% ({count})"

def visualize(file_path: str = "data/synthetic_dataset.jsonl", rejections_path: str = "data/rejections.jsonl"):
    data_path = Path(file_path)
    if not data_path.exists():
        print(f"Error: {file_path} not found.")
        return

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

    total = len(records)
    print("=" * 70)
    print(f"📊 DATASET VISUALIZER & QUALITY REPORT: {file_path}")
    print("=" * 70)
    print(f"Total Valid Records: {total}\n")

    if total == 0:
        print("Dataset is currently empty.")
        return

    # 1. Review Status Distribution
    statuses = Counter(r.get("review_status", "pending_review") for r in records)
    print("📋 [1] HUMAN REVIEW STATUS:")
    for status, count in statuses.items():
        print(f"  {status.upper():16} {draw_bar(count, total)}")
    print()

    # 2. Difficulty Distribution
    difficulties = Counter(r.get("difficulty", "intermediate") for r in records)
    print("🎯 [2] DIFFICULTY BALANCE:")
    for diff in ["beginner", "intermediate", "advanced"]:
        count = difficulties.get(diff, 0)
        print(f"  {diff.capitalize():16} {draw_bar(count, total)}")
    print()

    # 3. Category Breakdown
    categories = Counter(r.get("category", r.get("seed_topic", "General")) for r in records)
    print("🏷️  [3] CATEGORY BREAKDOWN:")
    for cat, count in categories.most_common(10):
        print(f"  {cat[:20]:20} {draw_bar(count, total)}")
    print()

    # 4. Length / Token Analytics
    inst_lengths = [len(r.get("instruction", "").split()) for r in records]
    out_lengths = [len(r.get("output", "").split()) for r in records]
    code_count = sum(1 for r in records if "```" in r.get("output", ""))

    avg_inst = sum(inst_lengths) / total if total > 0 else 0
    avg_out = sum(out_lengths) / total if total > 0 else 0

    print("📏 [4] LENGTH & CODE METRICS:")
    print(f"  Average Instruction Length:  {avg_inst:.1f} words (min: {min(inst_lengths)}, max: {max(inst_lengths)})")
    print(f"  Average Output Length:       {avg_out:.1f} words (min: {min(out_lengths)}, max: {max(out_lengths)})")
    print(f"  Samples with Python Code:    {code_count}/{total} ({code_count/total*100:.1f}%)")
    print()

    # 5. QC Rejection Log Summary
    rej_path = Path(rejections_path)
    if rej_path.exists():
        rejections = []
        with open(rej_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rejections.append(json.loads(line))
                    except Exception:
                        continue
        rej_total = len(rejections)
        print(f"🚫 [5] QC REJECTIONS ANALYSIS ({rej_total} total rejected attempts):")
        reasons = Counter(r.get("reason", "Unknown") for r in rejections)
        for r, count in reasons.most_common(5):
            print(f"  - [{count}x] {r}")
    print("=" * 70)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_dataset.jsonl"
    visualize(target)
