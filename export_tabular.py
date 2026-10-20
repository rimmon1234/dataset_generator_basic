import json
import sys
from pathlib import Path
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def convert_jsonl_to_tabular(
    input_jsonl: str = "synthetic_dataset.jsonl",
    output_format: str = "csv"
) -> Path:
    """
    Converts a JSONL synthetic dataset into tabular formats: CSV or Parquet.
    """
    in_path = Path(input_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"File not found: {input_jsonl}")

    records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

    if not records:
        print("Warning: No records found in file.")
        return None

    df = pd.DataFrame(records)

    # Reorder columns logically
    priority_cols = ["id", "category", "difficulty", "seed_topic", "instruction", "input", "output", "model_id", "review_status"]
    ordered_cols = [c for c in priority_cols if c in df.columns] + [c for c in df.columns if c not in priority_cols]
    df = df[ordered_cols]

    if output_format.lower() == "parquet":
        out_path = in_path.with_suffix(".parquet")
        # requires pyarrow or fastparquet if installed, fallback to csv if missing
        try:
            df.to_parquet(out_path, index=False)
            print(f"✅ Successfully exported Parquet: {out_path.resolve()}")
            return out_path
        except ImportError:
            print("pyarrow not installed, falling back to CSV.")
            out_path = in_path.with_suffix(".csv")
            df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"✅ Successfully exported CSV: {out_path.resolve()}")
            return out_path
    else:
        out_path = in_path.with_suffix(".csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✅ Successfully exported CSV ({len(df)} rows): {out_path.resolve()}")
        return out_path

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "synthetic_dataset.jsonl"
    fmt = sys.argv[2] if len(sys.argv) > 2 else "csv"
    convert_jsonl_to_tabular(src, fmt)
