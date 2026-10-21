"""
Unified CLI Entrypoint for the Synthetic Dataset Generator Studio.
"""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src package is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.generator import run_generator
from src.core.loader import MODEL_ID
from src.tools.visualizer import visualize
from src.tools.inspector import inspect
from src.tools.exporter import convert_jsonl_to_tabular

def main():
    parser = argparse.ArgumentParser(
        description="🧠 Synthetic Dataset Generator Studio (CLI & UI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --samples 10               # Generate 10 verified samples
  python main.py --ui                       # Launch interactive Gradio Web Studio
  python main.py --visualize                # View ASCII dataset quality & balance report
  python main.py --inspect                  # Pretty-print sample instruction cards
  python main.py --export csv               # Convert data/synthetic_dataset.jsonl to CSV
        """
    )

    parser.add_argument("--ui", action="store_true", help="Launch the Gradio Web Studio")
    parser.add_argument("--samples", type=int, default=None, help="Number of verified samples to generate")
    parser.add_argument("--topic", type=str, default=None, help="Custom topic to generate (overrides random seed)")
    parser.add_argument("--output", type=str, default="data/synthetic_dataset.jsonl", help="Target output JSONL path")
    parser.add_argument("--rejections", type=str, default="data/rejections.jsonl", help="Target rejections log path")
    parser.add_argument("--visualize", action="store_true", help="Visualize dataset distributions & QC report")
    parser.add_argument("--inspect", action="store_true", help="Inspect individual dataset samples in terminal")
    parser.add_argument("--export", choices=["csv", "parquet"], default=None, help="Export dataset to tabular format")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (creativity)")
    parser.add_argument("--max-tokens", type=int, default=896, help="Maximum generated tokens per sample")

    args = parser.parse_args()

    # If --ui or no action args provided, launch UI or print help
    if args.ui:
        import app
        demo = app.create_ui()
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
        return

    if args.visualize:
        visualize(args.output, args.rejections)
        return

    if args.inspect:
        inspect(args.output)
        return

    if args.export:
        convert_jsonl_to_tabular(args.output, args.export)
        return

    if args.samples is not None:
        run_generator(
            total_samples=args.samples,
            output_path=args.output,
            rejections_path=args.rejections,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            model_id=MODEL_ID
        )
        return

    # If no flags provided, show help
    parser.print_help()

if __name__ == "__main__":
    main()
