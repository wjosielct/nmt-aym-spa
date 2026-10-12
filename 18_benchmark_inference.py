# ===============================
# ./18_benchmark_inference.py
# ===============================

import argparse
from pathlib import Path

from src.benchmarking.inference_benchmark import run_inference_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark inference latency, generated-token throughput, and peak "
            "GPU memory for the prespecified base/biblical and "
            "fine-tuned/conversational architecture conditions."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing benchmark outputs intentionally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the 36-run benchmark plan without GPU timing.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    run_inference_benchmark(
        config_path=config_path,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
