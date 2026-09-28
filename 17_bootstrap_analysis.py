# ==============================
# ./17_bootstrap_analysis.py
# ==============================

import argparse
from pathlib import Path

from src.analysis.paired_bootstrap import run_bootstrap_analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired bootstrap 95% confidence intervals for the "
            "prespecified base/adapted/control and architecture contrasts "
            "using domain-specific resampling units."
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
        help="Regenerate compatible/incompatible outputs intentionally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the contrast plan without bootstrapping.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    run_bootstrap_analysis(
        config_path=config_path,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
