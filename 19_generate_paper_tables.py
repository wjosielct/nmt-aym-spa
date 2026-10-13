# ==================================
# ./19_generate_paper_tables.py
# ==================================

import argparse
from pathlib import Path

from src.analysis.paper_tables import generate_paper_tables


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate five reproducible LaTeX result tables from the "
            "validated dataset splits and public result artifacts."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_tables",
        help="Output directory for generated LaTeX tables (default: paper_tables).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing compatible table set intentionally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate all inputs and print the table-generation plan without "
            "writing any output files."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parent

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    generate_paper_tables(
        config_path=config_path,
        output_dir=output_dir,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
