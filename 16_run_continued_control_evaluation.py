# =================================================
# ./16_run_continued_control_evaluation.py
# =================================================

import argparse
import hashlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import yaml

from src.evaluation.evaluate_continued_control import (
    evaluate_continued_control_model,
)
from src.utils.config import load_config


# ==============================================================================
# AUXILIARY FUNCTIONS
# ==============================================================================


def _model_tag(src: str, tgt: str, encoder_layers: int, decoder_layers: int, seed: int) -> str:
    return f"{src}_to_{tgt}_Enc{encoder_layers}_Dec{decoder_layers}_Seed{seed}"


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _atomic_yaml_dump(payload: dict, output_path: Path) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _sha256_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File required for compatibility audit is missing: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return payload


def _load_experiment_grid(cfg: dict) -> tuple[list[int], list[dict], list[dict]]:
    seeds = [int(value) for value in cfg["experiment_seeds"]]
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("experiment_seeds must contain unique non-negative integers.")

    directions = []
    for item in cfg["experiment_directions"]:
        src = str(item["src"]).strip()
        tgt = str(item["tgt"]).strip()
        if not src or not tgt or src == tgt:
            raise ValueError("Each experiment direction must contain two different languages.")
        directions.append({"src": src, "tgt": tgt})
    if len({(x["src"], x["tgt"]) for x in directions}) != len(directions):
        raise ValueError("experiment_directions contains duplicates.")

    architectures = []
    for item in cfg["experiment_architectures"]:
        enc = int(item["encoder_layers"])
        dec = int(item["decoder_layers"])
        if enc <= 0 or dec <= 0 or enc + dec != 8:
            raise ValueError("Each architecture must preserve the fixed eight-layer budget.")
        architectures.append({"encoder_layers": enc, "decoder_layers": dec})
    if len({(x["encoder_layers"], x["decoder_layers"]) for x in architectures}) != len(architectures):
        raise ValueError("experiment_architectures contains duplicates.")

    return seeds, directions, architectures


def _build_effective_run_config(
    cfg: dict,
    src: str,
    tgt: str,
    encoder_layers: int,
    decoder_layers: int,
    seed: int,
) -> dict:
    run_cfg = deepcopy(cfg)
    run_cfg["experiment_seed"] = int(seed)
    run_cfg["src"] = src
    run_cfg["tgt"] = tgt
    run_cfg["encoder_layers"] = int(encoder_layers)
    run_cfg["decoder_layers"] = int(decoder_layers)
    return run_cfg


def _validate_existing_evaluation(
    project_root: Path,
    cfg: dict,
    model_tag: str,
    checkpoint_path: Path,
    metrics_path: Path,
    biblical_predictions_path: Path,
    conversational_predictions_path: Path,
    src: str,
    tgt: str,
    encoder_layers: int,
    decoder_layers: int,
    seed: int,
) -> None:
    metrics = _load_json(metrics_path)
    expected_cfg = _build_effective_run_config(
        cfg, src, tgt, encoder_layers, decoder_layers, seed
    )
    mismatches: list[str] = []

    scalar_checks = {
        "model_tag": (metrics.get("model_tag"), model_tag),
        "src": (metrics.get("src"), src),
        "tgt": (metrics.get("tgt"), tgt),
        "experiment_seed": (metrics.get("experiment_seed"), int(seed)),
        "evaluation.max_len": (
            metrics.get("evaluation", {}).get("max_len"),
            int(expected_cfg["ct_test_max_len"]),
        ),
        "model.architecture.encoder_layers": (
            metrics.get("model", {}).get("architecture", {}).get("encoder_layers"),
            int(encoder_layers),
        ),
        "model.architecture.decoder_layers": (
            metrics.get("model", {}).get("architecture", {}).get("decoder_layers"),
            int(decoder_layers),
        ),
    }
    for label, (stored, current) in scalar_checks.items():
        if stored != current:
            mismatches.append(f"{label}: stored={stored!r}, current={current!r}")

    control_cfg = metrics.get("control_config", {})
    for key in (
        "experiment_seed",
        "src",
        "tgt",
        "encoder_layers",
        "decoder_layers",
        "ct_subset_size",
        "ct_subset_seed",
        "ct_budget_source",
        "ct_optimizer",
        "ct_learning_rate",
        "ct_scheduler",
        "ct_scheduler_eta_min",
        "ct_scheduler_step_unit",
        "ct_test_max_len",
    ):
        if control_cfg.get(key) != expected_cfg.get(key):
            mismatches.append(
                f"control_config.{key}: stored={control_cfg.get(key)!r}, "
                f"current={expected_cfg.get(key)!r}"
            )

    tokenized_dir = _resolve_project_path(project_root, expected_cfg["tokenized_dir"])
    tokenizer_path = _resolve_project_path(project_root, expected_cfg["tokenizer_dir"]) / "SentencePiece.model"
    current_hashes = {
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "sentencepiece_model_sha256": _sha256_file(tokenizer_path),
        "test_bible_tokenized_sha256": _sha256_file(
            tokenized_dir / "test_bible_tokenized.tsv"
        ),
        "test_dialogue_tokenized_sha256": _sha256_file(
            tokenized_dir / "test_dialogue_tokenized.tsv"
        ),
        "biblical_predictions_sha256": _sha256_file(biblical_predictions_path),
        "conversational_predictions_sha256": _sha256_file(
            conversational_predictions_path
        ),
    }
    stored_hashes = metrics.get("evaluation_hashes", {})
    for key, current_hash in current_hashes.items():
        if stored_hashes.get(key) != current_hash:
            mismatches.append(
                f"evaluation_hashes.{key}: stored={stored_hashes.get(key)!r}, "
                f"current={current_hash!r}"
            )

    checkpoint_meta_hash = metrics.get("checkpoint", {}).get("sha256")
    if checkpoint_meta_hash != current_hashes["checkpoint_sha256"]:
        mismatches.append(
            f"checkpoint.sha256: stored={checkpoint_meta_hash!r}, "
            f"current={current_hashes['checkpoint_sha256']!r}"
        )

    if mismatches:
        text = "\n".join(f"  - {item}" for item in mismatches)
        raise RuntimeError(
            f"Complete continued-control evaluation artifacts exist for {model_tag}, "
            f"but they are not compatible with the current configuration and inputs.\n"
            f"Mismatches:\n{text}\n"
            "Do not delete or overwrite them automatically. Inspect the run first; "
            "use --force only when regeneration is intentional."
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate every continued-training control checkpoint."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing continued-control evaluation outputs intentionally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without running inference.",
    )
    return parser


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    main_config_path = project_root / "config.yaml"
    temporary_config_path = project_root / ".continued_control_evaluation_config.yaml"
    if not main_config_path.exists():
        raise FileNotFoundError(f"Main configuration not found: {main_config_path}")

    cfg = load_config(main_config_path)
    seeds, directions, architectures = _load_experiment_grid(cfg)
    models_dir = _resolve_project_path(project_root, cfg["models_dir"])
    results_dir = _resolve_project_path(project_root, cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(seeds) * len(directions) * len(architectures)

    missing_checkpoints = []
    for architecture in architectures:
        for direction in directions:
            for seed in seeds:
                tag = _model_tag(
                    direction["src"],
                    direction["tgt"],
                    architecture["encoder_layers"],
                    architecture["decoder_layers"],
                    seed,
                )
                path = models_dir / f"continued_control_checkpoint_{tag}.pt"
                if not path.exists():
                    missing_checkpoints.append(path)

    if missing_checkpoints:
        text = "\n".join(f"  - {path}" for path in missing_checkpoints)
        raise FileNotFoundError(
            "Continued-control evaluation cannot start because some checkpoints "
            f"are missing:\n{text}"
        )

    print("\n" + "=" * 80)
    print("CONTINUED-CONTROL EVALUATION GRID")
    print("=" * 80)
    print(f"Seeds          : {seeds}")
    print(
        "Directions     : "
        + ", ".join(f"{x['src']}->{x['tgt']}" for x in directions)
    )
    print(
        "Architectures  : "
        + ", ".join(
            f"{x['encoder_layers']}E-{x['decoder_layers']}D" for x in architectures
        )
    )
    print(f"Total runs     : {total_runs}")
    print(f"Models dir     : {cfg['models_dir']}")
    print(f"Results dir    : {cfg['results_dir']}")
    print(f"Force          : {args.force}")
    print(f"Dry run        : {args.dry_run}")
    print("=" * 80)

    completed = 0
    skipped = 0
    run_number = 0
    overall_start = time.perf_counter()

    try:
        for architecture in architectures:
            enc = int(architecture["encoder_layers"])
            dec = int(architecture["decoder_layers"])
            for direction in directions:
                src = direction["src"]
                tgt = direction["tgt"]
                for seed in seeds:
                    run_number += 1
                    tag = _model_tag(src, tgt, enc, dec, seed)
                    checkpoint_path = models_dir / f"continued_control_checkpoint_{tag}.pt"
                    metrics_path = results_dir / f"continued_control_test_metrics_{tag}.json"
                    biblical_predictions_path = (
                        results_dir / f"continued_control_predictions_biblical_{tag}.tsv"
                    )
                    conversational_predictions_path = (
                        results_dir / f"continued_control_predictions_conversational_{tag}.tsv"
                    )
                    expected = [
                        metrics_path,
                        biblical_predictions_path,
                        conversational_predictions_path,
                    ]
                    existing = [path for path in expected if path.exists()]

                    print("\n" + "-" * 80)
                    print(f"EVALUATION {run_number}/{total_runs}: {tag}")
                    print("-" * 80)

                    if len(existing) == len(expected) and not args.force:
                        _validate_existing_evaluation(
                            project_root,
                            cfg,
                            tag,
                            checkpoint_path,
                            metrics_path,
                            biblical_predictions_path,
                            conversational_predictions_path,
                            src,
                            tgt,
                            enc,
                            dec,
                            seed,
                        )
                        print(
                            "[SKIP] Compatible continued-control evaluation artifacts "
                            "already exist."
                        )
                        skipped += 1
                        continue

                    if existing and not args.force:
                        existing_text = "\n".join(f"  - {p}" for p in existing)
                        missing_text = "\n".join(f"  - {p}" for p in expected if not p.exists())
                        raise RuntimeError(
                            f"Partial continued-control evaluation artifacts detected for {tag}.\n"
                            f"Existing:\n{existing_text}\nMissing:\n{missing_text}\n"
                            "Inspect the run before continuing; use --force only intentionally."
                        )

                    if args.dry_run:
                        print(f"Checkpoint: {checkpoint_path}")
                        print("[DRY RUN] Continued-control evaluation would start here.")
                        continue

                    if args.force:
                        for path in expected:
                            if path.exists():
                                path.unlink()
                        print("[FORCE] Previous continued-control evaluation artifacts removed.")

                    run_cfg = _build_effective_run_config(cfg, src, tgt, enc, dec, seed)
                    _atomic_yaml_dump(run_cfg, temporary_config_path)
                    print(f"Checkpoint       : {checkpoint_path}")
                    print(f"Temporary config : {temporary_config_path.name}")
                    print("Starting continued-control evaluation...")
                    run_start = time.perf_counter()
                    evaluate_continued_control_model(
                        config_path=temporary_config_path,
                        checkpoint_path=checkpoint_path,
                    )
                    run_seconds = time.perf_counter() - run_start

                    missing_after = [path for path in expected if not path.exists()]
                    if missing_after:
                        text = "\n".join(f"  - {path}" for path in missing_after)
                        raise RuntimeError(
                            f"Continued-control evaluation returned without all expected "
                            f"artifacts for {tag}.\nMissing:\n{text}"
                        )

                    completed += 1
                    print(f"[DONE] {tag}")
                    print(f"Evaluation execution time: {run_seconds / 3600.0:.2f} h")

    finally:
        if temporary_config_path.exists():
            temporary_config_path.unlink()
            print(
                f"\n[Cleanup] Temporary configuration removed: "
                f"{temporary_config_path.name}"
            )

    elapsed = time.perf_counter() - overall_start
    print("\n" + "=" * 80)
    print("CONTINUED-CONTROL EVALUATION GRID FINISHED")
    print("=" * 80)
    print(f"Total planned runs : {total_runs}")
    print(f"Completed now      : {completed}")
    print(f"Skipped existing   : {skipped}")
    print(f"Elapsed time       : {elapsed / 3600.0:.2f} h")
    print("=" * 80)


if __name__ == "__main__":
    main()
