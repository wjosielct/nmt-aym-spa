# ================================================
# ./15_run_continued_training_control.py
# ================================================

import argparse
import hashlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import yaml

from src.training.continue_bible_control import run_continued_bible_control
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


def _sha256_integer_sequence(values: list[int]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return payload


def _sample_indices(population_size: int, subset_size: int, seed: int) -> list[int]:
    import random

    if subset_size > population_size:
        raise ValueError("ct_subset_size exceeds train_bible size.")
    rng = random.Random(seed)
    values = rng.sample(range(population_size), subset_size)
    values.sort()
    return values


def _count_tsv_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as file:
        return max(sum(1 for _ in file) - 1, 0)


def _load_experiment_grid(cfg: dict) -> tuple[list[int], list[dict], list[dict]]:
    for key in ("experiment_seeds", "experiment_directions", "experiment_architectures"):
        if key not in cfg:
            raise KeyError(f"Missing experiment-grid parameter: {key}")

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


_CONTROL_CONFIG_KEYS = [
    "d_model",
    "n_heads",
    "encoder_layers",
    "decoder_layers",
    "dim_feedforward",
    "positional_encoding_max_len",
    "base_dropout",
    "ct_subset_size",
    "ct_subset_seed",
    "ct_budget_source",
    "ct_batch_size_cuda",
    "ct_batch_size_cpu",
    "ct_shuffle_train",
    "ct_drop_last",
    "ct_optimizer",
    "ct_learning_rate",
    "ct_adam_beta1",
    "ct_adam_beta2",
    "ct_adam_eps",
    "ct_weight_decay",
    "ct_label_smoothing",
    "ct_grad_clip_norm",
    "ct_scheduler",
    "ct_scheduler_eta_min",
    "ct_scheduler_step_unit",
    "ct_amp_enabled",
    "ct_num_workers",
    "ct_pin_memory",
]


def _compare_config(stored: dict, current: dict, mismatches: list[str]) -> None:
    if not isinstance(stored, dict):
        mismatches.append("control_config: missing or invalid")
        return
    for key in ["experiment_seed", "src", "tgt"] + _CONTROL_CONFIG_KEYS:
        if stored.get(key) != current.get(key):
            mismatches.append(
                f"control_config.{key}: stored={stored.get(key)!r}, current={current.get(key)!r}"
            )


def _validate_existing_control_artifacts(
    project_root: Path,
    cfg: dict,
    model_tag: str,
    base_checkpoint_path: Path,
    finetuned_checkpoint_path: Path,
    summary_path: Path,
    src: str,
    tgt: str,
    encoder_layers: int,
    decoder_layers: int,
    seed: int,
) -> None:
    summary = _load_json(summary_path)
    expected_cfg = _build_effective_run_config(
        cfg, src, tgt, encoder_layers, decoder_layers, seed
    )
    mismatches: list[str] = []

    if summary.get("model_tag") != model_tag:
        mismatches.append(
            f"model_tag: stored={summary.get('model_tag')!r}, current={model_tag!r}"
        )
    if summary.get("experiment_seed") != int(seed):
        mismatches.append(
            f"experiment_seed: stored={summary.get('experiment_seed')!r}, current={seed!r}"
        )
    _compare_config(summary.get("config", {}), expected_cfg, mismatches)

    tokenizer_path = _resolve_project_path(project_root, expected_cfg["tokenizer_dir"]) / "SentencePiece.model"
    tokenized_dir = _resolve_project_path(project_root, expected_cfg["tokenized_dir"])
    bible_train_path = tokenized_dir / "train_bible_tokenized.tsv"
    dialogue_train_path = tokenized_dir / "train_dialogue_tokenized.tsv"

    population_size = _count_tsv_rows(bible_train_path)
    dialogue_size = _count_tsv_rows(dialogue_train_path)
    subset_size = int(expected_cfg["ct_subset_size"])
    if subset_size != dialogue_size:
        mismatches.append(
            f"ct_subset_size: current={subset_size}, dialogue_train_pairs={dialogue_size}"
        )
    selected_indices_hash = _sha256_integer_sequence(
        _sample_indices(population_size, subset_size, int(expected_cfg["ct_subset_seed"]))
    )

    current_hashes = {
        "base_checkpoint_sha256": _sha256_file(base_checkpoint_path),
        "finetuned_checkpoint_sha256": _sha256_file(finetuned_checkpoint_path),
        "sentencepiece_model_sha256": _sha256_file(tokenizer_path),
        "train_bible_tokenized_sha256": _sha256_file(bible_train_path),
        "train_dialogue_tokenized_sha256": _sha256_file(dialogue_train_path),
        "selected_bible_subset_indices_sha256": selected_indices_hash,
    }
    stored_hashes = summary.get("input_hashes", {})
    for key, current_hash in current_hashes.items():
        if stored_hashes.get(key) != current_hash:
            mismatches.append(
                f"input_hashes.{key}: stored={stored_hashes.get(key)!r}, current={current_hash!r}"
            )

    if mismatches:
        text = "\n".join(f"  - {item}" for item in mismatches)
        raise RuntimeError(
            f"Complete continued-control artifacts exist for {model_tag}, but they are not "
            f"compatible with the current configuration and inputs.\nMismatches:\n{text}\n"
            "Do not delete or overwrite them automatically. Inspect the run first; "
            "use --force only when replacement is intentional."
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the size-matched continued-biblical-training control for "
            "every experiment-grid condition."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace completed continued-control artifacts intentionally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without starting optimization.",
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
    temporary_config_path = project_root / ".continued_control_run_config.yaml"
    if not main_config_path.exists():
        raise FileNotFoundError(f"Main configuration not found: {main_config_path}")

    cfg = load_config(main_config_path)
    seeds, directions, architectures = _load_experiment_grid(cfg)
    models_dir = _resolve_project_path(project_root, cfg["models_dir"])
    results_dir = _resolve_project_path(project_root, cfg["results_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(seeds) * len(directions) * len(architectures)

    missing_inputs = []
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
                base_path = models_dir / f"base_checkpoint_{tag}.pt"
                ft_path = models_dir / f"finetuned_checkpoint_{tag}.pt"
                for path in (base_path, ft_path):
                    if not path.exists():
                        missing_inputs.append(path)

    if missing_inputs:
        text = "\n".join(f"  - {path}" for path in missing_inputs)
        raise FileNotFoundError(
            "Continued-training control cannot start because required base or "
            f"fine-tuned checkpoints are missing:\n{text}"
        )

    print("\n" + "=" * 80)
    print("CONTINUED-TRAINING CONTROL EXPERIMENT GRID")
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
    print(f"Subset size    : {cfg['ct_subset_size']}")
    print(f"Subset seed    : {cfg['ct_subset_seed']}")
    print(f"Budget source  : {cfg['ct_budget_source']}")
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
                    base_path = models_dir / f"base_checkpoint_{tag}.pt"
                    ft_path = models_dir / f"finetuned_checkpoint_{tag}.pt"
                    control_path = models_dir / f"continued_control_checkpoint_{tag}.pt"
                    history_path = results_dir / f"continued_control_history_{tag}.csv"
                    summary_path = results_dir / f"continued_control_summary_{tag}.json"
                    expected = [control_path, history_path, summary_path]
                    existing = [path for path in expected if path.exists()]

                    print("\n" + "-" * 80)
                    print(f"CONTROL {run_number}/{total_runs}: {tag}")
                    print("-" * 80)

                    if len(existing) == len(expected) and not args.force:
                        _validate_existing_control_artifacts(
                            project_root,
                            cfg,
                            tag,
                            base_path,
                            ft_path,
                            summary_path,
                            src,
                            tgt,
                            enc,
                            dec,
                            seed,
                        )
                        print("[SKIP] Compatible continued-control artifacts already exist.")
                        skipped += 1
                        continue

                    if existing and not args.force:
                        existing_text = "\n".join(f"  - {p}" for p in existing)
                        missing_text = "\n".join(f"  - {p}" for p in expected if not p.exists())
                        raise RuntimeError(
                            f"Partial continued-control artifacts detected for {tag}.\n"
                            f"Existing:\n{existing_text}\nMissing:\n{missing_text}\n"
                            "Inspect the run before continuing; use --force only intentionally."
                        )

                    if args.dry_run:
                        print(f"Base checkpoint       : {base_path}")
                        print(f"Fine-tuned checkpoint : {ft_path}")
                        print("[DRY RUN] Continued-training control would start here.")
                        continue

                    if args.force:
                        for path in expected:
                            if path.exists():
                                path.unlink()
                        print("[FORCE] Previous continued-control artifacts removed.")

                    run_cfg = _build_effective_run_config(cfg, src, tgt, enc, dec, seed)
                    _atomic_yaml_dump(run_cfg, temporary_config_path)
                    print(f"Temporary config      : {temporary_config_path.name}")
                    print("Starting continued-training control...")
                    run_start = time.perf_counter()
                    run_continued_bible_control(
                        config_path=temporary_config_path,
                        base_checkpoint_path=base_path,
                        finetuned_checkpoint_path=ft_path,
                    )
                    run_seconds = time.perf_counter() - run_start

                    missing_after = [path for path in expected if not path.exists()]
                    if missing_after:
                        text = "\n".join(f"  - {path}" for path in missing_after)
                        raise RuntimeError(
                            f"Continued-control run returned without all expected artifacts for "
                            f"{tag}.\nMissing:\n{text}"
                        )

                    completed += 1
                    print(f"[DONE] {tag}")
                    print(f"Run execution time: {run_seconds / 3600.0:.2f} h")

    finally:
        if temporary_config_path.exists():
            temporary_config_path.unlink()
            print(
                f"\n[Cleanup] Temporary configuration removed: "
                f"{temporary_config_path.name}"
            )

    elapsed = time.perf_counter() - overall_start
    print("\n" + "=" * 80)
    print("CONTINUED-TRAINING CONTROL GRID FINISHED")
    print("=" * 80)
    print(f"Total planned runs : {total_runs}")
    print(f"Completed now      : {completed}")
    print(f"Skipped existing   : {skipped}")
    print(f"Elapsed time       : {elapsed / 3600.0:.2f} h")
    print("=" * 80)


if __name__ == "__main__":
    main()
