# =====================================
# ./14_run_finetuned_evaluation.py
# =====================================

import argparse
import hashlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import yaml

from src.evaluation.evaluate_finetuned import (
    evaluate_finetuned_model,
)
from src.utils.config import load_config


# ==============================================================================
# AUXILIARY FUNCTIONS
# ==============================================================================


def _model_tag(
        src: str,
        tgt: str,
        encoder_layers: int,
        decoder_layers: int,
        seed: int,
) -> str:
    """Builds the experiment identifier shared by all stages."""

    return (
        f"{src}_to_{tgt}"
        f"_Enc{encoder_layers}"
        f"_Dec{decoder_layers}"
        f"_Seed{seed}"
    )


def _resolve_project_path(
        project_root: Path,
        path_value: str | Path,
) -> Path:
    """Resolves a config path relative to the project root."""

    path = Path(
        path_value
    )

    if path.is_absolute():
        return path

    return (
        project_root
        / path
    )


def _atomic_yaml_dump(
        payload: dict,
        output_path: Path,
) -> None:
    """Atomically writes a temporary YAML configuration."""

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    try:

        with open(
            temporary_path,
            "w",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                payload,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        temporary_path.replace(
            output_path
        )

    finally:

        if temporary_path.exists():
            temporary_path.unlink()


def _load_experiment_grid(
        cfg: dict,
) -> tuple[
    list[int],
    list[dict],
    list[dict],
]:
    """Reads and validates the final experimental grid."""

    required_grid_keys = [
        "experiment_seeds",
        "experiment_directions",
        "experiment_architectures",
    ]

    missing = [
        key
        for key in required_grid_keys
        if key not in cfg
    ]

    if missing:

        raise KeyError(
            "Missing experiment-grid configuration "
            f"parameter(s): {missing}"
        )

    raw_seeds = (
        cfg[
            "experiment_seeds"
        ]
    )

    if (
        not isinstance(
            raw_seeds,
            list,
        )
        or not raw_seeds
    ):

        raise ValueError(
            "experiment_seeds must be a non-empty list."
        )

    seeds = [
        int(seed)
        for seed in raw_seeds
    ]

    if any(
        seed < 0
        for seed in seeds
    ):

        raise ValueError(
            "All experiment seeds must be non-negative."
        )

    if len(
        seeds
    ) != len(
        set(seeds)
    ):

        raise ValueError(
            "experiment_seeds contains duplicate values."
        )

    raw_directions = (
        cfg[
            "experiment_directions"
        ]
    )

    if (
        not isinstance(
            raw_directions,
            list,
        )
        or not raw_directions
    ):

        raise ValueError(
            "experiment_directions must be a non-empty list."
        )

    directions = []

    for item in raw_directions:

        if not isinstance(
            item,
            dict,
        ):

            raise TypeError(
                "Each experiment direction must be a mapping."
            )

        if (
            "src" not in item
            or "tgt" not in item
        ):

            raise KeyError(
                "Each experiment direction must define "
                "'src' and 'tgt'."
            )

        src = str(
            item[
                "src"
            ]
        ).strip()

        tgt = str(
            item[
                "tgt"
            ]
        ).strip()

        if (
            not src
            or not tgt
            or src == tgt
        ):

            raise ValueError(
                "Each experiment direction must contain "
                "two different non-empty languages."
            )

        directions.append(
            {
                "src": src,
                "tgt": tgt,
            }
        )

    direction_pairs = [
        (
            item[
                "src"
            ],
            item[
                "tgt"
            ],
        )
        for item in directions
    ]

    if len(
        direction_pairs
    ) != len(
        set(
            direction_pairs
        )
    ):

        raise ValueError(
            "experiment_directions contains duplicates."
        )

    raw_architectures = (
        cfg[
            "experiment_architectures"
        ]
    )

    if (
        not isinstance(
            raw_architectures,
            list,
        )
        or not raw_architectures
    ):

        raise ValueError(
            "experiment_architectures must be "
            "a non-empty list."
        )

    architectures = []

    for item in raw_architectures:

        if not isinstance(
            item,
            dict,
        ):

            raise TypeError(
                "Each experiment architecture "
                "must be a mapping."
            )

        if (
            "encoder_layers" not in item
            or "decoder_layers" not in item
        ):

            raise KeyError(
                "Each architecture must define "
                "encoder_layers and decoder_layers."
            )

        encoder_layers = int(
            item[
                "encoder_layers"
            ]
        )

        decoder_layers = int(
            item[
                "decoder_layers"
            ]
        )

        if (
            encoder_layers <= 0
            or decoder_layers <= 0
        ):

            raise ValueError(
                "Encoder/decoder layer counts "
                "must be positive."
            )

        if (
            encoder_layers
            + decoder_layers
            != 8
        ):

            raise ValueError(
                "Every architecture must preserve "
                "the fixed eight-layer budget."
            )

        architectures.append(
            {
                "encoder_layers": (
                    encoder_layers
                ),
                "decoder_layers": (
                    decoder_layers
                ),
            }
        )

    architecture_pairs = [
        (
            item[
                "encoder_layers"
            ],
            item[
                "decoder_layers"
            ],
        )
        for item in architectures
    ]

    if len(
        architecture_pairs
    ) != len(
        set(
            architecture_pairs
        )
    ):

        raise ValueError(
            "experiment_architectures contains duplicates."
        )

    return (
        seeds,
        directions,
        architectures,
    )



_FINE_TUNING_COMPATIBILITY_KEYS = [
    'deterministic_training',
    'vocab_size',
    'd_model',
    'n_heads',
    'encoder_layers',
    'decoder_layers',
    'dim_feedforward',
    'positional_encoding_max_len',
    'base_dropout',
    'ft_epochs',
    'ft_batch_size_cuda',
    'ft_batch_size_cpu',
    'ft_shuffle_train',
    'ft_drop_last',
    'ft_optimizer',
    'ft_learning_rate',
    'ft_adam_beta1',
    'ft_adam_beta2',
    'ft_adam_eps',
    'ft_weight_decay',
    'ft_label_smoothing',
    'ft_grad_clip_norm',
    'ft_scheduler',
    'ft_scheduler_eta_min',
    'ft_scheduler_step_unit',
    'ft_amp_enabled',
    'ft_num_workers',
    'ft_pin_memory',
    'ft_val_metric_interval_epochs',
    'ft_val_max_len',
    'ft_checkpoint_metric',
]

def _load_json_file(path: Path) -> dict:
    """Loads a JSON artifact and fails clearly if it cannot be audited."""

    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read JSON artifact required for compatibility audit: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"JSON artifact must contain an object at the top level: {path}"
        )

    return payload


def _sha256_file(path: Path) -> str:
    """Computes SHA-256 without loading the whole file into memory."""

    if not path.exists():
        raise FileNotFoundError(
            f"File required for compatibility audit does not exist: {path}"
        )

    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_effective_run_config(
        cfg: dict,
        src: str,
        tgt: str,
        encoder_layers: int,
        decoder_layers: int,
        seed: int,
) -> dict:
    """Builds the scalar config that the low-level module receives for one run."""

    run_cfg = deepcopy(cfg)
    run_cfg["experiment_seed"] = int(seed)
    run_cfg["src"] = src
    run_cfg["tgt"] = tgt
    run_cfg["encoder_layers"] = int(encoder_layers)
    run_cfg["decoder_layers"] = int(decoder_layers)
    return run_cfg


def _compare_config_fields(
        stored_cfg: dict,
        expected_cfg: dict,
        keys: list[str],
        mismatches: list[str],
        prefix: str,
) -> None:
    """Appends human-readable config mismatches to ``mismatches``."""

    if not isinstance(stored_cfg, dict):
        mismatches.append(f"{prefix}: stored configuration is missing or invalid")
        return

    for key in keys:
        if key not in expected_cfg:
            mismatches.append(f"{prefix}.{key}: key missing from current config")
            continue
        if key not in stored_cfg:
            mismatches.append(f"{prefix}.{key}: key missing from stored artifact")
            continue

        stored_value = stored_cfg[key]
        expected_value = expected_cfg[key]
        if stored_value != expected_value:
            mismatches.append(
                f"{prefix}.{key}: stored={stored_value!r}, current={expected_value!r}"
            )


def _compare_value(
        stored_value,
        expected_value,
        label: str,
        mismatches: list[str],
) -> None:
    """Appends a scalar mismatch when two values are not equal."""

    if stored_value != expected_value:
        mismatches.append(
            f"{label}: stored={stored_value!r}, current={expected_value!r}"
        )


def _raise_incompatible_artifacts(
        stage: str,
        model_tag: str,
        mismatches: list[str],
) -> None:
    """Stops rather than silently reusing artifacts from another protocol."""

    if not mismatches:
        return

    mismatch_text = "\n".join(f"  - {item}" for item in mismatches)
    raise RuntimeError(
        f"Complete {stage} artifacts exist for {model_tag}, but they are not "
        "compatible with the current run configuration and inputs.\n"
        f"Mismatches:\n{mismatch_text}\n"
        "Do not delete or overwrite them automatically. Inspect the run first; "
        "use --force only when replacement is intentional."
    )

def _validate_existing_finetuned_evaluation_artifacts(
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
    """Verifies that stored fine-tuned evaluation outputs belong to the current run."""

    metrics = _load_json_file(metrics_path)
    expected_cfg = _build_effective_run_config(
        cfg, src, tgt, encoder_layers, decoder_layers, seed
    )
    mismatches: list[str] = []

    _compare_value(metrics.get("model_tag"), model_tag, "model_tag", mismatches)
    _compare_value(metrics.get("src"), src, "src", mismatches)
    _compare_value(metrics.get("tgt"), tgt, "tgt", mismatches)
    _compare_value(
        metrics.get("experiment_seed"), int(seed), "experiment_seed", mismatches
    )

    _compare_config_fields(
        metrics.get("fine_tuning_config", {}),
        expected_cfg,
        ["experiment_seed", "src", "tgt"] + _FINE_TUNING_COMPATIBILITY_KEYS,
        mismatches,
        "fine_tuning_config",
    )
    _compare_config_fields(
        metrics.get("evaluation_config", {}),
        expected_cfg,
        ["experiment_seed", "src", "tgt", "encoder_layers", "decoder_layers", "ft_test_max_len"],
        mismatches,
        "evaluation_config",
    )

    model_arch = metrics.get("model", {}).get("architecture", {})
    _compare_value(
        model_arch.get("encoder_layers"), int(encoder_layers),
        "model.architecture.encoder_layers", mismatches
    )
    _compare_value(
        model_arch.get("decoder_layers"), int(decoder_layers),
        "model.architecture.decoder_layers", mismatches
    )
    _compare_value(
        metrics.get("evaluation", {}).get("max_len"),
        int(expected_cfg["ft_test_max_len"]),
        "evaluation.max_len",
        mismatches,
    )

    tokenized_dir = _resolve_project_path(
        project_root, expected_cfg["tokenized_dir"]
    )
    tokenizer_path = _resolve_project_path(
        project_root, expected_cfg["tokenizer_dir"]
    ) / "SentencePiece.model"
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
        _compare_value(
            stored_hashes.get(key), current_hash, f"evaluation_hashes.{key}", mismatches
        )

    checkpoint_meta_hash = metrics.get("checkpoint", {}).get("sha256")
    _compare_value(
        checkpoint_meta_hash,
        current_hashes["checkpoint_sha256"],
        "checkpoint.sha256",
        mismatches,
    )

    _raise_incompatible_artifacts(
        "fine-tuned evaluation", model_tag, mismatches
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    """Builds command-line options for safe evaluation orchestration."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every selected conversational "
            "fine-tuned checkpoint."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Regenerate fine-tuned evaluation "
            "outputs that already exist."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the evaluation plan without "
            "running inference."
        ),
    )

    return parser


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:

    args = (
        _build_argument_parser()
        .parse_args()
    )

    project_root = Path(
        __file__
    ).resolve().parent

    os.chdir(
        project_root
    )

    main_config_path = (
        project_root
        / "config.yaml"
    )

    temporary_config_path = (
        project_root
        / ".finetuned_evaluation_config.yaml"
    )

    if not main_config_path.exists():

        raise FileNotFoundError(
            f"Main configuration not found: "
            f"{main_config_path}"
        )

    cfg = load_config(
        main_config_path
    )

    (
        seeds,
        directions,
        architectures,
    ) = _load_experiment_grid(
        cfg
    )

    models_dir = (
        _resolve_project_path(
            project_root=project_root,
            path_value=cfg[
                "models_dir"
            ],
        )
    )

    results_dir = (
        _resolve_project_path(
            project_root=project_root,
            path_value=cfg[
                "results_dir"
            ],
        )
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_runs = (
        len(
            seeds
        )
        * len(
            directions
        )
        * len(
            architectures
        )
    )

    # ==========================================================================
    # PREFLIGHT — ALL FINE-TUNED CHECKPOINTS MUST EXIST
    # ==========================================================================

    missing_checkpoints = []

    for architecture in architectures:

        encoder_layers = int(
            architecture[
                "encoder_layers"
            ]
        )

        decoder_layers = int(
            architecture[
                "decoder_layers"
            ]
        )

        for direction in directions:

            src = str(
                direction[
                    "src"
                ]
            )

            tgt = str(
                direction[
                    "tgt"
                ]
            )

            for seed in seeds:

                model_tag = (
                    _model_tag(
                        src=src,
                        tgt=tgt,
                        encoder_layers=(
                            encoder_layers
                        ),
                        decoder_layers=(
                            decoder_layers
                        ),
                        seed=seed,
                    )
                )

                checkpoint_path = (
                    models_dir
                    / (
                        f"finetuned_checkpoint_"
                        f"{model_tag}.pt"
                    )
                )

                if not checkpoint_path.exists():

                    missing_checkpoints.append(
                        checkpoint_path
                    )

    if missing_checkpoints:

        missing_text = (
            "\n".join(
                (
                    f"  - {path}"
                )
                for path
                in missing_checkpoints
            )
        )

        raise FileNotFoundError(
            "Fine-tuned evaluation cannot start "
            "because some checkpoints are missing:\n"
            f"{missing_text}"
        )

    # ==========================================================================
    # PLAN
    # ==========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL FINE-TUNED EVALUATION EXPERIMENT GRID"
    )

    print(
        "=" * 80
    )

    print(
        f"Seeds          : "
        f"{seeds}"
    )

    print(
        "Directions     : "
        + ", ".join(
            (
                f"{item['src']}"
                f"->{item['tgt']}"
            )
            for item in directions
        )
    )

    print(
        "Architectures  : "
        + ", ".join(
            (
                f"{item['encoder_layers']}E-"
                f"{item['decoder_layers']}D"
            )
            for item in architectures
        )
    )

    print(
        f"Total runs     : "
        f"{total_runs}"
    )

    print(
        f"Models dir     : "
        f"{cfg['models_dir']}"
    )

    print(
        f"Results dir    : "
        f"{cfg['results_dir']}"
    )

    print(
        f"Force          : "
        f"{args.force}"
    )

    print(
        f"Dry run        : "
        f"{args.dry_run}"
    )

    print(
        "=" * 80
    )

    completed_runs = 0
    skipped_runs = 0
    run_number = 0

    overall_start = (
        time.perf_counter()
    )

    try:

        for architecture in architectures:

            encoder_layers = int(
                architecture[
                    "encoder_layers"
                ]
            )

            decoder_layers = int(
                architecture[
                    "decoder_layers"
                ]
            )

            for direction in directions:

                src = str(
                    direction[
                        "src"
                    ]
                )

                tgt = str(
                    direction[
                        "tgt"
                    ]
                )

                for seed in seeds:

                    run_number += 1

                    model_tag = (
                        _model_tag(
                            src=src,
                            tgt=tgt,
                            encoder_layers=(
                                encoder_layers
                            ),
                            decoder_layers=(
                                decoder_layers
                            ),
                            seed=seed,
                        )
                    )

                    checkpoint_path = (
                        models_dir
                        / (
                            f"finetuned_checkpoint_"
                            f"{model_tag}.pt"
                        )
                    )

                    metrics_path = (
                        results_dir
                        / (
                            f"finetuned_test_metrics_"
                            f"{model_tag}.json"
                        )
                    )

                    biblical_predictions_path = (
                        results_dir
                        / (
                            f"finetuned_predictions_biblical_"
                            f"{model_tag}.tsv"
                        )
                    )

                    conversational_predictions_path = (
                        results_dir
                        / (
                            f"finetuned_predictions_conversational_"
                            f"{model_tag}.tsv"
                        )
                    )

                    expected_artifacts = [
                        metrics_path,
                        biblical_predictions_path,
                        conversational_predictions_path,
                    ]

                    existing_artifacts = [
                        path
                        for path
                        in expected_artifacts
                        if path.exists()
                    ]

                    print(
                        "\n"
                        + "-" * 80
                    )

                    print(
                        f"EVALUATION "
                        f"{run_number}/"
                        f"{total_runs}: "
                        f"{model_tag}"
                    )

                    print(
                        "-" * 80
                    )

                    # ==========================================================
                    # SAFE RESTART
                    # ==========================================================

                    if (
                        len(
                            existing_artifacts
                        )
                        == len(
                            expected_artifacts
                        )
                        and not args.force
                    ):

                        _validate_existing_finetuned_evaluation_artifacts(
                            project_root=project_root,
                            cfg=cfg,
                            model_tag=model_tag,
                            checkpoint_path=checkpoint_path,
                            metrics_path=metrics_path,
                            biblical_predictions_path=biblical_predictions_path,
                            conversational_predictions_path=conversational_predictions_path,
                            src=src,
                            tgt=tgt,
                            encoder_layers=encoder_layers,
                            decoder_layers=decoder_layers,
                            seed=seed,
                        )

                        print(
                            "[SKIP] Compatible fine-tuned "
                            "evaluation artifacts already exist."
                        )

                        skipped_runs += 1

                        continue

                    if (
                        existing_artifacts
                        and not args.force
                    ):

                        existing_text = (
                            "\n".join(
                                (
                                    f"  - {path}"
                                )
                                for path
                                in existing_artifacts
                            )
                        )

                        missing_text = (
                            "\n".join(
                                (
                                    f"  - {path}"
                                )
                                for path
                                in expected_artifacts
                                if not path.exists()
                            )
                        )

                        raise RuntimeError(
                            "Partial fine-tuned evaluation "
                            f"artifacts detected for {model_tag}.\n"
                            "Existing:\n"
                            f"{existing_text}\n"
                            "Missing:\n"
                            f"{missing_text}\n"
                            "Inspect the run before continuing. "
                            "Use --force only for intentional "
                            "regeneration."
                        )

                    if args.dry_run:

                        print(
                            f"Checkpoint: "
                            f"{checkpoint_path}"
                        )

                        print(
                            "[DRY RUN] Evaluation "
                            "would start here."
                        )

                        continue

                    if args.force:

                        for path in expected_artifacts:

                            if path.exists():
                                path.unlink()

                        print(
                            "[FORCE] Previous evaluation "
                            "artifacts removed."
                        )

                    # ==========================================================
                    # SINGLE-RUN CONFIGURATION
                    # ==========================================================

                    run_cfg = deepcopy(
                        cfg
                    )

                    run_cfg[
                        "experiment_seed"
                    ] = int(
                        seed
                    )

                    run_cfg[
                        "src"
                    ] = src

                    run_cfg[
                        "tgt"
                    ] = tgt

                    run_cfg[
                        "encoder_layers"
                    ] = encoder_layers

                    run_cfg[
                        "decoder_layers"
                    ] = decoder_layers

                    _atomic_yaml_dump(
                        payload=run_cfg,
                        output_path=(
                            temporary_config_path
                        ),
                    )

                    print(
                        f"Checkpoint: "
                        f"{checkpoint_path}"
                    )

                    print(
                        f"Temporary config: "
                        f"{temporary_config_path.name}"
                    )

                    print(
                        "Starting fine-tuned evaluation..."
                    )

                    run_start = (
                        time.perf_counter()
                    )

                    # ==========================================================
                    # FINE-TUNED EVALUATION ONLY
                    # ==========================================================

                    evaluate_finetuned_model(
                        config_path=(
                            temporary_config_path
                        ),
                        checkpoint_path=(
                            checkpoint_path
                        ),
                    )

                    run_seconds = (
                        time.perf_counter()
                        - run_start
                    )

                    # ==========================================================
                    # VERIFY ARTIFACTS
                    # ==========================================================

                    missing_after_run = [
                        path
                        for path
                        in expected_artifacts
                        if not path.exists()
                    ]

                    if missing_after_run:

                        missing_text = (
                            "\n".join(
                                (
                                    f"  - {path}"
                                )
                                for path
                                in missing_after_run
                            )
                        )

                        raise RuntimeError(
                            "Fine-tuned evaluation returned "
                            "without all expected artifacts "
                            f"for {model_tag}.\n"
                            "Missing:\n"
                            f"{missing_text}"
                        )

                    _validate_existing_finetuned_evaluation_artifacts(
                        project_root=project_root,
                        cfg=cfg,
                        model_tag=model_tag,
                        checkpoint_path=checkpoint_path,
                        metrics_path=metrics_path,
                        biblical_predictions_path=biblical_predictions_path,
                        conversational_predictions_path=conversational_predictions_path,
                        src=src,
                        tgt=tgt,
                        encoder_layers=encoder_layers,
                        decoder_layers=decoder_layers,
                        seed=seed,
                    )

                    completed_runs += 1

                    print(
                        f"[DONE] "
                        f"{model_tag}"
                    )

                    print(
                        f"Evaluation execution time: "
                        f"{run_seconds / 3600.0:.2f} h"
                    )

    finally:

        if temporary_config_path.exists():

            temporary_config_path.unlink()

            print(
                "\n"
                "[Cleanup] Temporary configuration removed: "
                f"{temporary_config_path.name}"
            )

    overall_seconds = (
        time.perf_counter()
        - overall_start
    )

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINE-TUNED EVALUATION GRID FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"Total planned runs : "
        f"{total_runs}"
    )

    print(
        f"Completed now      : "
        f"{completed_runs}"
    )

    print(
        f"Skipped existing   : "
        f"{skipped_runs}"
    )

    print(
        f"Elapsed time       : "
        f"{overall_seconds / 3600.0:.2f} h"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()
