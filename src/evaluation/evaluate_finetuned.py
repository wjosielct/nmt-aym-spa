# ======================================
# ./src/evaluation/evaluate_finetuned.py
# ======================================

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import sacrebleu
import sentencepiece as spm
import torch

from src.data.dataset import ParallelTranslationDataset
from src.models.transformer import TranslationTransformer
from src.utils.config import load_config
from src.utils.metrics import evaluate_metrics
from src.utils.reproducibility import set_seed


# ==============================================================================
# AUXILIARY FUNCTIONS
# ==============================================================================


def _sync_cuda(
        device: torch.device,
) -> None:
    """Synchronizes CUDA for accurate timing."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hours(
        seconds: float,
) -> float:
    """Converts seconds to hours."""

    return seconds / 3600.0


def _sha256_file(
        file_path: str | Path,
) -> str:
    """Calculates the SHA-256 checksum of a file."""

    file_path = Path(
        file_path
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot hash missing file: {file_path}"
        )

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb",
    ) as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def _get_git_metadata(
        repository_dir: str | Path,
) -> dict:
    """Retrieves Git metadata when available."""

    repository_dir = Path(
        repository_dir
    ).resolve()

    metadata = {
        "commit": None,
        "branch": None,
        "exact_tag": None,
        "dirty_worktree": None,
    }

    try:

        commit_result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_dir),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        metadata["commit"] = (
            commit_result.stdout.strip()
        )

        branch_result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_dir),
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        metadata["branch"] = (
            branch_result.stdout.strip()
        )

        status_result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_dir),
                "status",
                "--porcelain",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        metadata["dirty_worktree"] = bool(
            status_result.stdout.strip()
        )

        tag_result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_dir),
                "describe",
                "--tags",
                "--exact-match",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if tag_result.returncode == 0:

            metadata["exact_tag"] = (
                tag_result.stdout.strip()
            )

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        pass

    return metadata


def _atomic_json_dump(
        payload: dict,
        output_path: str | Path,
) -> None:
    """Atomically writes JSON."""

    output_path = Path(
        output_path
    )

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

            json.dump(
                payload,
                file,
                indent=4,
                ensure_ascii=False,
            )

        temporary_path.replace(
            output_path
        )

    finally:

        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_predictions_tsv_dump(
        references: list[str],
        hypotheses: list[str],
        output_path: str | Path,
) -> None:
    """
    Writes exact scoring references/hypotheses to a TSV for
    future paired bootstrap analysis.
    """

    if len(
        references
    ) != len(
        hypotheses
    ):

        raise ValueError(
            "Reference and hypothesis counts "
            "do not match: "
            f"{len(references)} != "
            f"{len(hypotheses)}"
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df = pd.DataFrame(
        {
            "example_index": range(
                len(references)
            ),
            "reference": references,
            "hypothesis": hypotheses,
        }
    )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    try:

        predictions_df.to_csv(
            temporary_path,
            sep="\t",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

        temporary_path.replace(
            output_path
        )

    finally:

        if temporary_path.exists():
            temporary_path.unlink()


def _expected_model_tag(
        cfg: dict,
) -> str:
    """Builds the experiment identifier shared by all stages."""

    return (
        f"{cfg['src']}_to_{cfg['tgt']}"
        f"_Enc{cfg['encoder_layers']}"
        f"_Dec{cfg['decoder_layers']}"
        f"_Seed{cfg['experiment_seed']}"
    )


def _validate_evaluation_config(
        cfg: dict,
) -> None:
    """Validates final fine-tuned evaluation settings."""

    required_keys = [
        "experiment_seed",
        "deterministic_training",
        "src",
        "tgt",
        "tokenizer_dir",
        "tokenized_dir",
        "models_dir",
        "results_dir",
        "d_model",
        "n_heads",
        "encoder_layers",
        "decoder_layers",
        "dim_feedforward",
        "positional_encoding_max_len",
        "base_dropout",
        "ft_checkpoint_metric",
        "ft_test_max_len",
    ]

    missing = [
        key
        for key in required_keys
        if key not in cfg
    ]

    if missing:

        raise KeyError(
            "Missing required fine-tuned evaluation "
            f"configuration parameter(s): {missing}"
        )

    if int(
        cfg["experiment_seed"]
    ) < 0:

        raise ValueError(
            "experiment_seed must be non-negative."
        )

    if (
        not str(
            cfg["src"]
        ).strip()
        or not str(
            cfg["tgt"]
        ).strip()
    ):

        raise ValueError(
            "src and tgt cannot be empty."
        )

    if str(
        cfg["src"]
    ) == str(
        cfg["tgt"]
    ):

        raise ValueError(
            "src and tgt must be different languages."
        )

    if int(
        cfg["ft_test_max_len"]
    ) <= 0:

        raise ValueError(
            "ft_test_max_len must be greater than zero."
        )

    if int(
        cfg["ft_test_max_len"]
    ) > int(
        cfg["positional_encoding_max_len"]
    ):

        raise ValueError(
            "ft_test_max_len cannot exceed "
            "positional_encoding_max_len."
        )

    if str(
        cfg["ft_checkpoint_metric"]
    ).lower() != "chrf++":

        raise ValueError(
            "This pipeline expects "
            "ft_checkpoint_metric='chrf++'."
        )


def _validate_checkpoint(
        checkpoint: dict,
        cfg: dict,
        expected_model_tag: str,
        tokenizer_vocab_size: int,
        tokenizer_pad_id: int,
        tokenizer_sha256: str,
) -> None:
    """Validates a fine-tuned checkpoint before final testing."""

    required_keys = [
        "checkpoint_type",
        "model_tag",
        "epoch",
        "optimizer_step",
        "steps_per_epoch",
        "planned_total_optimizer_steps",
        "experiment_seed",
        "training_domain",
        "checkpoint_selection_metric",
        "best_val_chrf",
        "val_bleu_at_best_chrf",
        "metric_signatures",
        "base_checkpoint",
        "model_state_dict",
        "vocab_size",
        "pad_id",
        "src",
        "tgt",
        "architecture",
        "fine_tuning_input_hashes",
        "dialogue_split",
        "optimization",
        "git",
        "config",
    ]

    missing = [
        key
        for key in required_keys
        if key not in checkpoint
    ]

    if missing:

        raise KeyError(
            "Fine-tuned checkpoint is missing "
            f"required field(s): {missing}"
        )

    if (
        checkpoint[
            "checkpoint_type"
        ]
        != "fine_tuning"
    ):

        raise ValueError(
            "Expected a fine-tuning checkpoint."
        )

    if (
        checkpoint[
            "model_tag"
        ]
        != expected_model_tag
    ):

        raise ValueError(
            "Fine-tuned checkpoint model_tag "
            "does not match config: "
            f"{checkpoint['model_tag']} != "
            f"{expected_model_tag}"
        )

    if int(
        checkpoint[
            "experiment_seed"
        ]
    ) != int(
        cfg[
            "experiment_seed"
        ]
    ):

        raise ValueError(
            "Fine-tuned checkpoint seed "
            "does not match config."
        )

    if str(
        checkpoint[
            "src"
        ]
    ) != str(
        cfg[
            "src"
        ]
    ):

        raise ValueError(
            "Fine-tuned checkpoint source "
            "language does not match config."
        )

    if str(
        checkpoint[
            "tgt"
        ]
    ) != str(
        cfg[
            "tgt"
        ]
    ):

        raise ValueError(
            "Fine-tuned checkpoint target "
            "language does not match config."
        )

    if str(
        checkpoint[
            "training_domain"
        ]
    ).lower() != "conversational":

        raise ValueError(
            "Fine-tuned checkpoint "
            "training_domain must be Conversational."
        )

    if str(
        checkpoint[
            "checkpoint_selection_metric"
        ]
    ).lower() != str(
        cfg[
            "ft_checkpoint_metric"
        ]
    ).lower():

        raise ValueError(
            "Fine-tuned checkpoint-selection "
            "metric does not match config."
        )

    if int(
        checkpoint[
            "vocab_size"
        ]
    ) != tokenizer_vocab_size:

        raise ValueError(
            "Fine-tuned checkpoint vocabulary "
            "size does not match the "
            "SentencePiece model."
        )

    if int(
        checkpoint[
            "pad_id"
        ]
    ) != tokenizer_pad_id:

        raise ValueError(
            "Fine-tuned checkpoint PAD ID does "
            "not match the SentencePiece model."
        )

    checkpoint_tokenizer_hash = (
        checkpoint
        .get(
            "fine_tuning_input_hashes",
            {},
        )
        .get(
            "sentencepiece_model_sha256"
        )
    )

    if (
        checkpoint_tokenizer_hash
        is not None
        and checkpoint_tokenizer_hash
        != tokenizer_sha256
    ):

        raise ValueError(
            "Current SentencePiece model does not "
            "match the tokenizer used for "
            "fine-tuning."
        )

    expected_architecture = {

        "d_model": int(
            cfg["d_model"]
        ),

        "n_heads": int(
            cfg["n_heads"]
        ),

        "encoder_layers": int(
            cfg[
                "encoder_layers"
            ]
        ),

        "decoder_layers": int(
            cfg[
                "decoder_layers"
            ]
        ),

        "dim_feedforward": int(
            cfg[
                "dim_feedforward"
            ]
        ),

        "dropout": float(
            cfg[
                "base_dropout"
            ]
        ),

        "positional_encoding_max_len": int(
            cfg[
                "positional_encoding_max_len"
            ]
        ),
    }

    checkpoint_architecture = (
        checkpoint[
            "architecture"
        ]
    )

    for (
        key,
        expected_value,
    ) in expected_architecture.items():

        if key not in checkpoint_architecture:

            raise KeyError(
                "Fine-tuned checkpoint "
                "architecture is missing field: "
                f"{key}"
            )

        if (
            checkpoint_architecture[
                key
            ]
            != expected_value
        ):

            raise ValueError(
                "Fine-tuned checkpoint architecture "
                "does not match config for "
                f"'{key}': "
                f"{checkpoint_architecture[key]} != "
                f"{expected_value}"
            )

    training_cfg = (
        checkpoint[
            "config"
        ]
    )

    if (
        "ft_test_max_len"
        in training_cfg
    ):

        if int(
            training_cfg[
                "ft_test_max_len"
            ]
        ) != int(
            cfg[
                "ft_test_max_len"
            ]
        ):

            raise ValueError(
                "ft_test_max_len differs from "
                "the protocol stored with the "
                "fine-tuned checkpoint."
            )


# ==============================================================================
# FINE-TUNED MODEL FINAL TEST EVALUATION
# ==============================================================================


def evaluate_finetuned_model(
        config_path: str | Path = "config.yaml",
        checkpoint_path: str | Path | None = None,
) -> dict:
    """
    Evaluates one selected fine-tuned checkpoint on:

        1. Biblical test set:
           original-domain retention.

        2. Conversational test set:
           target-domain performance.

    Exact references/hypotheses are exported for the
    future paired-bootstrap analysis.
    """

    script_start_time = (
        time.perf_counter()
    )

    config_path = Path(
        config_path
    ).resolve()

    # ==========================================================================
    # CONFIGURATION / REPRODUCIBILITY
    # ==========================================================================

    cfg = load_config(
        config_path
    )

    _validate_evaluation_config(
        cfg
    )

    experiment_seed = int(
        cfg[
            "experiment_seed"
        ]
    )

    deterministic_training = bool(
        cfg[
            "deterministic_training"
        ]
    )

    set_seed(
        seed=experiment_seed,
        deterministic=(
            deterministic_training
        ),
    )

    # ==========================================================================
    # DEVICE / PATHS
    # ==========================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer_dir = Path(
        cfg[
            "tokenizer_dir"
        ]
    )

    tokenized_dir = Path(
        cfg[
            "tokenized_dir"
        ]
    )

    models_dir = Path(
        cfg[
            "models_dir"
        ]
    )

    results_dir = Path(
        cfg[
            "results_dir"
        ]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_tag = (
        _expected_model_tag(
            cfg
        )
    )

    if checkpoint_path is None:

        checkpoint_path = (
            models_dir
            / (
                f"finetuned_checkpoint_"
                f"{model_tag}.pt"
            )
        )

    else:

        checkpoint_path = Path(
            checkpoint_path
        )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Fine-tuned checkpoint not found: "
            f"{checkpoint_path}"
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

    # ==========================================================================
    # TOKENIZER
    # ==========================================================================

    sp_model_path = (
        tokenizer_dir
        / "SentencePiece.model"
    )

    if not sp_model_path.exists():

        raise FileNotFoundError(
            f"SentencePiece model not found: "
            f"{sp_model_path}"
        )

    sp = (
        spm.SentencePieceProcessor(
            model_file=str(
                sp_model_path
            )
        )
    )

    pad_id = (
        sp.pad_id()
    )

    vocab_size = (
        sp.vocab_size()
    )

    if pad_id < 0:

        raise ValueError(
            "SentencePiece PAD token is not defined."
        )

    tokenizer_sha256 = (
        _sha256_file(
            sp_model_path
        )
    )

    # ==========================================================================
    # LOAD / VALIDATE FINE-TUNED CHECKPOINT
    # ==========================================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise TypeError(
            "Fine-tuned checkpoint must "
            "contain a dictionary payload."
        )

    _validate_checkpoint(
        checkpoint=checkpoint,
        cfg=cfg,
        expected_model_tag=(
            model_tag
        ),
        tokenizer_vocab_size=(
            vocab_size
        ),
        tokenizer_pad_id=(
            pad_id
        ),
        tokenizer_sha256=(
            tokenizer_sha256
        ),
    )

    architecture = (
        checkpoint[
            "architecture"
        ]
    )

    # ==========================================================================
    # MODEL RECONSTRUCTION
    # ==========================================================================

    eval_model = (
        TranslationTransformer(

            vocab_size=int(
                checkpoint[
                    "vocab_size"
                ]
            ),

            d_model=int(
                architecture[
                    "d_model"
                ]
            ),

            n_heads=int(
                architecture[
                    "n_heads"
                ]
            ),

            num_encoder_layers=int(
                architecture[
                    "encoder_layers"
                ]
            ),

            num_decoder_layers=int(
                architecture[
                    "decoder_layers"
                ]
            ),

            dim_feedforward=int(
                architecture[
                    "dim_feedforward"
                ]
            ),

            dropout=float(
                architecture[
                    "dropout"
                ]
            ),

            pad_id=int(
                checkpoint[
                    "pad_id"
                ]
            ),

            positional_encoding_max_len=int(
                architecture[
                    "positional_encoding_max_len"
                ]
            ),
        )
        .to(device)
    )

    model_state_dict = (
        checkpoint.pop(
            "model_state_dict"
        )
    )

    eval_model.load_state_dict(
        model_state_dict,
        strict=True,
    )

    del model_state_dict

    eval_model.eval()

    # ==========================================================================
    # TEST DATASETS
    # ==========================================================================

    test_bible_path = (
        tokenized_dir
        / "test_bible_tokenized.tsv"
    )

    test_dialogue_path = (
        tokenized_dir
        / "test_dialogue_tokenized.tsv"
    )

    test_bible_dataset = (
        ParallelTranslationDataset(

            tsv_path=(
                test_bible_path
            ),

            src_col=(
                f"{cfg['src']}_ids"
            ),

            tgt_col=(
                f"{cfg['tgt']}_ids"
            ),
        )
    )

    test_dialogue_dataset = (
        ParallelTranslationDataset(

            tsv_path=(
                test_dialogue_path
            ),

            src_col=(
                f"{cfg['src']}_ids"
            ),

            tgt_col=(
                f"{cfg['tgt']}_ids"
            ),
        )
    )

    # ==========================================================================
    # HASHES / GIT / DECODING PROTOCOL
    # ==========================================================================

    evaluation_hashes = {

        "checkpoint_sha256": (
            _sha256_file(
                checkpoint_path
            )
        ),

        "sentencepiece_model_sha256": (
            tokenizer_sha256
        ),

        "test_bible_tokenized_sha256": (
            _sha256_file(
                test_bible_path
            )
        ),

        "test_dialogue_tokenized_sha256": (
            _sha256_file(
                test_dialogue_path
            )
        ),
    }

    evaluation_git = (
        _get_git_metadata(
            config_path.parent
        )
    )

    test_max_len = int(
        cfg[
            "ft_test_max_len"
        ]
    )

    if test_max_len > int(
        architecture[
            "positional_encoding_max_len"
        ]
    ):

        raise ValueError(
            "ft_test_max_len exceeds the "
            "positional-encoding capacity "
            "stored in the fine-tuned checkpoint."
        )

    # ==========================================================================
    # CONSOLE HEADER
    # ==========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINE-TUNED MODEL FINAL TEST EVALUATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Model tag               : "
        f"{model_tag}"
    )

    print(
        f"Checkpoint              : "
        f"{checkpoint_path}"
    )

    print(
        f"Best FT epoch           : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best FT optimizer step  : "
        f"{checkpoint['optimizer_step']}"
    )

    print(
        f"Direction               : "
        f"{cfg['src']} -> "
        f"{cfg['tgt']}"
    )

    print(
        f"Architecture            : "
        f"{architecture['encoder_layers']}E-"
        f"{architecture['decoder_layers']}D"
    )

    print(
        f"Device                  : "
        f"{device.type.upper()}"
    )

    print(
        f"Decoding                : "
        f"greedy | "
        f"max_len={test_max_len}"
    )

    print(
        f"Biblical test pairs     : "
        f"{len(test_bible_dataset):,}"
    )

    print(
        f"Conversational test pairs: "
        f"{len(test_dialogue_dataset):,}"
    )

    print(
        "=" * 80
    )

    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats(
            device
        )

    # ==========================================================================
    # ORIGINAL-DOMAIN TEST — BIBLICAL
    # ==========================================================================

    print(
        "\n"
        "Evaluating original-domain biblical "
        "test set..."
    )

    _sync_cuda(
        device
    )

    biblical_start = (
        time.perf_counter()
    )

    (
        bib_bleu,
        bib_chrf,
        bib_bleu_signature,
        bib_chrf_signature,
        bib_references,
        bib_hypotheses,
    ) = evaluate_metrics(

        model=eval_model,

        dataset=(
            test_bible_dataset
        ),

        sp=sp,

        device=device,

        max_len=(
            test_max_len
        ),

        return_predictions=True,
    )

    _sync_cuda(
        device
    )

    biblical_seconds = (
        time.perf_counter()
        - biblical_start
    )

    if len(
        bib_references
    ) != len(
        test_bible_dataset
    ):

        raise RuntimeError(
            "Biblical prediction count does "
            "not match test-set size."
        )

    _atomic_predictions_tsv_dump(
        references=(
            bib_references
        ),
        hypotheses=(
            bib_hypotheses
        ),
        output_path=(
            biblical_predictions_path
        ),
    )

    evaluation_hashes[
        "biblical_predictions_sha256"
    ] = _sha256_file(
        biblical_predictions_path
    )

    biblical_results = {

        "domain": (
            "Biblical"
        ),

        "relation_to_fine_tuning_domain": (
            "original-domain-retention"
        ),

        "test_file": (
            str(
                test_bible_path
            )
        ),

        "predictions_file": (
            str(
                biblical_predictions_path
            )
        ),

        "test_pairs": (
            len(
                test_bible_dataset
            )
        ),

        "bleu_score": (
            bib_bleu
        ),

        "chrf_score": (
            bib_chrf
        ),

        "bleu_signature": (
            bib_bleu_signature
        ),

        "chrf_signature": (
            bib_chrf_signature
        ),

        "evaluation_time_seconds": (
            biblical_seconds
        ),

        "evaluation_time_hours": (
            _hours(
                biblical_seconds
            )
        ),
    }

    # ==========================================================================
    # TARGET-DOMAIN TEST — CONVERSATIONAL
    # ==========================================================================

    print(
        "Evaluating target-domain "
        "conversational test set..."
    )

    _sync_cuda(
        device
    )

    conversational_start = (
        time.perf_counter()
    )

    (
        conv_bleu,
        conv_chrf,
        conv_bleu_signature,
        conv_chrf_signature,
        conv_references,
        conv_hypotheses,
    ) = evaluate_metrics(

        model=eval_model,

        dataset=(
            test_dialogue_dataset
        ),

        sp=sp,

        device=device,

        max_len=(
            test_max_len
        ),

        return_predictions=True,
    )

    _sync_cuda(
        device
    )

    conversational_seconds = (
        time.perf_counter()
        - conversational_start
    )

    if len(
        conv_references
    ) != len(
        test_dialogue_dataset
    ):

        raise RuntimeError(
            "Conversational prediction count "
            "does not match test-set size."
        )

    _atomic_predictions_tsv_dump(
        references=(
            conv_references
        ),
        hypotheses=(
            conv_hypotheses
        ),
        output_path=(
            conversational_predictions_path
        ),
    )

    evaluation_hashes[
        "conversational_predictions_sha256"
    ] = _sha256_file(
        conversational_predictions_path
    )

    conversational_results = {

        "domain": (
            "Conversational"
        ),

        "relation_to_fine_tuning_domain": (
            "target-domain"
        ),

        "test_file": (
            str(
                test_dialogue_path
            )
        ),

        "predictions_file": (
            str(
                conversational_predictions_path
            )
        ),

        "test_pairs": (
            len(
                test_dialogue_dataset
            )
        ),

        "bleu_score": (
            conv_bleu
        ),

        "chrf_score": (
            conv_chrf
        ),

        "bleu_signature": (
            conv_bleu_signature
        ),

        "chrf_signature": (
            conv_chrf_signature
        ),

        "evaluation_time_seconds": (
            conversational_seconds
        ),

        "evaluation_time_hours": (
            _hours(
                conversational_seconds
            )
        ),
    }

    # ==========================================================================
    # TIMING / HARDWARE
    # ==========================================================================

    _sync_cuda(
        device
    )

    total_execution_time_seconds = (
        time.perf_counter()
        - script_start_time
    )

    if device.type == "cuda":

        device_name = (
            torch.cuda.get_device_name(
                device
            )
        )

        gpu_total_memory_mb = (
            torch.cuda
            .get_device_properties(
                device
            )
            .total_memory
            / (1024 ** 2)
        )

        peak_gpu_memory_allocated_mb = (
            torch.cuda
            .max_memory_allocated(
                device
            )
            / (1024 ** 2)
        )

        peak_gpu_memory_reserved_mb = (
            torch.cuda
            .max_memory_reserved(
                device
            )
            / (1024 ** 2)
        )

    else:

        device_name = (
            platform.processor()
            or "CPU"
        )

        gpu_total_memory_mb = None
        peak_gpu_memory_allocated_mb = None
        peak_gpu_memory_reserved_mb = None

    # ==========================================================================
    # CONSOLIDATED PAYLOAD
    # ==========================================================================

    payload = {

        "model_tag": (
            model_tag
        ),

        "experiment_type": (
            "Fine-Tuned-Test-Evaluation"
        ),

        "base_training_domain": (
            "Biblical"
        ),

        "fine_tuning_domain": (
            "Conversational"
        ),

        "src": (
            cfg["src"]
        ),

        "tgt": (
            cfg["tgt"]
        ),

        "experiment_seed": (
            experiment_seed
        ),

        "checkpoint_selection_metric": (
            checkpoint[
                "checkpoint_selection_metric"
            ]
        ),

        "best_checkpoint_epoch": (
            int(
                checkpoint[
                    "epoch"
                ]
            )
        ),

        "best_checkpoint_optimizer_step": (
            int(
                checkpoint[
                    "optimizer_step"
                ]
            )
        ),

        "checkpoint": {

            "path": (
                str(
                    checkpoint_path
                )
            ),

            "sha256": (
                evaluation_hashes[
                    "checkpoint_sha256"
                ]
            ),

            "best_val_chrf": (
                checkpoint[
                    "best_val_chrf"
                ]
            ),

            "val_bleu_at_best_chrf": (
                checkpoint[
                    "val_bleu_at_best_chrf"
                ]
            ),

            "validation_metric_signatures": (
                checkpoint[
                    "metric_signatures"
                ]
            ),

            "fine_tuning_input_hashes": (
                checkpoint[
                    "fine_tuning_input_hashes"
                ]
            ),

            "fine_tuning_git": (
                checkpoint[
                    "git"
                ]
            ),

            "base_checkpoint": (
                checkpoint[
                    "base_checkpoint"
                ]
            ),

            "optimization": (
                checkpoint[
                    "optimization"
                ]
            ),

            "dialogue_split": (
                checkpoint[
                    "dialogue_split"
                ]
            ),
        },

        "model": {

            "vocab_size": (
                int(
                    checkpoint[
                        "vocab_size"
                    ]
                )
            ),

            "pad_id": (
                int(
                    checkpoint[
                        "pad_id"
                    ]
                )
            ),

            "architecture": (
                architecture
            ),
        },

        "evaluation": {

            "decoding_strategy": (
                "greedy"
            ),

            "max_len": (
                test_max_len
            ),

            "prediction_export": {

                "format": (
                    "tsv"
                ),

                "columns": [
                    "example_index",
                    "reference",
                    "hypothesis",
                ],
            },
        },

        "biblical_results": (
            biblical_results
        ),

        "conversational_results": (
            conversational_results
        ),

        "evaluation_hashes": (
            evaluation_hashes
        ),

        "timing": {

            "biblical_test_seconds": (
                biblical_seconds
            ),

            "conversational_test_seconds": (
                conversational_seconds
            ),

            "total_execution_time_seconds": (
                total_execution_time_seconds
            ),

            "total_execution_time_hours": (
                _hours(
                    total_execution_time_seconds
                )
            ),
        },

        "hardware": {

            "device_type": (
                device.type
            ),

            "device_name": (
                device_name
            ),

            "gpu_total_memory_mb": (
                gpu_total_memory_mb
            ),

            "peak_gpu_memory_allocated_mb": (
                peak_gpu_memory_allocated_mb
            ),

            "peak_gpu_memory_reserved_mb": (
                peak_gpu_memory_reserved_mb
            ),
        },

        "reproducibility": {

            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),

            "cudnn_deterministic": (
                torch.backends.cudnn.deterministic
            ),

            "cudnn_benchmark": (
                torch.backends.cudnn.benchmark
            ),
        },

        "software": {

            "python_version": (
                sys.version.split()[0]
            ),

            "pytorch_version": (
                str(
                    torch.__version__
                )
            ),

            "cuda_version": (
                torch.version.cuda
            ),

            "cudnn_version": (
                torch.backends.cudnn.version()
            ),

            "sentencepiece_version": (
                getattr(
                    spm,
                    "__version__",
                    None,
                )
            ),

            "sacrebleu_version": (
                sacrebleu.__version__
            ),

            "pandas_version": (
                pd.__version__
            ),

            "platform": (
                platform.platform()
            ),
        },

        "git": (
            evaluation_git
        ),

        "fine_tuning_config": (
            checkpoint[
                "config"
            ]
        ),

        "evaluation_config": (
            cfg
        ),
    }

    output_path = (
        results_dir
        / (
            f"finetuned_test_metrics_"
            f"{model_tag}.json"
        )
    )

    _atomic_json_dump(
        payload=payload,
        output_path=output_path,
    )

    # ==========================================================================
    # FINAL CONSOLE SUMMARY
    # ==========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"FINAL FINE-TUNED TEST RESULTS "
        f"[{model_tag}]"
    )

    print(
        "=" * 80
    )

    print(
        f"ORIGINAL DOMAIN — Biblical "
        f"[{len(test_bible_dataset):,} pairs]"
    )

    print(
        f"  BLEU   : "
        f"{bib_bleu:.2f}"
    )

    print(
        f"  chrF++ : "
        f"{bib_chrf:.2f}"
    )

    print(
        f"TARGET DOMAIN — Conversational "
        f"[{len(test_dialogue_dataset):,} pairs]"
    )

    print(
        f"  BLEU   : "
        f"{conv_bleu:.2f}"
    )

    print(
        f"  chrF++ : "
        f"{conv_chrf:.2f}"
    )

    print(
        f"Total execution time    : "
        f"{_hours(total_execution_time_seconds):.2f} h"
    )

    print(
        f"Results                 : "
        f"{output_path}"
    )

    print(
        f"Biblical predictions    : "
        f"{biblical_predictions_path}"
    )

    print(
        f"Conversational predictions: "
        f"{conversational_predictions_path}"
    )

    print(
        "=" * 80
    )

    return payload


if __name__ == "__main__":

    evaluate_finetuned_model(
        config_path="config.yaml"
    )
