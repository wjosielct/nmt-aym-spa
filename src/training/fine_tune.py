# ==============================
# ./src/training/fine_tune.py
# ==============================

import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import pandas as pd
import sacrebleu
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.dataset import (
    ParallelTranslationDataset,
    collate_fn_pad,
)
from src.models.transformer import TranslationTransformer
from src.utils.config import load_config
from src.utils.metrics import evaluate_metrics
from src.utils.reproducibility import (
    set_seed,
    seed_worker,
)


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.nn.modules.transformer",
)


# ==============================================================================
# AUXILIARY FUNCTIONS
# ==============================================================================


def _sync_cuda(
        device: torch.device,
) -> None:
    """Synchronizes CUDA before wall-clock timing measurements."""

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

    file_path = Path(file_path)

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
    """Retrieves Git commit/tag/branch information when available."""

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


def _atomic_torch_save(
        payload: dict,
        output_path: str | Path,
) -> None:
    """Atomically writes a PyTorch checkpoint."""

    output_path = Path(output_path)

    temporary_path = output_path.with_suffix(
        output_path.suffix
        + ".tmp"
    )

    try:

        torch.save(
            payload,
            temporary_path,
        )

        temporary_path.replace(
            output_path
        )

    finally:

        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_json_dump(
        payload: dict,
        output_path: str | Path,
) -> None:
    """Atomically writes JSON."""

    output_path = Path(output_path)

    temporary_path = output_path.with_suffix(
        output_path.suffix
        + ".tmp"
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


def _atomic_csv_dump(
        rows: list[dict],
        output_path: str | Path,
) -> None:
    """Atomically writes the per-epoch history CSV."""

    output_path = Path(output_path)

    temporary_path = output_path.with_suffix(
        output_path.suffix
        + ".tmp"
    )

    try:

        pd.DataFrame(
            rows
        ).to_csv(
            temporary_path,
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


def _validate_finetuning_config(
        cfg: dict,
) -> None:
    """Validates the complete fine-tuning protocol."""

    required_keys = [

        # ----------------------------------------------------------------------
        # Experiment identity
        # ----------------------------------------------------------------------

        "experiment_seed",
        "deterministic_training",
        "src",
        "tgt",

        # ----------------------------------------------------------------------
        # Paths
        # ----------------------------------------------------------------------

        "tokenizer_dir",
        "tokenized_dir",
        "models_dir",
        "results_dir",

        # ----------------------------------------------------------------------
        # Architecture expected from the base checkpoint
        # ----------------------------------------------------------------------

        "d_model",
        "n_heads",
        "encoder_layers",
        "decoder_layers",
        "dim_feedforward",
        "positional_encoding_max_len",
        "base_dropout",
        "base_checkpoint_metric",

        # ----------------------------------------------------------------------
        # Fine-tuning duration / batching
        # ----------------------------------------------------------------------

        "ft_epochs",
        "ft_batch_size_cuda",
        "ft_batch_size_cpu",
        "ft_shuffle_train",
        "ft_drop_last",

        # ----------------------------------------------------------------------
        # Optimizer
        # ----------------------------------------------------------------------

        "ft_optimizer",
        "ft_learning_rate",
        "ft_adam_beta1",
        "ft_adam_beta2",
        "ft_adam_eps",
        "ft_weight_decay",

        # ----------------------------------------------------------------------
        # Loss / gradients
        # ----------------------------------------------------------------------

        "ft_label_smoothing",
        "ft_grad_clip_norm",

        # ----------------------------------------------------------------------
        # Scheduler
        # ----------------------------------------------------------------------

        "ft_scheduler",
        "ft_scheduler_eta_min",
        "ft_scheduler_step_unit",

        # ----------------------------------------------------------------------
        # AMP / DataLoader
        # ----------------------------------------------------------------------

        "ft_amp_enabled",
        "ft_num_workers",
        "ft_pin_memory",

        # ----------------------------------------------------------------------
        # Validation / checkpointing
        # ----------------------------------------------------------------------

        "ft_val_metric_interval_epochs",
        "ft_val_max_len",
        "ft_checkpoint_metric",
    ]

    missing = [
        key
        for key in required_keys
        if key not in cfg
    ]

    if missing:
        raise KeyError(
            "Missing required fine-tuning configuration "
            f"parameter(s): {missing}"
        )

    if int(
        cfg["experiment_seed"]
    ) < 0:
        raise ValueError(
            "experiment_seed must be non-negative."
        )

    if not str(
        cfg["src"]
    ).strip():
        raise ValueError(
            "src cannot be empty."
        )

    if not str(
        cfg["tgt"]
    ).strip():
        raise ValueError(
            "tgt cannot be empty."
        )

    if str(
        cfg["src"]
    ) == str(
        cfg["tgt"]
    ):
        raise ValueError(
            "src and tgt must be different languages."
        )

    positive_integer_keys = [
        "d_model",
        "n_heads",
        "encoder_layers",
        "decoder_layers",
        "dim_feedforward",
        "positional_encoding_max_len",
        "ft_epochs",
        "ft_batch_size_cuda",
        "ft_batch_size_cpu",
        "ft_val_metric_interval_epochs",
        "ft_val_max_len",
    ]

    for key in positive_integer_keys:

        if int(
            cfg[key]
        ) <= 0:
            raise ValueError(
                f"{key} must be greater than zero."
            )

    if int(
        cfg["ft_num_workers"]
    ) < 0:
        raise ValueError(
            "ft_num_workers cannot be negative."
        )

    if int(
        cfg["ft_val_max_len"]
    ) > int(
        cfg["positional_encoding_max_len"]
    ):
        raise ValueError(
            "ft_val_max_len cannot exceed "
            "positional_encoding_max_len."
        )

    if not (
        0.0
        <= float(
            cfg["base_dropout"]
        )
        < 1.0
    ):
        raise ValueError(
            "base_dropout must be in [0, 1)."
        )

    if not (
        0.0
        <= float(
            cfg["ft_label_smoothing"]
        )
        < 1.0
    ):
        raise ValueError(
            "ft_label_smoothing must be in [0, 1)."
        )

    if float(
        cfg["ft_learning_rate"]
    ) <= 0:
        raise ValueError(
            "ft_learning_rate must be greater than zero."
        )

    if float(
        cfg["ft_scheduler_eta_min"]
    ) < 0:
        raise ValueError(
            "ft_scheduler_eta_min cannot be negative."
        )

    if float(
        cfg["ft_scheduler_eta_min"]
    ) > float(
        cfg["ft_learning_rate"]
    ):
        raise ValueError(
            "ft_scheduler_eta_min cannot exceed "
            "ft_learning_rate."
        )

    if float(
        cfg["ft_adam_eps"]
    ) <= 0:
        raise ValueError(
            "ft_adam_eps must be greater than zero."
        )

    if float(
        cfg["ft_weight_decay"]
    ) < 0:
        raise ValueError(
            "ft_weight_decay cannot be negative."
        )

    if float(
        cfg["ft_grad_clip_norm"]
    ) <= 0:
        raise ValueError(
            "ft_grad_clip_norm must be greater than zero."
        )

    beta1 = float(
        cfg["ft_adam_beta1"]
    )

    beta2 = float(
        cfg["ft_adam_beta2"]
    )

    if not (
        0.0
        <= beta1
        < 1.0
    ):
        raise ValueError(
            "ft_adam_beta1 must be in [0, 1)."
        )

    if not (
        0.0
        <= beta2
        < 1.0
    ):
        raise ValueError(
            "ft_adam_beta2 must be in [0, 1)."
        )

    if str(
        cfg["base_checkpoint_metric"]
    ).lower() != "chrf++":
        raise ValueError(
            "The pipeline expects "
            "base_checkpoint_metric='chrf++'."
        )

    if str(
        cfg["ft_checkpoint_metric"]
    ).lower() != "chrf++":
        raise ValueError(
            "The fine-tuning pipeline selects checkpoints "
            "using ft_checkpoint_metric='chrf++'."
        )

    if str(
        cfg["ft_optimizer"]
    ).lower() != "adamw":
        raise ValueError(
            "This implementation currently supports "
            "ft_optimizer='adamw' only."
        )

    if str(
        cfg["ft_scheduler"]
    ).lower() != "cosine":
        raise ValueError(
            "This implementation currently supports "
            "ft_scheduler='cosine' only."
        )

    if str(
        cfg["ft_scheduler_step_unit"]
    ).lower() != "optimizer_step":
        raise ValueError(
            "ft_scheduler_step_unit must be "
            "'optimizer_step'. This is explicit so the "
            "continued-training control can match "
            "the optimization budget exactly."
        )


def _validate_base_checkpoint(
        checkpoint: dict,
        cfg: dict,
        expected_model_tag: str,
        tokenizer_vocab_size: int,
        tokenizer_pad_id: int,
        tokenizer_sha256: str,
) -> None:
    """
    Validates that the selected base checkpoint is the exact model
    requested for fine-tuning.
    """

    required_keys = [
        "checkpoint_type",
        "model_tag",
        "epoch",
        "experiment_seed",
        "checkpoint_selection_metric",
        "best_val_chrf",
        "val_bleu_at_best_chrf",
        "metric_signatures",
        "model_state_dict",
        "vocab_size",
        "pad_id",
        "src",
        "tgt",
        "architecture",
        "input_hashes",
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
            "Base checkpoint is missing required field(s): "
            f"{missing}"
        )

    if (
        checkpoint["checkpoint_type"]
        != "base_training"
    ):
        raise ValueError(
            "Fine-tuning must start from a "
            "base-training checkpoint."
        )

    if (
        checkpoint["model_tag"]
        != expected_model_tag
    ):
        raise ValueError(
            "Base checkpoint model_tag does not match "
            "the requested run: "
            f"{checkpoint['model_tag']} != "
            f"{expected_model_tag}"
        )

    if int(
        checkpoint["experiment_seed"]
    ) != int(
        cfg["experiment_seed"]
    ):
        raise ValueError(
            "Base checkpoint experiment seed does "
            "not match config."
        )

    if str(
        checkpoint["src"]
    ) != str(
        cfg["src"]
    ):
        raise ValueError(
            "Base checkpoint source language does "
            "not match config."
        )

    if str(
        checkpoint["tgt"]
    ) != str(
        cfg["tgt"]
    ):
        raise ValueError(
            "Base checkpoint target language does "
            "not match config."
        )

    if str(
        checkpoint[
            "checkpoint_selection_metric"
        ]
    ).lower() != str(
        cfg["base_checkpoint_metric"]
    ).lower():
        raise ValueError(
            "Base checkpoint-selection metric does "
            "not match config."
        )

    if int(
        checkpoint["vocab_size"]
    ) != tokenizer_vocab_size:
        raise ValueError(
            "Base checkpoint vocabulary size does "
            "not match the current SentencePiece model."
        )

    if int(
        checkpoint["pad_id"]
    ) != tokenizer_pad_id:
        raise ValueError(
            "Base checkpoint PAD ID does not match "
            "the current SentencePiece model."
        )

    checkpoint_tokenizer_hash = (
        checkpoint
        .get(
            "input_hashes",
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
            "match the tokenizer used to train the "
            "base checkpoint."
        )

    expected_architecture = {
        "d_model": int(
            cfg["d_model"]
        ),
        "n_heads": int(
            cfg["n_heads"]
        ),
        "encoder_layers": int(
            cfg["encoder_layers"]
        ),
        "decoder_layers": int(
            cfg["decoder_layers"]
        ),
        "dim_feedforward": int(
            cfg["dim_feedforward"]
        ),
        "dropout": float(
            cfg["base_dropout"]
        ),
        "positional_encoding_max_len": int(
            cfg["positional_encoding_max_len"]
        ),
    }

    checkpoint_architecture = (
        checkpoint["architecture"]
    )

    for (
        key,
        expected_value,
    ) in expected_architecture.items():

        if key not in checkpoint_architecture:
            raise KeyError(
                "Base checkpoint architecture is "
                f"missing field: {key}"
            )

        if (
            checkpoint_architecture[key]
            != expected_value
        ):
            raise ValueError(
                "Base checkpoint architecture does "
                f"not match config for '{key}': "
                f"{checkpoint_architecture[key]} != "
                f"{expected_value}"
            )


def _dialogue_split_metadata(
        train_dataset: ParallelTranslationDataset,
        val_dataset: ParallelTranslationDataset,
) -> dict:
    """
    Verifies dialogue-level separation between conversational
    train and validation partitions.
    """

    dialogue_column = "dialogue_id"

    if (
        dialogue_column
        not in train_dataset.df.columns
    ):
        raise ValueError(
            "Conversational training data must "
            "contain 'dialogue_id'."
        )

    if (
        dialogue_column
        not in val_dataset.df.columns
    ):
        raise ValueError(
            "Conversational validation data must "
            "contain 'dialogue_id'."
        )

    train_dialogues = set(
        train_dataset.df[
            dialogue_column
        ].astype(str)
    )

    val_dialogues = set(
        val_dataset.df[
            dialogue_column
        ].astype(str)
    )

    overlap = (
        train_dialogues
        .intersection(
            val_dialogues
        )
    )

    if overlap:
        raise ValueError(
            "Dialogue-level leakage detected between "
            "conversational train and validation "
            "partitions: "
            f"{sorted(overlap)}"
        )

    return {
        "group_column": dialogue_column,
        "train_dialogues": len(
            train_dialogues
        ),
        "validation_dialogues": len(
            val_dialogues
        ),
        "train_validation_overlap": 0,
    }


# ==============================================================================
# FINE-TUNING
# ==============================================================================


def run_fine_tuning(
        config_path: str | Path = "config.yaml",
        base_checkpoint_path: str | Path | None = None,
) -> None:
    """
    Fine-tunes one selected biblical base checkpoint on the
    conversational training partition.

    The adapted checkpoint is selected by conversational-validation
    chrF++. Test sets are intentionally not accessed here.
    """

    script_start_time = (
        time.perf_counter()
    )

    config_path = Path(
        config_path
    ).resolve()

    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================

    cfg = load_config(
        config_path
    )

    _validate_finetuning_config(
        cfg
    )

    experiment_seed = int(
        cfg["experiment_seed"]
    )

    deterministic_training = bool(
        cfg["deterministic_training"]
    )

    set_seed(
        seed=experiment_seed,
        deterministic=deterministic_training,
    )

    # ==========================================================================
    # DEVICE / AMP
    # ==========================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_requested = bool(
        cfg["ft_amp_enabled"]
    )

    use_amp = (
        amp_requested
        and device.type == "cuda"
    )

    print(
        f"Device: {device.type.upper()} | "
        f"AMP: {use_amp} | "
        f"Deterministic: "
        f"{deterministic_training} | "
        f"Seed: {experiment_seed}"
    )

    # ==========================================================================
    # PATHS
    # ==========================================================================

    tokenizer_dir = Path(
        cfg["tokenizer_dir"]
    )

    tokenized_dir = Path(
        cfg["tokenized_dir"]
    )

    models_dir = Path(
        cfg["models_dir"]
    )

    results_dir = Path(
        cfg["results_dir"]
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    if base_checkpoint_path is None:

        base_checkpoint_path = (
            models_dir
            / (
                f"base_checkpoint_"
                f"{model_tag}.pt"
            )
        )

    else:

        base_checkpoint_path = Path(
            base_checkpoint_path
        )

    if not base_checkpoint_path.exists():

        raise FileNotFoundError(
            "Base checkpoint required for fine-tuning "
            "was not found: "
            f"{base_checkpoint_path}"
        )

    best_model_path = (
        models_dir
        / (
            f"finetuned_checkpoint_"
            f"{model_tag}.pt"
        )
    )

    history_path = (
        results_dir
        / (
            f"finetuning_history_"
            f"{model_tag}.csv"
        )
    )

    summary_path = (
        results_dir
        / (
            f"finetuning_summary_"
            f"{model_tag}.json"
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
    # LOAD / VALIDATE BASE CHECKPOINT
    # ==========================================================================

    base_checkpoint = torch.load(
        base_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(
        base_checkpoint,
        dict,
    ):

        raise TypeError(
            "Base checkpoint must contain "
            "a dictionary payload."
        )

    _validate_base_checkpoint(
        checkpoint=base_checkpoint,
        cfg=cfg,
        expected_model_tag=model_tag,
        tokenizer_vocab_size=vocab_size,
        tokenizer_pad_id=pad_id,
        tokenizer_sha256=tokenizer_sha256,
    )

    architecture = dict(
        base_checkpoint[
            "architecture"
        ]
    )

    base_checkpoint_sha256 = (
        _sha256_file(
            base_checkpoint_path
        )
    )

    # ==========================================================================
    # CONVERSATIONAL TRAIN / VALIDATION DATASETS
    # ==========================================================================

    train_path = (
        tokenized_dir
        / "train_dialogue_tokenized.tsv"
    )

    val_path = (
        tokenized_dir
        / "val_dialogue_tokenized.tsv"
    )

    train_dataset = (
        ParallelTranslationDataset(
            tsv_path=train_path,
            src_col=(
                f"{cfg['src']}_ids"
            ),
            tgt_col=(
                f"{cfg['tgt']}_ids"
            ),
        )
    )

    val_dataset = (
        ParallelTranslationDataset(
            tsv_path=val_path,
            src_col=(
                f"{cfg['src']}_ids"
            ),
            tgt_col=(
                f"{cfg['tgt']}_ids"
            ),
        )
    )

    dialogue_split = (
        _dialogue_split_metadata(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
    )

    # ==========================================================================
    # INPUT HASHES / GIT
    # ==========================================================================

    input_hashes = {

        "config_sha256": (
            _sha256_file(
                config_path
            )
        ),

        "base_checkpoint_sha256": (
            base_checkpoint_sha256
        ),

        "sentencepiece_model_sha256": (
            tokenizer_sha256
        ),

        "train_dialogue_tokenized_sha256": (
            _sha256_file(
                train_path
            )
        ),

        "val_dialogue_tokenized_sha256": (
            _sha256_file(
                val_path
            )
        ),
    }

    git_metadata = (
        _get_git_metadata(
            config_path.parent
        )
    )

    # ==========================================================================
    # DATALOADERS
    # ==========================================================================

    batch_size = (
        int(
            cfg["ft_batch_size_cuda"]
        )
        if device.type == "cuda"
        else int(
            cfg["ft_batch_size_cpu"]
        )
    )

    num_workers = int(
        cfg["ft_num_workers"]
    )

    pin_memory = (
        bool(
            cfg["ft_pin_memory"]
        )
        and device.type == "cuda"
    )

    shuffle_train = bool(
        cfg["ft_shuffle_train"]
    )

    drop_last = bool(
        cfg["ft_drop_last"]
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        experiment_seed
    )

    collate_with_pad = partial(
        collate_fn_pad,
        pad_id=pad_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=drop_last,
        generator=train_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        collate_fn=collate_with_pad,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        collate_fn=collate_with_pad,
    )

    if len(
        train_loader
    ) == 0:

        raise ValueError(
            "Fine-tuning DataLoader contains "
            "zero optimizer batches."
        )

    # ==========================================================================
    # RECONSTRUCT BASE MODEL AND LOAD SELECTED WEIGHTS
    # ==========================================================================

    model = (
        TranslationTransformer(

            vocab_size=int(
                base_checkpoint[
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
                base_checkpoint[
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

    model.load_state_dict(
        base_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if (
        trainable_parameters
        != total_parameters
    ):

        raise RuntimeError(
            "Fine-tuning is configured as "
            "full-model adaptation, but some "
            "parameters are frozen."
        )

    # ==========================================================================
    # LOSS
    # ==========================================================================

    criterion = (
        nn.CrossEntropyLoss(

            ignore_index=pad_id,

            label_smoothing=float(
                cfg[
                    "ft_label_smoothing"
                ]
            ),
        )
    )

    # ==========================================================================
    # OPTIMIZER — NEW STATE FOR THE ADAPTATION STAGE
    # ==========================================================================

    optimizer_name = str(
        cfg["ft_optimizer"]
    ).lower()

    optimizer = (
        torch.optim.AdamW(

            model.parameters(),

            lr=float(
                cfg[
                    "ft_learning_rate"
                ]
            ),

            betas=(
                float(
                    cfg[
                        "ft_adam_beta1"
                    ]
                ),
                float(
                    cfg[
                        "ft_adam_beta2"
                    ]
                ),
            ),

            eps=float(
                cfg[
                    "ft_adam_eps"
                ]
            ),

            weight_decay=float(
                cfg[
                    "ft_weight_decay"
                ]
            ),
        )
    )

    # ==========================================================================
    # OPTIMIZATION BUDGET / LEARNING-RATE SCHEDULER
    # ==========================================================================

    epochs = int(
        cfg["ft_epochs"]
    )

    steps_per_epoch = len(
        train_loader
    )

    planned_total_optimizer_steps = (
        steps_per_epoch
        * epochs
    )

    scheduler_name = str(
        cfg["ft_scheduler"]
    ).lower()

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(

            optimizer,

            T_max=(
                planned_total_optimizer_steps
            ),

            eta_min=float(
                cfg[
                    "ft_scheduler_eta_min"
                ]
            ),
        )
    )

    initial_scheduled_lr = float(
        optimizer.param_groups[0][
            "lr"
        ]
    )

    # ==========================================================================
    # MIXED PRECISION
    # ==========================================================================

    scaler = GradScaler(
        enabled=use_amp
    )

    # The model now owns its loaded parameters. Remove the duplicate
    # state dictionary from the in-memory base-checkpoint payload.
    base_model_state_dict = (
        base_checkpoint.pop(
            "model_state_dict"
        )
    )

    del base_model_state_dict

    # ==========================================================================
    # TRAINING STATE
    # ==========================================================================

    best_val_chrf = float(
        "-inf"
    )

    best_val_bleu_at_best_chrf = None
    best_bleu_signature = None
    best_chrf_signature = None

    best_epoch = None
    best_optimizer_step = None

    minimum_val_loss = float(
        "inf"
    )

    validation_metric_evaluations = 0
    optimizer_steps_completed = 0

    total_train_phase_seconds = 0.0
    total_val_loss_seconds = 0.0
    total_val_metrics_seconds = 0.0

    history = []

    setup_time_seconds = (
        time.perf_counter()
        - script_start_time
    )

    # ==========================================================================
    # CONSOLE HEADER
    # ==========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"CONVERSATIONAL FINE-TUNING "
        f"[{cfg['src'].upper()} -> "
        f"{cfg['tgt'].upper()} | "
        f"Enc:{architecture['encoder_layers']} "
        f"Dec:{architecture['decoder_layers']}]"
    )

    print(
        f"Epochs: {epochs} | "
        f"Batch size: {batch_size} | "
        f"Shuffle: {shuffle_train} | "
        f"Drop last: {drop_last}"
    )

    print(
        f"Train pairs: "
        f"{len(train_dataset):,} | "
        f"Validation pairs: "
        f"{len(val_dataset):,}"
    )

    print(
        f"Dialogue groups: "
        f"train="
        f"{dialogue_split['train_dialogues']} | "
        f"val="
        f"{dialogue_split['validation_dialogues']} | "
        f"overlap="
        f"{dialogue_split['train_validation_overlap']}"
    )

    print(
        f"Base checkpoint epoch: "
        f"{base_checkpoint['epoch']} | "
        f"Base val chrF++: "
        f"{base_checkpoint['best_val_chrf']:.2f}"
    )

    print(
        f"Parameters: "
        f"{total_parameters:,} "
        f"(trainable: "
        f"{trainable_parameters:,})"
    )

    print(
        f"Optimizer: {optimizer_name} | "
        f"Scheduler: {scheduler_name} "
        f"per optimizer step"
    )

    print(
        f"Initial LR: "
        f"{initial_scheduled_lr:.2e} | "
        f"Cosine eta_min: "
        f"{float(cfg['ft_scheduler_eta_min']):.2e}"
    )

    print(
        f"Steps/epoch: "
        f"{steps_per_epoch} | "
        f"Planned optimizer steps: "
        f"{planned_total_optimizer_steps}"
    )

    print(
        f"Checkpoint criterion: "
        f"{cfg['ft_checkpoint_metric']} | "
        f"Validation interval: every "
        f"{cfg['ft_val_metric_interval_epochs']} "
        f"epoch(s)"
    )

    print(
        f"Validation decoding: greedy | "
        f"max_len="
        f"{cfg['ft_val_max_len']}"
    )

    print(
        "=" * 80
    )

    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats(
            device
        )

    training_loop_start = (
        time.perf_counter()
    )

    # ==========================================================================
    # FINE-TUNING LOOP
    # ==========================================================================

    for epoch in range(
        1,
        epochs + 1,
    ):

        epoch_start_time = (
            time.perf_counter()
        )

        # ======================================================================
        # TRAINING
        # ======================================================================

        model.train()

        train_loss_sum = 0.0

        _sync_cuda(
            device
        )

        train_phase_start = (
            time.perf_counter()
        )

        for src, tgt in (
            train_loader
        ):

            src = src.to(
                device,
                non_blocking=pin_memory,
            )

            tgt = tgt.to(
                device,
                non_blocking=pin_memory,
            )

            tgt_in = (
                tgt[:, :-1]
            )

            tgt_out = (
                tgt[:, 1:]
                .contiguous()
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast(
                device_type=device.type,
                enabled=use_amp,
                dtype=torch.float16,
            ):

                logits = model(
                    src,
                    tgt_in,
                )

                loss = criterion(

                    logits.reshape(
                        -1,
                        vocab_size,
                    ),

                    tgt_out.reshape(
                        -1
                    ),
                )

            if use_amp:

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    float(
                        cfg[
                            "ft_grad_clip_norm"
                        ]
                    ),
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    float(
                        cfg[
                            "ft_grad_clip_norm"
                        ]
                    ),
                )

                optimizer.step()

            scheduler.step()

            optimizer_steps_completed += 1

            train_loss_sum += (
                loss.item()
            )

        _sync_cuda(
            device
        )

        train_phase_seconds = (
            time.perf_counter()
            - train_phase_start
        )

        total_train_phase_seconds += (
            train_phase_seconds
        )

        train_loss = (
            train_loss_sum
            / len(train_loader)
        )

        train_pairs_per_second = (
            len(train_dataset)
            / train_phase_seconds
        )

        # ======================================================================
        # VALIDATION LOSS
        # ======================================================================

        model.eval()

        val_loss_sum = 0.0

        _sync_cuda(
            device
        )

        val_loss_start = (
            time.perf_counter()
        )

        with torch.no_grad():

            for src, tgt in (
                val_loader
            ):

                src = src.to(
                    device,
                    non_blocking=pin_memory,
                )

                tgt = tgt.to(
                    device,
                    non_blocking=pin_memory,
                )

                tgt_in = (
                    tgt[:, :-1]
                )

                tgt_out = (
                    tgt[:, 1:]
                    .contiguous()
                )

                with autocast(
                    device_type=device.type,
                    enabled=use_amp,
                    dtype=torch.float16,
                ):

                    logits = model(
                        src,
                        tgt_in,
                    )

                    loss = criterion(

                        logits.reshape(
                            -1,
                            vocab_size,
                        ),

                        tgt_out.reshape(
                            -1
                        ),
                    )

                val_loss_sum += (
                    loss.item()
                )

        _sync_cuda(
            device
        )

        val_loss_seconds = (
            time.perf_counter()
            - val_loss_start
        )

        total_val_loss_seconds += (
            val_loss_seconds
        )

        val_loss = (
            val_loss_sum
            / len(val_loader)
        )

        minimum_val_loss = min(
            minimum_val_loss,
            val_loss,
        )

        val_loss_pairs_per_second = (
            len(val_dataset)
            / val_loss_seconds
        )

        # ======================================================================
        # PERIODIC GREEDY VALIDATION METRICS
        # ======================================================================

        val_bleu = None
        val_chrf = None

        val_bleu_signature = None
        val_chrf_signature = None

        val_metrics_seconds = None
        val_metrics_pairs_per_second = None

        metric_interval = int(
            cfg[
                "ft_val_metric_interval_epochs"
            ]
        )

        should_evaluate_metrics = (
            epoch % metric_interval == 0
            or epoch == epochs
        )

        checkpoint_saved = False

        if should_evaluate_metrics:

            validation_metric_evaluations += 1

            _sync_cuda(
                device
            )

            val_metrics_start = (
                time.perf_counter()
            )

            (
                val_bleu,
                val_chrf,
                val_bleu_signature,
                val_chrf_signature,
            ) = evaluate_metrics(

                model=model,

                dataset=(
                    val_dataset
                ),

                sp=sp,

                device=device,

                max_len=int(
                    cfg[
                        "ft_val_max_len"
                    ]
                ),
            )

            _sync_cuda(
                device
            )

            val_metrics_seconds = (
                time.perf_counter()
                - val_metrics_start
            )

            total_val_metrics_seconds += (
                val_metrics_seconds
            )

            val_metrics_pairs_per_second = (
                len(val_dataset)
                / val_metrics_seconds
            )

            # ==================================================================
            # BEST FINE-TUNED CHECKPOINT — VALIDATION chrF++
            # ==================================================================

            if (
                val_chrf
                > best_val_chrf
            ):

                best_val_chrf = (
                    val_chrf
                )

                best_val_bleu_at_best_chrf = (
                    val_bleu
                )

                best_bleu_signature = (
                    val_bleu_signature
                )

                best_chrf_signature = (
                    val_chrf_signature
                )

                best_epoch = (
                    epoch
                )

                best_optimizer_step = (
                    optimizer_steps_completed
                )

                checkpoint = {

                    "checkpoint_type": (
                        "fine_tuning"
                    ),

                    "model_tag": (
                        model_tag
                    ),

                    "epoch": (
                        epoch
                    ),

                    "optimizer_step": (
                        optimizer_steps_completed
                    ),

                    "steps_per_epoch": (
                        steps_per_epoch
                    ),

                    "planned_total_optimizer_steps": (
                        planned_total_optimizer_steps
                    ),

                    "experiment_seed": (
                        experiment_seed
                    ),

                    "training_domain": (
                        "Conversational"
                    ),

                    "checkpoint_selection_metric": (
                        "chrF++"
                    ),

                    "best_val_chrf": (
                        best_val_chrf
                    ),

                    "val_bleu_at_best_chrf": (
                        best_val_bleu_at_best_chrf
                    ),

                    "metric_signatures": {

                        "bleu": (
                            best_bleu_signature
                        ),

                        "chrf++": (
                            best_chrf_signature
                        ),
                    },

                    "base_checkpoint": {

                        "path": (
                            str(
                                base_checkpoint_path
                            )
                        ),

                        "sha256": (
                            base_checkpoint_sha256
                        ),

                        "epoch": int(
                            base_checkpoint[
                                "epoch"
                            ]
                        ),

                        "best_val_chrf": (
                            base_checkpoint[
                                "best_val_chrf"
                            ]
                        ),

                        "val_bleu_at_best_chrf": (
                            base_checkpoint[
                                "val_bleu_at_best_chrf"
                            ]
                        ),

                        "input_hashes": (
                            base_checkpoint[
                                "input_hashes"
                            ]
                        ),

                        "git": (
                            base_checkpoint[
                                "git"
                            ]
                        ),
                    },

                    "model_state_dict": (
                        model.state_dict()
                    ),

                    "vocab_size": (
                        vocab_size
                    ),

                    "pad_id": (
                        pad_id
                    ),

                    "src": (
                        cfg["src"]
                    ),

                    "tgt": (
                        cfg["tgt"]
                    ),

                    "architecture": (
                        architecture
                    ),

                    "fine_tuning_input_hashes": (
                        input_hashes
                    ),

                    "dialogue_split": (
                        dialogue_split
                    ),

                    "optimization": {

                        "full_model_fine_tuning": (
                            True
                        ),

                        "batch_size": (
                            batch_size
                        ),

                        "optimizer": (
                            optimizer_name
                        ),

                        "learning_rate": float(
                            cfg[
                                "ft_learning_rate"
                            ]
                        ),

                        "scheduler": (
                            scheduler_name
                        ),

                        "scheduler_step_unit": str(
                            cfg[
                                "ft_scheduler_step_unit"
                            ]
                        ),

                        "scheduler_eta_min": float(
                            cfg[
                                "ft_scheduler_eta_min"
                            ]
                        ),

                        "steps_per_epoch": (
                            steps_per_epoch
                        ),

                        "planned_total_optimizer_steps": (
                            planned_total_optimizer_steps
                        ),

                        "optimizer_step_at_checkpoint": (
                            optimizer_steps_completed
                        ),
                    },

                    "git": (
                        git_metadata
                    ),

                    "config": (
                        cfg
                    ),
                }

                _atomic_torch_save(
                    checkpoint,
                    best_model_path,
                )

                checkpoint_saved = True

        # ======================================================================
        # EPOCH HISTORY
        # ======================================================================

        _sync_cuda(
            device
        )

        epoch_time_seconds = (
            time.perf_counter()
            - epoch_start_time
        )

        current_lr = float(
            optimizer.param_groups[0][
                "lr"
            ]
        )

        row = {

            "epoch": (
                epoch
            ),

            "optimizer_step_end": (
                optimizer_steps_completed
            ),

            "train_loss": (
                train_loss
            ),

            "val_loss": (
                val_loss
            ),

            "val_bleu": (
                val_bleu
            ),

            "val_chrf": (
                val_chrf
            ),

            "learning_rate_end": (
                current_lr
            ),

            "checkpoint_saved": (
                checkpoint_saved
            ),

            "train_time_seconds": (
                train_phase_seconds
            ),

            "validation_loss_time_seconds": (
                val_loss_seconds
            ),

            "validation_metrics_time_seconds": (
                val_metrics_seconds
            ),

            "epoch_time_seconds": (
                epoch_time_seconds
            ),

            "train_pairs_per_second": (
                train_pairs_per_second
            ),

            "validation_loss_pairs_per_second": (
                val_loss_pairs_per_second
            ),

            "validation_metrics_pairs_per_second": (
                val_metrics_pairs_per_second
            ),
        }

        history.append(
            row
        )

        _atomic_csv_dump(
            rows=history,
            output_path=history_path,
        )

        bleu_text = (
            f"{val_bleu:.2f}"
            if val_bleu is not None
            else "-"
        )

        chrf_text = (
            f"{val_chrf:.2f}"
            if val_chrf is not None
            else "-"
        )

        metrics_time_text = (
            f"{val_metrics_seconds:.1f}s"
            if val_metrics_seconds is not None
            else "-"
        )

        print(
            f"Epoch {epoch:03d}/"
            f"{epochs:03d} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Val Loss: "
            f"{val_loss:.4f} | "
            f"Val BLEU: "
            f"{bleu_text} | "
            f"Val chrF++: "
            f"{chrf_text} | "
            f"LR: "
            f"{current_lr:.2e} | "
            f"Step: "
            f"{optimizer_steps_completed}/"
            f"{planned_total_optimizer_steps} | "
            f"Train: "
            f"{train_phase_seconds:.1f}s | "
            f"ValLoss: "
            f"{val_loss_seconds:.1f}s | "
            f"ValMetrics: "
            f"{metrics_time_text} | "
            f"Epoch: "
            f"{epoch_time_seconds:.1f}s"
        )

    # ==========================================================================
    # FINAL CONSISTENCY
    # ==========================================================================

    _sync_cuda(
        device
    )

    training_loop_time_seconds = (
        time.perf_counter()
        - training_loop_start
    )

    if (
        optimizer_steps_completed
        != planned_total_optimizer_steps
    ):

        raise RuntimeError(
            "Fine-tuning completed with an "
            "unexpected optimizer-step count: "
            f"{optimizer_steps_completed} != "
            f"{planned_total_optimizer_steps}"
        )

    if best_epoch is None:

        raise RuntimeError(
            "Fine-tuning finished without "
            "selecting a validation checkpoint."
        )

    if not best_model_path.exists():

        raise RuntimeError(
            "Fine-tuning finished but the selected "
            "checkpoint is missing: "
            f"{best_model_path}"
        )

    total_execution_time_seconds = (
        time.perf_counter()
        - script_start_time
    )

    # ==========================================================================
    # HARDWARE INFORMATION
    # ==========================================================================

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
    # SUMMARY JSON
    # ==========================================================================

    summary = {

        "model_tag": (
            model_tag
        ),

        "experiment_type": (
            "Conversational-Fine-Tuning"
        ),

        "base_training_domain": (
            "Biblical"
        ),

        "fine_tuning_domain": (
            "Conversational"
        ),

        "experiment_seed": (
            experiment_seed
        ),

        "deterministic_training": (
            deterministic_training
        ),

        "checkpoint_selection_metric": (
            "chrF++"
        ),

        "decoding_strategy_validation": (
            "greedy"
        ),

        "validation_max_len": int(
            cfg[
                "ft_val_max_len"
            ]
        ),

        "validation_metric_interval_epochs": int(
            cfg[
                "ft_val_metric_interval_epochs"
            ]
        ),

        "validation_metric_evaluations": (
            validation_metric_evaluations
        ),

        "best_checkpoint_epoch": (
            best_epoch
        ),

        "best_checkpoint_optimizer_step": (
            best_optimizer_step
        ),

        "best_val_chrf": (
            best_val_chrf
        ),

        "val_bleu_at_best_chrf": (
            best_val_bleu_at_best_chrf
        ),

        "minimum_val_loss": (
            minimum_val_loss
        ),

        "metric_signatures": {

            "bleu": (
                best_bleu_signature
            ),

            "chrf++": (
                best_chrf_signature
            ),
        },

        "base_checkpoint": {

            "path": (
                str(
                    base_checkpoint_path
                )
            ),

            "sha256": (
                base_checkpoint_sha256
            ),

            "epoch": int(
                base_checkpoint[
                    "epoch"
                ]
            ),

            "best_val_chrf": (
                base_checkpoint[
                    "best_val_chrf"
                ]
            ),

            "val_bleu_at_best_chrf": (
                base_checkpoint[
                    "val_bleu_at_best_chrf"
                ]
            ),

            "input_hashes": (
                base_checkpoint[
                    "input_hashes"
                ]
            ),

            "git": (
                base_checkpoint[
                    "git"
                ]
            ),
        },

        "checkpoint": (
            str(
                best_model_path
            )
        ),

        "history_file": (
            str(
                history_path
            )
        ),

        "dataset": {

            "train_file": (
                str(
                    train_path
                )
            ),

            "validation_file": (
                str(
                    val_path
                )
            ),

            "train_pairs": (
                len(
                    train_dataset
                )
            ),

            "validation_pairs": (
                len(
                    val_dataset
                )
            ),

            "dialogue_split": (
                dialogue_split
            ),
        },

        "input_hashes": (
            input_hashes
        ),

        "git": (
            git_metadata
        ),

        "model": {

            "total_parameters": (
                total_parameters
            ),

            "trainable_parameters": (
                trainable_parameters
            ),

            "full_model_fine_tuning": (
                True
            ),

            "architecture": (
                architecture
            ),
        },

        "optimizer_scheduler": {

            "optimizer": (
                optimizer_name
            ),

            "learning_rate": float(
                cfg[
                    "ft_learning_rate"
                ]
            ),

            "adam_beta1": float(
                cfg[
                    "ft_adam_beta1"
                ]
            ),

            "adam_beta2": float(
                cfg[
                    "ft_adam_beta2"
                ]
            ),

            "adam_eps": float(
                cfg[
                    "ft_adam_eps"
                ]
            ),

            "weight_decay": float(
                cfg[
                    "ft_weight_decay"
                ]
            ),

            "scheduler": (
                scheduler_name
            ),

            "scheduler_step_unit": str(
                cfg[
                    "ft_scheduler_step_unit"
                ]
            ),

            "initial_learning_rate": (
                initial_scheduled_lr
            ),

            "scheduler_eta_min": float(
                cfg[
                    "ft_scheduler_eta_min"
                ]
            ),
        },

        "optimization_budget": {

            "batch_size": (
                batch_size
            ),

            "epochs": (
                epochs
            ),

            "steps_per_epoch": (
                steps_per_epoch
            ),

            "planned_total_optimizer_steps": (
                planned_total_optimizer_steps
            ),

            "completed_optimizer_steps": (
                optimizer_steps_completed
            ),

            "optimizer_step_at_best_checkpoint": (
                best_optimizer_step
            ),
        },

        "timing": {

            "setup_time_seconds": (
                setup_time_seconds
            ),

            "training_loop_time_seconds": (
                training_loop_time_seconds
            ),

            "training_loop_time_hours": (
                _hours(
                    training_loop_time_seconds
                )
            ),

            "optimization_time_seconds": (
                total_train_phase_seconds
            ),

            "optimization_time_hours": (
                _hours(
                    total_train_phase_seconds
                )
            ),

            "validation_loss_total_seconds": (
                total_val_loss_seconds
            ),

            "validation_metrics_total_seconds": (
                total_val_metrics_seconds
            ),

            "validation_metrics_total_hours": (
                _hours(
                    total_val_metrics_seconds
                )
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

            "mixed_precision_amp": (
                use_amp
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

        "config": (
            cfg
        ),
    }

    _atomic_json_dump(
        payload=summary,
        output_path=summary_path,
    )

    # ==========================================================================
    # FINAL CONSOLE SUMMARY
    # ==========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CONVERSATIONAL FINE-TUNING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Best epoch              : "
        f"{best_epoch}"
    )

    print(
        f"Best optimizer step     : "
        f"{best_optimizer_step}"
    )

    print(
        f"Best val chrF++         : "
        f"{best_val_chrf:.2f}"
    )

    print(
        f"Val BLEU @ best chrF++  : "
        f"{best_val_bleu_at_best_chrf:.2f}"
    )

    print(
        f"Minimum val loss        : "
        f"{minimum_val_loss:.4f}"
    )

    print(
        f"Checkpoint              : "
        f"{best_model_path}"
    )

    print(
        f"Optimization time       : "
        f"{_hours(total_train_phase_seconds):.2f} h"
    )

    print(
        f"Validation metric time  : "
        f"{_hours(total_val_metrics_seconds):.2f} h"
    )

    print(
        f"Total execution time    : "
        f"{_hours(total_execution_time_seconds):.2f} h"
    )

    print(
        f"History                 : "
        f"{history_path}"
    )

    print(
        f"Summary                 : "
        f"{summary_path}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    run_fine_tuning(
        config_path="config.yaml"
    )
