# ==============================
# ./src/training/train_base.py
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
    """
    Synchronizes CUDA before wall-clock timing measurements.

    CUDA operations are asynchronous, so synchronization is required
    for accurate elapsed-time measurements.
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hours(
        seconds: float,
) -> float:
    """
    Converts seconds to hours.
    """

    return seconds / 3600.0


def _sha256_file(
        file_path: str | Path,
) -> str:
    """
    Calculates the SHA-256 checksum of a file.
    """

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
    """
    Retrieves Git commit/tag/branch information when the project
    is located inside a Git repository.

    Returns None-valued fields when Git information is unavailable.
    """

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
    """
    Saves a PyTorch checkpoint through a temporary file and then
    atomically replaces the destination.

    This reduces the risk of leaving a corrupted checkpoint if
    execution is interrupted during serialization.
    """

    output_path = Path(
        output_path
    )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(
        output_path
    )


def _validate_base_config(
        cfg: dict,
) -> None:
    """
    Validates required base-training configuration parameters.
    """

    required_keys = [

        # ----------------------------------------------------------------------
        # Global experiment
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
        # Architecture
        # ----------------------------------------------------------------------

        "d_model",
        "n_heads",
        "encoder_layers",
        "decoder_layers",
        "dim_feedforward",
        "positional_encoding_max_len",

        # ----------------------------------------------------------------------
        # Training
        # ----------------------------------------------------------------------

        "base_epochs",
        "base_batch_size_cuda",
        "base_batch_size_cpu",
        "base_shuffle_train",
        "base_drop_last",
        "base_dropout",

        # ----------------------------------------------------------------------
        # Initialization
        # ----------------------------------------------------------------------

        "base_weight_init",

        # ----------------------------------------------------------------------
        # Optimizer
        # ----------------------------------------------------------------------

        "base_optimizer",
        "base_max_learning_rate",
        "base_adam_beta1",
        "base_adam_beta2",
        "base_adam_eps",
        "base_weight_decay",

        # ----------------------------------------------------------------------
        # Loss / gradients
        # ----------------------------------------------------------------------

        "base_label_smoothing",
        "base_grad_clip_norm",

        # ----------------------------------------------------------------------
        # Scheduler
        # ----------------------------------------------------------------------

        "base_scheduler",
        "base_scheduler_pct_start",
        "base_scheduler_anneal_strategy",
        "base_scheduler_div_factor",
        "base_scheduler_final_div_factor",
        "base_scheduler_three_phase",
        "base_scheduler_cycle_momentum",
        "base_scheduler_base_momentum",
        "base_scheduler_max_momentum",

        # ----------------------------------------------------------------------
        # AMP / DataLoader
        # ----------------------------------------------------------------------

        "base_amp_enabled",
        "base_num_workers",
        "base_pin_memory",

        # ----------------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------------

        "base_val_metric_interval_epochs",
        "base_val_max_len",
        "base_checkpoint_metric",
    ]

    missing = [
        key
        for key in required_keys
        if key not in cfg
    ]

    if missing:

        raise KeyError(
            "Missing required base-training configuration "
            f"parameter(s): {missing}"
        )

    # ==========================================================================
    # BASIC NUMERICAL VALIDATION
    # ==========================================================================

    if int(
        cfg["experiment_seed"]
    ) < 0:

        raise ValueError(
            "experiment_seed must be non-negative."
        )

    if int(
        cfg["d_model"]
    ) <= 0:

        raise ValueError(
            "d_model must be greater than zero."
        )

    if int(
        cfg["n_heads"]
    ) <= 0:

        raise ValueError(
            "n_heads must be greater than zero."
        )

    if int(
        cfg["encoder_layers"]
    ) <= 0:

        raise ValueError(
            "encoder_layers must be greater than zero."
        )

    if int(
        cfg["decoder_layers"]
    ) <= 0:

        raise ValueError(
            "decoder_layers must be greater than zero."
        )

    if int(
        cfg["dim_feedforward"]
    ) <= 0:

        raise ValueError(
            "dim_feedforward must be greater than zero."
        )

    if int(
        cfg["positional_encoding_max_len"]
    ) <= 0:

        raise ValueError(
            "positional_encoding_max_len must be "
            "greater than zero."
        )

    if int(
        cfg["base_epochs"]
    ) <= 0:

        raise ValueError(
            "base_epochs must be greater than zero."
        )

    if int(
        cfg["base_batch_size_cuda"]
    ) <= 0:

        raise ValueError(
            "base_batch_size_cuda must be greater than zero."
        )

    if int(
        cfg["base_batch_size_cpu"]
    ) <= 0:

        raise ValueError(
            "base_batch_size_cpu must be greater than zero."
        )

    if int(
        cfg["base_num_workers"]
    ) < 0:

        raise ValueError(
            "base_num_workers cannot be negative."
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
            cfg["base_label_smoothing"]
        )
        < 1.0
    ):

        raise ValueError(
            "base_label_smoothing must be in [0, 1)."
        )

    if float(
        cfg["base_max_learning_rate"]
    ) <= 0:

        raise ValueError(
            "base_max_learning_rate must be greater than zero."
        )

    if float(
        cfg["base_adam_eps"]
    ) <= 0:

        raise ValueError(
            "base_adam_eps must be greater than zero."
        )

    if float(
        cfg["base_weight_decay"]
    ) < 0:

        raise ValueError(
            "base_weight_decay cannot be negative."
        )

    if float(
        cfg["base_grad_clip_norm"]
    ) <= 0:

        raise ValueError(
            "base_grad_clip_norm must be greater than zero."
        )

    # ==========================================================================
    # ADAM PARAMETERS
    # ==========================================================================

    beta1 = float(
        cfg["base_adam_beta1"]
    )

    beta2 = float(
        cfg["base_adam_beta2"]
    )

    if not (
        0.0
        <= beta1
        < 1.0
    ):

        raise ValueError(
            "base_adam_beta1 must be in [0, 1)."
        )

    if not (
        0.0
        <= beta2
        < 1.0
    ):

        raise ValueError(
            "base_adam_beta2 must be in [0, 1)."
        )

    # ==========================================================================
    # SCHEDULER
    # ==========================================================================

    pct_start = float(
        cfg[
            "base_scheduler_pct_start"
        ]
    )

    if not (
        0.0
        < pct_start
        < 1.0
    ):

        raise ValueError(
            "base_scheduler_pct_start must be in (0, 1)."
        )

    if float(
        cfg[
            "base_scheduler_div_factor"
        ]
    ) <= 0:

        raise ValueError(
            "base_scheduler_div_factor must be greater than zero."
        )

    if float(
        cfg[
            "base_scheduler_final_div_factor"
        ]
    ) <= 0:

        raise ValueError(
            "base_scheduler_final_div_factor must be greater "
            "than zero."
        )

    if bool(
        cfg[
            "base_scheduler_cycle_momentum"
        ]
    ):

        base_momentum = float(
            cfg[
                "base_scheduler_base_momentum"
            ]
        )

        max_momentum = float(
            cfg[
                "base_scheduler_max_momentum"
            ]
        )

        if not (
            0.0
            <= base_momentum
            <= max_momentum
            < 1.0
        ):

            raise ValueError(
                "OneCycle momentum parameters must satisfy "
                "0 <= base_momentum <= max_momentum < 1."
            )

    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    if int(
        cfg[
            "base_val_metric_interval_epochs"
        ]
    ) <= 0:

        raise ValueError(
            "base_val_metric_interval_epochs must be greater "
            "than zero."
        )

    if int(
        cfg["base_val_max_len"]
    ) <= 0:

        raise ValueError(
            "base_val_max_len must be greater than zero."
        )

    if int(
        cfg["base_val_max_len"]
    ) > int(
        cfg["positional_encoding_max_len"]
    ):

        raise ValueError(
            "base_val_max_len cannot exceed "
            "positional_encoding_max_len."
        )

    # ==========================================================================
    # SUPPORTED IMPLEMENTATION CHOICES
    # ==========================================================================

    if (
        str(
            cfg["base_optimizer"]
        ).lower()
        != "adamw"
    ):

        raise ValueError(
            "This pipeline currently supports "
            "base_optimizer='adamw' only."
        )

    if (
        str(
            cfg["base_scheduler"]
        ).lower()
        != "onecycle"
    ):

        raise ValueError(
            "This pipeline currently supports "
            "base_scheduler='onecycle' only."
        )

    if (
        str(
            cfg["base_weight_init"]
        ).lower()
        != "xavier_uniform"
    ):

        raise ValueError(
            "This pipeline currently supports "
            "base_weight_init='xavier_uniform' only."
        )

    if (
        str(
            cfg[
                "base_checkpoint_metric"
            ]
        ).lower()
        != "chrf++"
    ):

        raise ValueError(
            "This experimental pipeline currently selects "
            "checkpoints using base_checkpoint_metric='chrf++'."
        )


# ==============================================================================
# BASE TRAINING
# ==============================================================================


def run_base_training(
        config_path: str | Path = "config.yaml",
) -> None:

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

    _validate_base_config(
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
        cfg["base_amp_enabled"]
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

    # ==========================================================================
    # BASE TRAIN / VALIDATION DATASETS
    # ==========================================================================

    train_path = (
        tokenized_dir
        / "train_bible_tokenized.tsv"
    )

    val_path = (
        tokenized_dir
        / "val_bible_tokenized.tsv"
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

    # ==========================================================================
    # INPUT ARTIFACT HASHES
    # ==========================================================================

    input_hashes = {

        "config_sha256": (
            _sha256_file(
                config_path
            )
        ),

        "sentencepiece_model_sha256": (
            _sha256_file(
                sp_model_path
            )
        ),

        "train_bible_tokenized_sha256": (
            _sha256_file(
                train_path
            )
        ),

        "val_bible_tokenized_sha256": (
            _sha256_file(
                val_path
            )
        ),
    }

    # ==========================================================================
    # GIT METADATA
    # ==========================================================================

    git_metadata = (
        _get_git_metadata(
            config_path.parent
        )
    )

    # ==========================================================================
    # DATALOADER
    # ==========================================================================

    batch_size = (
        int(
            cfg["base_batch_size_cuda"]
        )
        if device.type == "cuda"
        else int(
            cfg["base_batch_size_cpu"]
        )
    )

    num_workers = int(
        cfg["base_num_workers"]
    )

    pin_memory = (
        bool(
            cfg["base_pin_memory"]
        )
        and device.type == "cuda"
    )

    shuffle_train = bool(
        cfg["base_shuffle_train"]
    )

    drop_last = bool(
        cfg["base_drop_last"]
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

    # ==========================================================================
    # MODEL
    # ==========================================================================

    model = (
        TranslationTransformer(
            vocab_size=vocab_size,

            d_model=int(
                cfg["d_model"]
            ),

            n_heads=int(
                cfg["n_heads"]
            ),

            num_encoder_layers=int(
                cfg["encoder_layers"]
            ),

            num_decoder_layers=int(
                cfg["decoder_layers"]
            ),

            dim_feedforward=int(
                cfg["dim_feedforward"]
            ),

            dropout=float(
                cfg["base_dropout"]
            ),

            pad_id=pad_id,

            positional_encoding_max_len=int(
                cfg[
                    "positional_encoding_max_len"
                ]
            ),
        )
        .to(device)
    )

    # ==========================================================================
    # WEIGHT INITIALIZATION
    # ==========================================================================

    weight_init = str(
        cfg["base_weight_init"]
    ).lower()

    if weight_init == "xavier_uniform":

        for parameter in model.parameters():

            if parameter.dim() > 1:

                nn.init.xavier_uniform_(
                    parameter
                )

    # ==========================================================================
    # PARAMETER COUNTS
    # ==========================================================================

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    # ==========================================================================
    # LOSS
    # ==========================================================================

    criterion = (
        nn.CrossEntropyLoss(
            ignore_index=pad_id,

            label_smoothing=float(
                cfg[
                    "base_label_smoothing"
                ]
            ),
        )
    )

    # ==========================================================================
    # OPTIMIZER
    # ==========================================================================

    optimizer_name = str(
        cfg["base_optimizer"]
    ).lower()

    if optimizer_name == "adamw":

        optimizer = (
            torch.optim.AdamW(
                model.parameters(),

                lr=float(
                    cfg[
                        "base_max_learning_rate"
                    ]
                ),

                betas=(
                    float(
                        cfg[
                            "base_adam_beta1"
                        ]
                    ),

                    float(
                        cfg[
                            "base_adam_beta2"
                        ]
                    ),
                ),

                eps=float(
                    cfg[
                        "base_adam_eps"
                    ]
                ),

                weight_decay=float(
                    cfg[
                        "base_weight_decay"
                    ]
                ),
            )
        )

    else:

        raise RuntimeError(
            f"Unsupported optimizer: "
            f"{optimizer_name}"
        )

    # ==========================================================================
    # LEARNING-RATE SCHEDULER
    # ==========================================================================

    epochs = int(
        cfg["base_epochs"]
    )

    total_steps = (
        len(train_loader)
        * epochs
    )

    scheduler_name = str(
        cfg["base_scheduler"]
    ).lower()

    if scheduler_name == "onecycle":

        scheduler = (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,

                max_lr=float(
                    cfg[
                        "base_max_learning_rate"
                    ]
                ),

                total_steps=total_steps,

                pct_start=float(
                    cfg[
                        "base_scheduler_pct_start"
                    ]
                ),

                anneal_strategy=str(
                    cfg[
                        "base_scheduler_anneal_strategy"
                    ]
                ),

                cycle_momentum=bool(
                    cfg[
                        "base_scheduler_cycle_momentum"
                    ]
                ),

                base_momentum=float(
                    cfg[
                        "base_scheduler_base_momentum"
                    ]
                ),

                max_momentum=float(
                    cfg[
                        "base_scheduler_max_momentum"
                    ]
                ),

                div_factor=float(
                    cfg[
                        "base_scheduler_div_factor"
                    ]
                ),

                final_div_factor=float(
                    cfg[
                        "base_scheduler_final_div_factor"
                    ]
                ),

                three_phase=bool(
                    cfg[
                        "base_scheduler_three_phase"
                    ]
                ),
            )
        )

    else:

        raise RuntimeError(
            f"Unsupported scheduler: "
            f"{scheduler_name}"
        )

    initial_scheduled_lr = float(
        optimizer.param_groups[0]["lr"]
    )

    initial_scheduled_beta1 = float(
        optimizer.param_groups[0]["betas"][0]
    )

    # ==========================================================================
    # MIXED PRECISION
    # ==========================================================================

    scaler = GradScaler(
        enabled=use_amp
    )

    # ==========================================================================
    # ARTIFACT NAMES
    # ==========================================================================

    model_tag = (
        f"{cfg['src']}_to_"
        f"{cfg['tgt']}"
        f"_Enc{cfg['encoder_layers']}"
        f"_Dec{cfg['decoder_layers']}"
        f"_Seed{experiment_seed}"
    )

    best_model_path = (
        models_dir
        / (
            f"base_checkpoint_"
            f"{model_tag}.pt"
        )
    )

    history_path = (
        results_dir
        / (
            f"base_training_history_"
            f"{model_tag}.csv"
        )
    )

    summary_path = (
        results_dir
        / (
            f"base_training_summary_"
            f"{model_tag}.json"
        )
    )

    # ==========================================================================
    # SETUP TIMING
    # ==========================================================================

    setup_time_seconds = (
        time.perf_counter()
        - script_start_time
    )

    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats(
            device
        )

    # ==========================================================================
    # TRAINING STATE
    # ==========================================================================

    minimum_val_loss = float(
        "inf"
    )

    best_val_chrf = float(
        "-inf"
    )

    best_val_bleu_at_best_chrf = None

    best_bleu_signature = None
    best_chrf_signature = None

    best_epoch = None

    history = []

    total_train_phase_seconds = 0.0
    total_val_loss_seconds = 0.0
    total_val_metrics_seconds = 0.0

    validation_metric_evaluations = 0

    # ==========================================================================
    # CONSOLE HEADER
    # ==========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"BASE TRAINING "
        f"[{cfg['src'].upper()} -> "
        f"{cfg['tgt'].upper()} | "
        f"Enc:{cfg['encoder_layers']} "
        f"Dec:{cfg['decoder_layers']}]"
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
        f"Parameters: "
        f"{total_parameters:,} "
        f"(trainable: "
        f"{trainable_parameters:,})"
    )

    print(
        f"Positional encoding "
        f"max length: "
        f"{cfg['positional_encoding_max_len']}"
    )

    print(
        f"Optimizer: "
        f"{optimizer_name} | "
        f"Scheduler: "
        f"{scheduler_name}"
    )

    print(
        f"Max LR: "
        f"{float(cfg['base_max_learning_rate']):.2e} | "
        f"Initial scheduled LR: "
        f"{initial_scheduled_lr:.2e}"
    )

    print(
        f"Initial scheduler beta1: "
        f"{initial_scheduled_beta1:.4f} | "
        f"Cycle momentum: "
        f"{bool(cfg['base_scheduler_cycle_momentum'])}"
    )

    print(
        f"Dropout: "
        f"{float(cfg['base_dropout']):.2f} | "
        f"Label smoothing: "
        f"{float(cfg['base_label_smoothing']):.2f}"
    )

    print(
        f"Checkpoint criterion: "
        f"{cfg['base_checkpoint_metric']}"
    )

    print(
        f"Validation decoding: "
        f"greedy | "
        f"max_len="
        f"{cfg['base_val_max_len']}"
    )

    print(
        "Validation metric interval: "
        f"every "
        f"{cfg['base_val_metric_interval_epochs']} "
        "epoch(s)"
    )

    if (
        git_metadata["commit"]
        is not None
    ):

        print(
            f"Git commit: "
            f"{git_metadata['commit']}"
        )

        print(
            f"Git dirty worktree: "
            f"{git_metadata['dirty_worktree']}"
        )

    print(
        "=" * 80
    )

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================

    _sync_cuda(
        device
    )

    training_loop_start_time = (
        time.perf_counter()
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        _sync_cuda(
            device
        )

        epoch_start_time = (
            time.perf_counter()
        )

        # ======================================================================
        # TRAINING PHASE
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
                            "base_grad_clip_norm"
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
                            "base_grad_clip_norm"
                        ]
                    ),
                )

                optimizer.step()

            scheduler.step()

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

        val_loss_pairs_per_second = (
            len(val_dataset)
            / val_loss_seconds
        )

        minimum_val_loss = min(
            minimum_val_loss,
            val_loss,
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
                "base_val_metric_interval_epochs"
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
                dataset=val_dataset,
                sp=sp,
                device=device,

                max_len=int(
                    cfg[
                        "base_val_max_len"
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
            # BEST CHECKPOINT — VALIDATION chrF++
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

                checkpoint = {

                    "checkpoint_type": (
                        "base_training"
                    ),

                    "model_tag": (
                        model_tag
                    ),

                    "epoch": (
                        epoch
                    ),

                    "experiment_seed": (
                        experiment_seed
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

                    "architecture": {

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
                    },

                    "input_hashes": (
                        input_hashes
                    ),

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
        # EPOCH LOGGING
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

        current_beta1 = float(
            optimizer.param_groups[0][
                "betas"
            ][0]
        )

        bleu_str = (
            f"{val_bleu:.2f}"
            if val_bleu is not None
            else "-"
        )

        chrf_str = (
            f"{val_chrf:.2f}"
            if val_chrf is not None
            else "-"
        )

        metrics_time_str = (
            f"{val_metrics_seconds:.1f}s"
            if val_metrics_seconds
            is not None
            else "-"
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{epochs:03d} | "

            f"Train Loss: "
            f"{train_loss:.4f} | "

            f"Val Loss: "
            f"{val_loss:.4f} | "

            f"Val BLEU: "
            f"{bleu_str} | "

            f"Val chrF++: "
            f"{chrf_str} | "

            f"LR: "
            f"{current_lr:.2e} | "

            f"Beta1: "
            f"{current_beta1:.4f} | "

            f"Train: "
            f"{train_phase_seconds:.1f}s | "

            f"ValLoss: "
            f"{val_loss_seconds:.1f}s | "

            f"ValMetrics: "
            f"{metrics_time_str} | "

            f"Epoch: "
            f"{epoch_time_seconds:.1f}s"
        )

        history.append({

            "epoch": (
                epoch
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

            "learning_rate": (
                current_lr
            ),

            "adam_beta1": (
                current_beta1
            ),

            "checkpoint_saved": (
                checkpoint_saved
            ),

            "train_time_seconds": (
                train_phase_seconds
            ),

            "train_pairs_per_second": (
                train_pairs_per_second
            ),

            "val_loss_time_seconds": (
                val_loss_seconds
            ),

            "val_loss_pairs_per_second": (
                val_loss_pairs_per_second
            ),

            "val_metrics_time_seconds": (
                val_metrics_seconds
            ),

            "val_metrics_pairs_per_second": (
                val_metrics_pairs_per_second
            ),

            "epoch_time_seconds": (
                epoch_time_seconds
            ),
        })

        # Persist history after every epoch.
        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
            encoding="utf-8",
        )

    # ==========================================================================
    # FINAL TIMING
    # ==========================================================================

    _sync_cuda(
        device
    )

    training_loop_time_seconds = (
        time.perf_counter()
        - training_loop_start_time
    )

    total_execution_time_seconds = (
        time.perf_counter()
        - script_start_time
    )

    # ==========================================================================
    # HARDWARE INFORMATION
    # ==========================================================================

    if device.type == "cuda":

        peak_gpu_memory_allocated_mb = (
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024 ** 2)
        )

        peak_gpu_memory_reserved_mb = (
            torch.cuda.max_memory_reserved(
                device
            )
            / (1024 ** 2)
        )

        device_name = (
            torch.cuda.get_device_name(
                device
            )
        )

        gpu_total_memory_mb = (
            torch.cuda.get_device_properties(
                device
            ).total_memory
            / (1024 ** 2)
        )

    else:

        peak_gpu_memory_allocated_mb = None
        peak_gpu_memory_reserved_mb = None
        gpu_total_memory_mb = None

        device_name = (
            platform.processor()
            or "CPU"
        )

    # ==========================================================================
    # SANITY CHECK
    # ==========================================================================

    if best_epoch is None:

        raise RuntimeError(
            "No best checkpoint was created. "
            "Check the validation metric schedule."
        )

    # ==========================================================================
    # AGGREGATE THROUGHPUT
    # ==========================================================================

    overall_train_pairs_per_second = (
        len(train_dataset)
        * epochs
        / total_train_phase_seconds
    )

    # ==========================================================================
    # SUMMARY JSON
    # ==========================================================================

    summary = {

        "model_tag": (
            model_tag
        ),

        "experiment_type": (
            "Base-Training"
        ),

        "training_domain": (
            "Biblical"
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
            cfg["base_val_max_len"]
        ),

        "validation_metric_interval_epochs": int(
            cfg[
                "base_val_metric_interval_epochs"
            ]
        ),

        "validation_metric_evaluations": (
            validation_metric_evaluations
        ),

        "best_checkpoint_epoch": (
            best_epoch
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

        "checkpoint": (
            str(best_model_path)
        ),

        "history_file": (
            str(history_path)
        ),

        "dataset": {

            "train_file": (
                str(train_path)
            ),

            "validation_file": (
                str(val_path)
            ),

            "train_pairs": (
                len(train_dataset)
            ),

            "validation_pairs": (
                len(val_dataset)
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

            "architecture": {

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
            },
        },

        "optimizer_scheduler": {

            "optimizer": (
                optimizer_name
            ),

            "scheduler": (
                scheduler_name
            ),

            "maximum_learning_rate": float(
                cfg[
                    "base_max_learning_rate"
                ]
            ),

            "initial_scheduled_learning_rate": (
                initial_scheduled_lr
            ),

            "initial_scheduled_beta1": (
                initial_scheduled_beta1
            ),

            "cycle_momentum": bool(
                cfg[
                    "base_scheduler_cycle_momentum"
                ]
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

            "overall_train_pairs_per_second": (
                overall_train_pairs_per_second
            ),

            "mean_epoch_time_seconds": (
                sum(
                    row[
                        "epoch_time_seconds"
                    ]
                    for row in history
                )
                / len(history)
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

            "hostname": (
                platform.node()
            ),
        },

        "config": (
            cfg
        ),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # ==========================================================================
    # FINAL CONSOLE SUMMARY
    # ==========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BASE TRAINING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Best epoch              : "
        f"{best_epoch}"
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
        f"Training loop time      : "
        f"{_hours(training_loop_time_seconds):.2f} h"
    )

    print(
        f"Total execution time    : "
        f"{_hours(total_execution_time_seconds):.2f} h"
    )

    print(
        f"Train throughput        : "
        f"{overall_train_pairs_per_second:.2f} pairs/s"
    )

    if (
        peak_gpu_memory_allocated_mb
        is not None
    ):

        print(
            f"Peak GPU memory         : "
            f"{peak_gpu_memory_allocated_mb:.1f} MB"
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

    run_base_training(
        config_path="config.yaml"
    )