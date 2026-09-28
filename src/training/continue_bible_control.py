# ==============================================
# ./src/training/continue_bible_control.py
# ==============================================

import hashlib
import json
import platform
import random
import subprocess
import sys
import time
import warnings
from importlib.metadata import PackageNotFoundError, version
from functools import partial
from pathlib import Path

import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from src.data.dataset import ParallelTranslationDataset, collate_fn_pad
from src.models.transformer import TranslationTransformer
from src.utils.config import load_config
from src.utils.reproducibility import set_seed, seed_worker


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.nn.modules.transformer",
)


# ==============================================================================
# AUXILIARY FUNCTIONS
# ==============================================================================


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hours(seconds: float) -> float:
    return seconds / 3600.0


def _sha256_file(file_path: str | Path) -> str:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {file_path}")

    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_integer_sequence(values: list[int]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _get_git_metadata(repository_dir: str | Path) -> dict:
    repository_dir = Path(repository_dir).resolve()
    metadata = {
        "commit": None,
        "branch": None,
        "exact_tag": None,
        "dirty_worktree": None,
    }

    try:
        commit = subprocess.run(
            ["git", "-C", str(repository_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        metadata["commit"] = commit.stdout.strip()

        branch = subprocess.run(
            ["git", "-C", str(repository_dir), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        metadata["branch"] = branch.stdout.strip()

        status = subprocess.run(
            ["git", "-C", str(repository_dir), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        metadata["dirty_worktree"] = bool(status.stdout.strip())

        tag = subprocess.run(
            ["git", "-C", str(repository_dir), "describe", "--tags", "--exact-match"],
            capture_output=True,
            text=True,
            check=False,
        )
        if tag.returncode == 0:
            metadata["exact_tag"] = tag.stdout.strip()

    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return metadata


def _atomic_torch_save(payload: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_json_dump(payload: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4, ensure_ascii=False)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_csv_dump(rows: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        pd.DataFrame(rows).to_csv(
            temporary_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _expected_model_tag(cfg: dict) -> str:
    return (
        f"{cfg['src']}_to_{cfg['tgt']}"
        f"_Enc{cfg['encoder_layers']}"
        f"_Dec{cfg['decoder_layers']}"
        f"_Seed{cfg['experiment_seed']}"
    )


def _validate_control_config(cfg: dict) -> None:
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

    missing = [key for key in required_keys if key not in cfg]
    if missing:
        raise KeyError(
            "Missing required continued-training control configuration "
            f"parameter(s): {missing}"
        )

    if int(cfg["experiment_seed"]) < 0:
        raise ValueError("experiment_seed must be non-negative.")

    if not str(cfg["src"]).strip() or not str(cfg["tgt"]).strip():
        raise ValueError("src and tgt cannot be empty.")
    if str(cfg["src"]) == str(cfg["tgt"]):
        raise ValueError("src and tgt must be different languages.")

    positive_ints = [
        "d_model",
        "n_heads",
        "encoder_layers",
        "decoder_layers",
        "dim_feedforward",
        "positional_encoding_max_len",
        "ct_subset_size",
        "ct_batch_size_cuda",
        "ct_batch_size_cpu",
    ]
    for key in positive_ints:
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be greater than zero.")

    if int(cfg["ct_subset_seed"]) < 0:
        raise ValueError("ct_subset_seed must be non-negative.")
    if int(cfg["ct_num_workers"]) < 0:
        raise ValueError("ct_num_workers cannot be negative.")

    if str(cfg["ct_budget_source"]).lower() != "finetuned_checkpoint":
        raise ValueError(
            "ct_budget_source must be 'finetuned_checkpoint' so the control "
            "matches the selected adapted checkpoint's optimizer budget."
        )
    if str(cfg["ct_optimizer"]).lower() != "adamw":
        raise ValueError("ct_optimizer must be 'adamw'.")
    if str(cfg["ct_scheduler"]).lower() != "cosine":
        raise ValueError("ct_scheduler must be 'cosine'.")
    if str(cfg["ct_scheduler_step_unit"]).lower() != "optimizer_step":
        raise ValueError("ct_scheduler_step_unit must be 'optimizer_step'.")

    if float(cfg["ct_learning_rate"]) <= 0:
        raise ValueError("ct_learning_rate must be greater than zero.")
    if float(cfg["ct_scheduler_eta_min"]) < 0:
        raise ValueError("ct_scheduler_eta_min cannot be negative.")
    if float(cfg["ct_scheduler_eta_min"]) > float(cfg["ct_learning_rate"]):
        raise ValueError("ct_scheduler_eta_min cannot exceed ct_learning_rate.")
    if float(cfg["ct_adam_eps"]) <= 0:
        raise ValueError("ct_adam_eps must be greater than zero.")
    if float(cfg["ct_weight_decay"]) < 0:
        raise ValueError("ct_weight_decay cannot be negative.")
    if not (0.0 <= float(cfg["ct_label_smoothing"]) < 1.0):
        raise ValueError("ct_label_smoothing must be in [0, 1).")
    if float(cfg["ct_grad_clip_norm"]) <= 0:
        raise ValueError("ct_grad_clip_norm must be greater than zero.")
    if not (0.0 <= float(cfg["ct_adam_beta1"]) < 1.0):
        raise ValueError("ct_adam_beta1 must be in [0, 1).")
    if not (0.0 <= float(cfg["ct_adam_beta2"]) < 1.0):
        raise ValueError("ct_adam_beta2 must be in [0, 1).")


def _validate_checkpoint_identity(
    checkpoint: dict,
    expected_type: str,
    expected_model_tag: str,
    cfg: dict,
) -> None:
    if checkpoint.get("checkpoint_type") != expected_type:
        raise ValueError(
            f"Expected checkpoint_type={expected_type!r}, found "
            f"{checkpoint.get('checkpoint_type')!r}."
        )
    if checkpoint.get("model_tag") != expected_model_tag:
        raise ValueError("Checkpoint model_tag does not match the requested run.")
    if int(checkpoint.get("experiment_seed", -1)) != int(cfg["experiment_seed"]):
        raise ValueError("Checkpoint seed does not match config.")
    if checkpoint.get("src") != cfg["src"] or checkpoint.get("tgt") != cfg["tgt"]:
        raise ValueError("Checkpoint translation direction does not match config.")


def _validate_architecture(checkpoint: dict, cfg: dict, label: str) -> None:
    expected = {
        "d_model": int(cfg["d_model"]),
        "n_heads": int(cfg["n_heads"]),
        "encoder_layers": int(cfg["encoder_layers"]),
        "decoder_layers": int(cfg["decoder_layers"]),
        "dim_feedforward": int(cfg["dim_feedforward"]),
        "dropout": float(cfg["base_dropout"]),
        "positional_encoding_max_len": int(cfg["positional_encoding_max_len"]),
    }
    stored = checkpoint.get("architecture", {})
    for key, expected_value in expected.items():
        if stored.get(key) != expected_value:
            raise ValueError(
                f"{label} architecture mismatch for {key}: "
                f"stored={stored.get(key)!r}, current={expected_value!r}."
            )


def _validate_matched_optimization(
    cfg: dict,
    finetuned_checkpoint: dict,
    device: torch.device,
) -> tuple[int, int, int]:
    """
    Verifies that the control protocol matches the optimization settings used
    by the conversational fine-tuning checkpoint.

    Returns
    -------
    tuple
        (batch_size, matched_optimizer_steps, scheduler_horizon_steps)
    """

    ft_optimization = finetuned_checkpoint.get("optimization", {})
    ft_config = finetuned_checkpoint.get("config", {})

    batch_size = (
        int(cfg["ct_batch_size_cuda"])
        if device.type == "cuda"
        else int(cfg["ct_batch_size_cpu"])
    )

    comparisons = {
        "batch_size": (batch_size, ft_optimization.get("batch_size")),
        "optimizer": (str(cfg["ct_optimizer"]).lower(), str(ft_optimization.get("optimizer", "")).lower()),
        "learning_rate": (float(cfg["ct_learning_rate"]), ft_optimization.get("learning_rate")),
        "scheduler": (str(cfg["ct_scheduler"]).lower(), str(ft_optimization.get("scheduler", "")).lower()),
        "scheduler_step_unit": (
            str(cfg["ct_scheduler_step_unit"]).lower(),
            str(ft_optimization.get("scheduler_step_unit", "")).lower(),
        ),
        "scheduler_eta_min": (
            float(cfg["ct_scheduler_eta_min"]),
            ft_optimization.get("scheduler_eta_min"),
        ),
        "shuffle_train": (bool(cfg["ct_shuffle_train"]), ft_config.get("ft_shuffle_train")),
        "drop_last": (bool(cfg["ct_drop_last"]), ft_config.get("ft_drop_last")),
        "adam_beta1": (float(cfg["ct_adam_beta1"]), ft_config.get("ft_adam_beta1")),
        "adam_beta2": (float(cfg["ct_adam_beta2"]), ft_config.get("ft_adam_beta2")),
        "adam_eps": (float(cfg["ct_adam_eps"]), ft_config.get("ft_adam_eps")),
        "weight_decay": (float(cfg["ct_weight_decay"]), ft_config.get("ft_weight_decay")),
        "label_smoothing": (
            float(cfg["ct_label_smoothing"]),
            ft_config.get("ft_label_smoothing"),
        ),
        "grad_clip_norm": (
            float(cfg["ct_grad_clip_norm"]),
            ft_config.get("ft_grad_clip_norm"),
        ),
    }

    mismatches = []
    for label, (control_value, ft_value) in comparisons.items():
        if control_value != ft_value:
            mismatches.append(
                f"{label}: control={control_value!r}, fine_tuning={ft_value!r}"
            )

    if mismatches:
        raise ValueError(
            "Continued-training control does not match the fine-tuning "
            "optimization protocol:\n  - " + "\n  - ".join(mismatches)
        )

    matched_optimizer_steps = int(finetuned_checkpoint.get("optimizer_step", 0))
    scheduler_horizon_steps = int(
        finetuned_checkpoint.get("planned_total_optimizer_steps", 0)
    )

    nested_steps_per_epoch = int(ft_optimization.get("steps_per_epoch", 0))
    top_steps_per_epoch = int(finetuned_checkpoint.get("steps_per_epoch", 0))
    if nested_steps_per_epoch != top_steps_per_epoch:
        raise ValueError(
            "Fine-tuned checkpoint has inconsistent steps_per_epoch metadata."
        )

    nested_horizon = int(ft_optimization.get("planned_total_optimizer_steps", 0))
    if nested_horizon != scheduler_horizon_steps:
        raise ValueError(
            "Fine-tuned checkpoint has inconsistent scheduler-horizon metadata."
        )

    nested_checkpoint_step = int(ft_optimization.get("optimizer_step_at_checkpoint", 0))
    if nested_checkpoint_step != matched_optimizer_steps:
        raise ValueError(
            "Fine-tuned checkpoint has inconsistent selected-step metadata."
        )

    if matched_optimizer_steps <= 0:
        raise ValueError("Fine-tuned checkpoint optimizer_step must be positive.")
    if scheduler_horizon_steps <= 0:
        raise ValueError(
            "Fine-tuned checkpoint planned_total_optimizer_steps must be positive."
        )
    if matched_optimizer_steps > scheduler_horizon_steps:
        raise ValueError(
            "Fine-tuned checkpoint optimizer_step exceeds its planned scheduler horizon."
        )

    return batch_size, matched_optimizer_steps, scheduler_horizon_steps


def _sample_subset_indices(
    population_size: int,
    subset_size: int,
    subset_seed: int,
) -> list[int]:
    if subset_size > population_size:
        raise ValueError(
            f"ct_subset_size={subset_size} exceeds biblical training size "
            f"{population_size}."
        )

    rng = random.Random(subset_seed)
    selected = rng.sample(range(population_size), subset_size)
    selected.sort()
    return selected


# ==============================================================================
# CONTINUED-TRAINING CONTROL
# ==============================================================================


def run_continued_bible_control(
    config_path: str | Path = "config.yaml",
    base_checkpoint_path: str | Path | None = None,
    finetuned_checkpoint_path: str | Path | None = None,
) -> None:
    """
    Runs the E2 continued-training control for one architecture/direction/seed.

    The control starts from the same selected biblical base checkpoint as the
    conversational fine-tuning run, trains on a deterministic size-matched
    subset of biblical training pairs, and uses the exact optimizer-step budget
    and LR-schedule horizon stored in the selected fine-tuned checkpoint.

    Test sets are intentionally not accessed here.
    """

    script_start_time = time.perf_counter()
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    _validate_control_config(cfg)

    experiment_seed = int(cfg["experiment_seed"])
    deterministic_training = bool(cfg["deterministic_training"])
    set_seed(experiment_seed, deterministic=deterministic_training)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["ct_amp_enabled"]) and device.type == "cuda"

    print(
        f"Device: {device.type.upper()} | AMP: {use_amp} | "
        f"Deterministic: {deterministic_training} | Seed: {experiment_seed}"
    )

    tokenizer_dir = Path(cfg["tokenizer_dir"])
    tokenized_dir = Path(cfg["tokenized_dir"])
    models_dir = Path(cfg["models_dir"])
    results_dir = Path(cfg["results_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_tag = _expected_model_tag(cfg)

    if base_checkpoint_path is None:
        base_checkpoint_path = models_dir / f"base_checkpoint_{model_tag}.pt"
    else:
        base_checkpoint_path = Path(base_checkpoint_path)

    if finetuned_checkpoint_path is None:
        finetuned_checkpoint_path = models_dir / f"finetuned_checkpoint_{model_tag}.pt"
    else:
        finetuned_checkpoint_path = Path(finetuned_checkpoint_path)

    if not base_checkpoint_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint_path}")
    if not finetuned_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint required to define matched budget not found: "
            f"{finetuned_checkpoint_path}"
        )

    control_checkpoint_path = models_dir / f"continued_control_checkpoint_{model_tag}.pt"
    history_path = results_dir / f"continued_control_history_{model_tag}.csv"
    summary_path = results_dir / f"continued_control_summary_{model_tag}.json"

    sp_model_path = tokenizer_dir / "SentencePiece.model"
    if not sp_model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {sp_model_path}")

    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    pad_id = sp.pad_id()
    vocab_size = sp.vocab_size()
    if pad_id < 0:
        raise ValueError("SentencePiece PAD token is not defined.")

    tokenizer_sha256 = _sha256_file(sp_model_path)
    base_checkpoint_sha256 = _sha256_file(base_checkpoint_path)
    finetuned_checkpoint_sha256 = _sha256_file(finetuned_checkpoint_path)

    base_checkpoint = torch.load(
        base_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    finetuned_checkpoint = torch.load(
        finetuned_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(base_checkpoint, dict) or not isinstance(finetuned_checkpoint, dict):
        raise TypeError("Base and fine-tuned checkpoints must contain dictionaries.")

    _validate_checkpoint_identity(base_checkpoint, "base_training", model_tag, cfg)
    _validate_checkpoint_identity(finetuned_checkpoint, "fine_tuning", model_tag, cfg)
    _validate_architecture(base_checkpoint, cfg, "Base checkpoint")
    _validate_architecture(finetuned_checkpoint, cfg, "Fine-tuned checkpoint")

    if int(base_checkpoint.get("vocab_size", -1)) != vocab_size:
        raise ValueError("Base checkpoint vocabulary size does not match tokenizer.")
    if int(base_checkpoint.get("pad_id", -1)) != pad_id:
        raise ValueError("Base checkpoint PAD ID does not match tokenizer.")
    if int(finetuned_checkpoint.get("vocab_size", -1)) != vocab_size:
        raise ValueError("Fine-tuned checkpoint vocabulary size does not match tokenizer.")
    if int(finetuned_checkpoint.get("pad_id", -1)) != pad_id:
        raise ValueError("Fine-tuned checkpoint PAD ID does not match tokenizer.")

    base_tokenizer_hash = base_checkpoint.get("input_hashes", {}).get(
        "sentencepiece_model_sha256"
    )
    if base_tokenizer_hash and base_tokenizer_hash != tokenizer_sha256:
        raise ValueError("Tokenizer differs from the tokenizer used for base training.")

    ft_tokenizer_hash = finetuned_checkpoint.get("fine_tuning_input_hashes", {}).get(
        "sentencepiece_model_sha256"
    )
    if ft_tokenizer_hash and ft_tokenizer_hash != tokenizer_sha256:
        raise ValueError("Tokenizer differs from the tokenizer used for fine-tuning.")

    ft_base_hash = finetuned_checkpoint.get("base_checkpoint", {}).get("sha256")
    if ft_base_hash != base_checkpoint_sha256:
        raise ValueError(
            "Fine-tuned checkpoint was not derived from the supplied base checkpoint."
        )

    batch_size, matched_optimizer_steps, scheduler_horizon_steps = (
        _validate_matched_optimization(cfg, finetuned_checkpoint, device)
    )

    train_bible_path = tokenized_dir / "train_bible_tokenized.tsv"
    train_dialogue_path = tokenized_dir / "train_dialogue_tokenized.tsv"

    current_train_bible_hash = _sha256_file(train_bible_path)
    current_train_dialogue_hash = _sha256_file(train_dialogue_path)

    stored_base_train_hash = base_checkpoint.get("input_hashes", {}).get(
        "train_bible_tokenized_sha256"
    )
    if stored_base_train_hash and stored_base_train_hash != current_train_bible_hash:
        raise ValueError(
            "Current biblical training file differs from the file used to train "
            "the selected base checkpoint."
        )

    stored_ft_dialogue_hash = finetuned_checkpoint.get(
        "fine_tuning_input_hashes", {}
    ).get("train_dialogue_tokenized_sha256")
    if stored_ft_dialogue_hash and stored_ft_dialogue_hash != current_train_dialogue_hash:
        raise ValueError(
            "Current conversational training file differs from the file used by "
            "the selected fine-tuned checkpoint."
        )

    bible_dataset = ParallelTranslationDataset(
        tsv_path=train_bible_path,
        src_col=f"{cfg['src']}_ids",
        tgt_col=f"{cfg['tgt']}_ids",
    )
    dialogue_dataset = ParallelTranslationDataset(
        tsv_path=train_dialogue_path,
        src_col=f"{cfg['src']}_ids",
        tgt_col=f"{cfg['tgt']}_ids",
    )

    subset_size = int(cfg["ct_subset_size"])
    if subset_size != len(dialogue_dataset):
        raise ValueError(
            "ct_subset_size must exactly match the conversational training-set "
            f"size for E2: {subset_size} != {len(dialogue_dataset)}."
        )

    subset_seed = int(cfg["ct_subset_seed"])
    selected_indices = _sample_subset_indices(
        population_size=len(bible_dataset),
        subset_size=subset_size,
        subset_seed=subset_seed,
    )
    subset_indices_sha256 = _sha256_integer_sequence(selected_indices)
    control_dataset = Subset(bible_dataset, selected_indices)

    num_workers = int(cfg["ct_num_workers"])
    pin_memory = bool(cfg["ct_pin_memory"]) and device.type == "cuda"
    shuffle_train = bool(cfg["ct_shuffle_train"])
    drop_last = bool(cfg["ct_drop_last"])

    generator = torch.Generator()
    generator.manual_seed(experiment_seed)

    collate_with_pad = partial(collate_fn_pad, pad_id=pad_id)
    train_loader = DataLoader(
        control_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=drop_last,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        collate_fn=collate_with_pad,
    )

    if len(train_loader) == 0:
        raise ValueError("Continued-control DataLoader contains zero batches.")

    ft_steps_per_epoch = int(finetuned_checkpoint.get("steps_per_epoch", 0))
    if len(train_loader) != ft_steps_per_epoch:
        raise ValueError(
            "Size-matched biblical subset does not reproduce the fine-tuning "
            f"steps/epoch: control={len(train_loader)}, fine_tuning={ft_steps_per_epoch}."
        )

    architecture = dict(base_checkpoint["architecture"])
    model = TranslationTransformer(
        vocab_size=int(base_checkpoint["vocab_size"]),
        d_model=int(architecture["d_model"]),
        n_heads=int(architecture["n_heads"]),
        num_encoder_layers=int(architecture["encoder_layers"]),
        num_decoder_layers=int(architecture["decoder_layers"]),
        dim_feedforward=int(architecture["dim_feedforward"]),
        dropout=float(architecture["dropout"]),
        pad_id=int(base_checkpoint["pad_id"]),
        positional_encoding_max_len=int(architecture["positional_encoding_max_len"]),
    ).to(device)
    model.load_state_dict(base_checkpoint["model_state_dict"], strict=True)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if total_parameters != trainable_parameters:
        raise RuntimeError("Continued-training control requires full-model training.")

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
        label_smoothing=float(cfg["ct_label_smoothing"]),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["ct_learning_rate"]),
        betas=(float(cfg["ct_adam_beta1"]), float(cfg["ct_adam_beta2"])),
        eps=float(cfg["ct_adam_eps"]),
        weight_decay=float(cfg["ct_weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_horizon_steps,
        eta_min=float(cfg["ct_scheduler_eta_min"]),
    )
    initial_lr = float(optimizer.param_groups[0]["lr"])
    scaler = GradScaler(enabled=use_amp)

    input_hashes = {
        "config_sha256": _sha256_file(config_path),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "finetuned_checkpoint_sha256": finetuned_checkpoint_sha256,
        "sentencepiece_model_sha256": tokenizer_sha256,
        "train_bible_tokenized_sha256": current_train_bible_hash,
        "train_dialogue_tokenized_sha256": current_train_dialogue_hash,
        "selected_bible_subset_indices_sha256": subset_indices_sha256,
    }
    git_metadata = _get_git_metadata(config_path.parent)

    equivalent_full_epochs = matched_optimizer_steps // len(train_loader)
    remainder_steps = matched_optimizer_steps % len(train_loader)

    print("\n" + "=" * 80)
    print(
        f"CONTINUED BIBLICAL TRAINING CONTROL [{cfg['src'].upper()} -> "
        f"{cfg['tgt'].upper()} | Enc:{architecture['encoder_layers']} "
        f"Dec:{architecture['decoder_layers']}]"
    )
    print(f"Biblical subset pairs  : {len(control_dataset):,}")
    print(f"Matched dialogue pairs : {len(dialogue_dataset):,}")
    print(f"Subset seed            : {subset_seed}")
    print(f"Batch size             : {batch_size}")
    print(f"Steps/epoch            : {len(train_loader)}")
    print(f"Matched optimizer steps: {matched_optimizer_steps}")
    print(f"FT scheduler horizon   : {scheduler_horizon_steps}")
    print(
        f"Equivalent duration    : {equivalent_full_epochs} full epoch(s)"
        + (f" + {remainder_steps} step(s)" if remainder_steps else "")
    )
    print(
        f"Optimizer/Scheduler    : {str(cfg['ct_optimizer']).lower()} / "
        f"{str(cfg['ct_scheduler']).lower()} per optimizer step"
    )
    print(
        f"Initial LR / eta_min   : {initial_lr:.2e} / "
        f"{float(cfg['ct_scheduler_eta_min']):.2e}"
    )
    print(
        f"Budget source          : selected FT checkpoint "
        f"epoch={finetuned_checkpoint['epoch']}, step={matched_optimizer_steps}"
    )
    print("Checkpoint policy      : save exactly at matched optimizer step")
    print("=" * 80)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    setup_time_seconds = time.perf_counter() - script_start_time
    optimizer_steps_completed = 0
    total_train_seconds = 0.0
    history: list[dict] = []
    control_epoch = 0
    training_start = time.perf_counter()

    while optimizer_steps_completed < matched_optimizer_steps:
        control_epoch += 1
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        batches_completed = 0

        for src, tgt in train_loader:
            if optimizer_steps_completed >= matched_optimizer_steps:
                break

            src = src.to(device, non_blocking=pin_memory)
            tgt = tgt.to(device, non_blocking=pin_memory)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:].contiguous()

            optimizer.zero_grad(set_to_none=True)
            with autocast(
                device_type=device.type,
                enabled=use_amp,
                dtype=torch.float16,
            ):
                logits = model(src, tgt_in)
                loss = criterion(
                    logits.reshape(-1, vocab_size),
                    tgt_out.reshape(-1),
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg["ct_grad_clip_norm"]),
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg["ct_grad_clip_norm"]),
                )
                optimizer.step()

            scheduler.step()
            optimizer_steps_completed += 1
            batches_completed += 1
            loss_sum += float(loss.item())

        _sync_cuda(device)
        epoch_seconds = time.perf_counter() - epoch_start
        total_train_seconds += epoch_seconds

        if batches_completed == 0:
            raise RuntimeError("No optimizer batches were completed in control epoch.")

        train_loss = loss_sum / batches_completed
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "control_epoch": control_epoch,
                "optimizer_steps_completed": optimizer_steps_completed,
                "batches_completed_this_epoch": batches_completed,
                "train_loss": train_loss,
                "learning_rate": current_lr,
                "epoch_time_seconds": epoch_seconds,
            }
        )
        _atomic_csv_dump(history, history_path)

        print(
            f"Control epoch {control_epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Step: {optimizer_steps_completed}/{matched_optimizer_steps} | "
            f"Batches: {batches_completed} | "
            f"Time: {epoch_seconds:.1f}s"
        )

    if optimizer_steps_completed != matched_optimizer_steps:
        raise RuntimeError(
            "Control training ended with an unexpected optimizer-step count: "
            f"{optimizer_steps_completed} != {matched_optimizer_steps}."
        )

    checkpoint = {
        "checkpoint_type": "continued_bible_control",
        "model_tag": model_tag,
        "experiment_seed": experiment_seed,
        "training_domain": "Biblical-Control-Subset",
        "checkpoint_selection_metric": None,
        "checkpoint_policy": "matched_finetuned_optimizer_step",
        "optimizer_step": optimizer_steps_completed,
        "steps_per_epoch": len(train_loader),
        "scheduler_horizon_steps": scheduler_horizon_steps,
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "pad_id": pad_id,
        "src": cfg["src"],
        "tgt": cfg["tgt"],
        "architecture": architecture,
        "base_checkpoint": {
            "path": str(base_checkpoint_path),
            "sha256": base_checkpoint_sha256,
            "epoch": int(base_checkpoint["epoch"]),
            "best_val_chrf": base_checkpoint.get("best_val_chrf"),
            "val_bleu_at_best_chrf": base_checkpoint.get("val_bleu_at_best_chrf"),
            "input_hashes": base_checkpoint.get("input_hashes"),
            "git": base_checkpoint.get("git"),
        },
        "matched_finetuned_checkpoint": {
            "path": str(finetuned_checkpoint_path),
            "sha256": finetuned_checkpoint_sha256,
            "epoch": int(finetuned_checkpoint["epoch"]),
            "optimizer_step": matched_optimizer_steps,
            "planned_total_optimizer_steps": scheduler_horizon_steps,
            "best_val_chrf": finetuned_checkpoint.get("best_val_chrf"),
            "val_bleu_at_best_chrf": finetuned_checkpoint.get("val_bleu_at_best_chrf"),
            "base_checkpoint_sha256": ft_base_hash,
            "git": finetuned_checkpoint.get("git"),
        },
        "control_subset": {
            "source_file": str(train_bible_path),
            "population_pairs": len(bible_dataset),
            "subset_pairs": subset_size,
            "matched_dialogue_train_pairs": len(dialogue_dataset),
            "subset_seed": subset_seed,
            "sampling": "uniform_without_replacement_over_training_pairs",
            "selected_indices_sha256": subset_indices_sha256,
        },
        "optimization": {
            "full_model_training": True,
            "batch_size": batch_size,
            "optimizer": str(cfg["ct_optimizer"]).lower(),
            "learning_rate": float(cfg["ct_learning_rate"]),
            "adam_beta1": float(cfg["ct_adam_beta1"]),
            "adam_beta2": float(cfg["ct_adam_beta2"]),
            "adam_eps": float(cfg["ct_adam_eps"]),
            "weight_decay": float(cfg["ct_weight_decay"]),
            "label_smoothing": float(cfg["ct_label_smoothing"]),
            "grad_clip_norm": float(cfg["ct_grad_clip_norm"]),
            "scheduler": str(cfg["ct_scheduler"]).lower(),
            "scheduler_step_unit": str(cfg["ct_scheduler_step_unit"]).lower(),
            "scheduler_eta_min": float(cfg["ct_scheduler_eta_min"]),
            "scheduler_horizon_steps": scheduler_horizon_steps,
            "matched_optimizer_steps": matched_optimizer_steps,
            "completed_optimizer_steps": optimizer_steps_completed,
            "control_epochs_completed": control_epoch,
            "equivalent_full_epochs": equivalent_full_epochs,
            "remainder_steps": remainder_steps,
            "initial_learning_rate": initial_lr,
            "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        },
        "input_hashes": input_hashes,
        "git": git_metadata,
        "config": cfg,
    }
    _atomic_torch_save(checkpoint, control_checkpoint_path)

    _sync_cuda(device)
    training_loop_time_seconds = time.perf_counter() - training_start
    total_execution_time_seconds = time.perf_counter() - script_start_time

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        gpu_total_memory_mb = props.total_memory / (1024 ** 2)
        peak_gpu_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_gpu_memory_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        device_name = platform.processor() or "CPU"
        gpu_total_memory_mb = None
        peak_gpu_memory_allocated_mb = None
        peak_gpu_memory_reserved_mb = None

    summary = {
        "model_tag": model_tag,
        "experiment_type": "Continued-Biblical-Training-Control",
        "objective": "E2-continued-training-control",
        "experiment_seed": experiment_seed,
        "deterministic_training": deterministic_training,
        "training_domain": "Biblical-Control-Subset",
        "checkpoint_policy": "matched_finetuned_optimizer_step",
        "checkpoint": str(control_checkpoint_path),
        "history_file": str(history_path),
        "base_checkpoint": checkpoint["base_checkpoint"],
        "matched_finetuned_checkpoint": checkpoint["matched_finetuned_checkpoint"],
        "control_subset": checkpoint["control_subset"],
        "input_hashes": input_hashes,
        "git": git_metadata,
        "model": {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "full_model_training": True,
            "architecture": architecture,
        },
        "optimization": checkpoint["optimization"],
        "timing": {
            "setup_time_seconds": setup_time_seconds,
            "training_loop_time_seconds": training_loop_time_seconds,
            "training_loop_time_hours": _hours(training_loop_time_seconds),
            "optimization_time_seconds": total_train_seconds,
            "optimization_time_hours": _hours(total_train_seconds),
            "total_execution_time_seconds": total_execution_time_seconds,
            "total_execution_time_hours": _hours(total_execution_time_seconds),
        },
        "hardware": {
            "device_type": device.type,
            "device_name": device_name,
            "mixed_precision_amp": use_amp,
            "gpu_total_memory_mb": gpu_total_memory_mb,
            "peak_gpu_memory_allocated_mb": peak_gpu_memory_allocated_mb,
            "peak_gpu_memory_reserved_mb": peak_gpu_memory_reserved_mb,
        },
        "reproducibility": {
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "software": {
            "python_version": sys.version.split()[0],
            "pytorch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "sentencepiece_version": getattr(spm, "__version__", None),
            "sacrebleu_version": _package_version("sacrebleu"),
            "pandas_version": pd.__version__,
            "platform": platform.platform(),
        },
        "config": cfg,
    }
    _atomic_json_dump(summary, summary_path)

    print("\n" + "=" * 80)
    print("CONTINUED BIBLICAL TRAINING CONTROL COMPLETE")
    print("=" * 80)
    print(f"Matched FT epoch        : {finetuned_checkpoint['epoch']}")
    print(f"Matched optimizer step  : {matched_optimizer_steps}")
    print(f"Scheduler horizon       : {scheduler_horizon_steps}")
    print(f"Control subset pairs    : {subset_size:,}")
    print(f"Subset indices SHA-256  : {subset_indices_sha256}")
    print(f"Checkpoint              : {control_checkpoint_path}")
    print(f"History                 : {history_path}")
    print(f"Summary                 : {summary_path}")
    print(f"Total execution time    : {_hours(total_execution_time_seconds):.2f} h")
    print("=" * 80)


if __name__ == "__main__":
    run_continued_bible_control("config.yaml")
