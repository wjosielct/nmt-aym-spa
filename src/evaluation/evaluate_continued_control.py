# ==================================================
# ./src/evaluation/evaluate_continued_control.py
# ==================================================

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


def _atomic_predictions_tsv_dump(
    references: list[str],
    hypotheses: list[str],
    output_path: str | Path,
) -> None:
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Reference and hypothesis counts do not match: "
            f"{len(references)} != {len(hypotheses)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "example_index": range(len(references)),
            "reference": references,
            "hypothesis": hypotheses,
        }
    )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        frame.to_csv(
            temporary_path,
            sep="\t",
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


def _validate_evaluation_config(cfg: dict) -> None:
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
        "ct_test_max_len",
    ]
    missing = [key for key in required_keys if key not in cfg]
    if missing:
        raise KeyError(
            "Missing required continued-control evaluation configuration "
            f"parameter(s): {missing}"
        )

    if int(cfg["experiment_seed"]) < 0:
        raise ValueError("experiment_seed must be non-negative.")
    if int(cfg["ct_test_max_len"]) <= 0:
        raise ValueError("ct_test_max_len must be greater than zero.")
    if int(cfg["ct_test_max_len"]) > int(cfg["positional_encoding_max_len"]):
        raise ValueError(
            "ct_test_max_len cannot exceed positional_encoding_max_len."
        )


def _validate_checkpoint(
    checkpoint: dict,
    cfg: dict,
    expected_model_tag: str,
    tokenizer_vocab_size: int,
    tokenizer_pad_id: int,
    tokenizer_sha256: str,
) -> None:
    if checkpoint.get("checkpoint_type") != "continued_bible_control":
        raise ValueError("Expected a continued_bible_control checkpoint.")
    if checkpoint.get("model_tag") != expected_model_tag:
        raise ValueError("Control checkpoint model_tag does not match config.")
    if int(checkpoint.get("experiment_seed", -1)) != int(cfg["experiment_seed"]):
        raise ValueError("Control checkpoint seed does not match config.")
    if checkpoint.get("src") != cfg["src"] or checkpoint.get("tgt") != cfg["tgt"]:
        raise ValueError("Control checkpoint direction does not match config.")
    if int(checkpoint.get("vocab_size", -1)) != tokenizer_vocab_size:
        raise ValueError("Control checkpoint vocabulary size does not match tokenizer.")
    if int(checkpoint.get("pad_id", -1)) != tokenizer_pad_id:
        raise ValueError("Control checkpoint PAD ID does not match tokenizer.")

    stored_tokenizer_hash = checkpoint.get("input_hashes", {}).get(
        "sentencepiece_model_sha256"
    )
    if stored_tokenizer_hash and stored_tokenizer_hash != tokenizer_sha256:
        raise ValueError("Tokenizer differs from the one used by the control run.")

    expected_architecture = {
        "d_model": int(cfg["d_model"]),
        "n_heads": int(cfg["n_heads"]),
        "encoder_layers": int(cfg["encoder_layers"]),
        "decoder_layers": int(cfg["decoder_layers"]),
        "dim_feedforward": int(cfg["dim_feedforward"]),
        "dropout": float(cfg["base_dropout"]),
        "positional_encoding_max_len": int(cfg["positional_encoding_max_len"]),
    }
    stored_architecture = checkpoint.get("architecture", {})
    for key, expected_value in expected_architecture.items():
        if stored_architecture.get(key) != expected_value:
            raise ValueError(
                f"Control checkpoint architecture mismatch for {key}: "
                f"stored={stored_architecture.get(key)!r}, "
                f"current={expected_value!r}."
            )

    if checkpoint.get("checkpoint_policy") != "matched_finetuned_optimizer_step":
        raise ValueError("Unexpected continued-control checkpoint policy.")


# ==============================================================================
# FINAL TEST EVALUATION
# ==============================================================================


def evaluate_continued_control_model(
    config_path: str | Path = "config.yaml",
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Evaluates one E2 continued-training control checkpoint on both test sets."""

    script_start_time = time.perf_counter()
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    _validate_evaluation_config(cfg)

    experiment_seed = int(cfg["experiment_seed"])
    deterministic_training = bool(cfg["deterministic_training"])
    set_seed(experiment_seed, deterministic=deterministic_training)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_dir = Path(cfg["tokenizer_dir"])
    tokenized_dir = Path(cfg["tokenized_dir"])
    models_dir = Path(cfg["models_dir"])
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    model_tag = _expected_model_tag(cfg)
    if checkpoint_path is None:
        checkpoint_path = models_dir / f"continued_control_checkpoint_{model_tag}.pt"
    else:
        checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Continued-control checkpoint not found: {checkpoint_path}")

    biblical_predictions_path = (
        results_dir / f"continued_control_predictions_biblical_{model_tag}.tsv"
    )
    conversational_predictions_path = (
        results_dir / f"continued_control_predictions_conversational_{model_tag}.tsv"
    )

    sp_model_path = tokenizer_dir / "SentencePiece.model"
    if not sp_model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {sp_model_path}")

    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    pad_id = sp.pad_id()
    vocab_size = sp.vocab_size()
    if pad_id < 0:
        raise ValueError("SentencePiece PAD token is not defined.")
    tokenizer_sha256 = _sha256_file(sp_model_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Continued-control checkpoint must contain a dictionary payload.")

    _validate_checkpoint(
        checkpoint=checkpoint,
        cfg=cfg,
        expected_model_tag=model_tag,
        tokenizer_vocab_size=vocab_size,
        tokenizer_pad_id=pad_id,
        tokenizer_sha256=tokenizer_sha256,
    )

    architecture = checkpoint["architecture"]
    eval_model = TranslationTransformer(
        vocab_size=int(checkpoint["vocab_size"]),
        d_model=int(architecture["d_model"]),
        n_heads=int(architecture["n_heads"]),
        num_encoder_layers=int(architecture["encoder_layers"]),
        num_decoder_layers=int(architecture["decoder_layers"]),
        dim_feedforward=int(architecture["dim_feedforward"]),
        dropout=float(architecture["dropout"]),
        pad_id=int(checkpoint["pad_id"]),
        positional_encoding_max_len=int(architecture["positional_encoding_max_len"]),
    ).to(device)

    model_state_dict = checkpoint.pop("model_state_dict")
    eval_model.load_state_dict(model_state_dict, strict=True)
    del model_state_dict
    eval_model.eval()

    test_bible_path = tokenized_dir / "test_bible_tokenized.tsv"
    test_dialogue_path = tokenized_dir / "test_dialogue_tokenized.tsv"
    test_bible_dataset = ParallelTranslationDataset(
        tsv_path=test_bible_path,
        src_col=f"{cfg['src']}_ids",
        tgt_col=f"{cfg['tgt']}_ids",
    )
    test_dialogue_dataset = ParallelTranslationDataset(
        tsv_path=test_dialogue_path,
        src_col=f"{cfg['src']}_ids",
        tgt_col=f"{cfg['tgt']}_ids",
    )

    test_max_len = int(cfg["ct_test_max_len"])
    evaluation_hashes = {
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "sentencepiece_model_sha256": tokenizer_sha256,
        "test_bible_tokenized_sha256": _sha256_file(test_bible_path),
        "test_dialogue_tokenized_sha256": _sha256_file(test_dialogue_path),
    }
    evaluation_git = _get_git_metadata(config_path.parent)

    print("\n" + "=" * 80)
    print("CONTINUED-TRAINING CONTROL FINAL TEST EVALUATION")
    print("=" * 80)
    print(f"Model tag               : {model_tag}")
    print(f"Checkpoint              : {checkpoint_path}")
    print(f"Matched FT optimizer step: {checkpoint['optimizer_step']}")
    print(
        f"Matched FT epoch        : "
        f"{checkpoint['matched_finetuned_checkpoint']['epoch']}"
    )
    print(f"Direction               : {cfg['src']} -> {cfg['tgt']}")
    print(
        f"Architecture            : {architecture['encoder_layers']}E-"
        f"{architecture['decoder_layers']}D"
    )
    print(f"Device                  : {device.type.upper()}")
    print(f"Decoding                : greedy | max_len={test_max_len}")
    print(f"Biblical test pairs     : {len(test_bible_dataset):,}")
    print(f"Conversational test pairs: {len(test_dialogue_dataset):,}")
    print("=" * 80)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("\nEvaluating biblical test set...")
    _sync_cuda(device)
    biblical_start = time.perf_counter()
    (
        bib_bleu,
        bib_chrf,
        bib_bleu_signature,
        bib_chrf_signature,
        bib_references,
        bib_hypotheses,
    ) = evaluate_metrics(
        model=eval_model,
        dataset=test_bible_dataset,
        sp=sp,
        device=device,
        max_len=test_max_len,
        return_predictions=True,
    )
    _sync_cuda(device)
    biblical_seconds = time.perf_counter() - biblical_start

    if len(bib_references) != len(test_bible_dataset):
        raise RuntimeError("Biblical prediction count does not match test-set size.")
    _atomic_predictions_tsv_dump(
        bib_references,
        bib_hypotheses,
        biblical_predictions_path,
    )
    evaluation_hashes["biblical_predictions_sha256"] = _sha256_file(
        biblical_predictions_path
    )

    print("Evaluating conversational test set...")
    _sync_cuda(device)
    conversational_start = time.perf_counter()
    (
        conv_bleu,
        conv_chrf,
        conv_bleu_signature,
        conv_chrf_signature,
        conv_references,
        conv_hypotheses,
    ) = evaluate_metrics(
        model=eval_model,
        dataset=test_dialogue_dataset,
        sp=sp,
        device=device,
        max_len=test_max_len,
        return_predictions=True,
    )
    _sync_cuda(device)
    conversational_seconds = time.perf_counter() - conversational_start

    if len(conv_references) != len(test_dialogue_dataset):
        raise RuntimeError(
            "Conversational prediction count does not match test-set size."
        )
    _atomic_predictions_tsv_dump(
        conv_references,
        conv_hypotheses,
        conversational_predictions_path,
    )
    evaluation_hashes["conversational_predictions_sha256"] = _sha256_file(
        conversational_predictions_path
    )

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

    biblical_results = {
        "domain": "Biblical",
        "relation_to_control_training_domain": "continued-training-domain",
        "test_file": str(test_bible_path),
        "predictions_file": str(biblical_predictions_path),
        "test_pairs": len(test_bible_dataset),
        "bleu_score": bib_bleu,
        "chrf_score": bib_chrf,
        "bleu_signature": bib_bleu_signature,
        "chrf_signature": bib_chrf_signature,
        "evaluation_time_seconds": biblical_seconds,
        "evaluation_time_hours": _hours(biblical_seconds),
    }
    conversational_results = {
        "domain": "Conversational",
        "relation_to_control_training_domain": "out-of-domain",
        "test_file": str(test_dialogue_path),
        "predictions_file": str(conversational_predictions_path),
        "test_pairs": len(test_dialogue_dataset),
        "bleu_score": conv_bleu,
        "chrf_score": conv_chrf,
        "bleu_signature": conv_bleu_signature,
        "chrf_signature": conv_chrf_signature,
        "evaluation_time_seconds": conversational_seconds,
        "evaluation_time_hours": _hours(conversational_seconds),
    }

    payload = {
        "model_tag": model_tag,
        "experiment_type": "Continued-Control-Test-Evaluation",
        "objective": "E2-continued-training-control",
        "src": cfg["src"],
        "tgt": cfg["tgt"],
        "experiment_seed": experiment_seed,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": evaluation_hashes["checkpoint_sha256"],
            "checkpoint_policy": checkpoint["checkpoint_policy"],
            "optimizer_step": int(checkpoint["optimizer_step"]),
            "steps_per_epoch": int(checkpoint["steps_per_epoch"]),
            "scheduler_horizon_steps": int(checkpoint["scheduler_horizon_steps"]),
            "base_checkpoint": checkpoint["base_checkpoint"],
            "matched_finetuned_checkpoint": checkpoint["matched_finetuned_checkpoint"],
            "control_subset": checkpoint["control_subset"],
            "optimization": checkpoint["optimization"],
            "control_input_hashes": checkpoint["input_hashes"],
            "control_git": checkpoint["git"],
        },
        "model": {
            "vocab_size": int(checkpoint["vocab_size"]),
            "pad_id": int(checkpoint["pad_id"]),
            "architecture": architecture,
        },
        "evaluation": {
            "decoding_strategy": "greedy",
            "max_len": test_max_len,
            "prediction_export": {
                "format": "tsv",
                "columns": ["example_index", "reference", "hypothesis"],
            },
        },
        "biblical_results": biblical_results,
        "conversational_results": conversational_results,
        "evaluation_hashes": evaluation_hashes,
        "timing": {
            "biblical_test_seconds": biblical_seconds,
            "conversational_test_seconds": conversational_seconds,
            "total_execution_time_seconds": total_execution_time_seconds,
            "total_execution_time_hours": _hours(total_execution_time_seconds),
        },
        "hardware": {
            "device_type": device.type,
            "device_name": device_name,
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
            "sacrebleu_version": sacrebleu.__version__,
            "pandas_version": pd.__version__,
            "platform": platform.platform(),
        },
        "git": evaluation_git,
        "control_config": checkpoint["config"],
        "evaluation_config": cfg,
    }

    output_path = results_dir / f"continued_control_test_metrics_{model_tag}.json"
    _atomic_json_dump(payload, output_path)

    print("\n" + "=" * 80)
    print(f"FINAL CONTINUED-CONTROL TEST RESULTS [{model_tag}]")
    print("=" * 80)
    print(f"BIBLICAL [{len(test_bible_dataset):,} pairs]")
    print(f"  BLEU   : {bib_bleu:.2f}")
    print(f"  chrF++ : {bib_chrf:.2f}")
    print(f"CONVERSATIONAL [{len(test_dialogue_dataset):,} pairs]")
    print(f"  BLEU   : {conv_bleu:.2f}")
    print(f"  chrF++ : {conv_chrf:.2f}")
    print(f"Total execution time    : {_hours(total_execution_time_seconds):.2f} h")
    print(f"Results                 : {output_path}")
    print(f"Biblical predictions    : {biblical_predictions_path}")
    print(f"Conversational predictions: {conversational_predictions_path}")
    print("=" * 80)

    return payload


if __name__ == "__main__":
    evaluate_continued_control_model("config.yaml")
