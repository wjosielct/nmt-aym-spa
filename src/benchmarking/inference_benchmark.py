# ============================================
# ./src/benchmarking/inference_benchmark.py
# ============================================

"""Post-hoc inference benchmark for the fixed eight-layer NMT study.

The benchmark is intentionally narrow. It compares inference cost across the
three encoder-decoder depth allocations under the two prespecified
architecture-analysis conditions:

    1) base checkpoints on the biblical test set;
    2) fine-tuned checkpoints on the conversational test set.

Only three primary inference metrics are reported:

    - latency_ms_per_sentence;
    - generated_tokens_per_second;
    - peak_gpu_memory_mib.

The timed region starts from already-tokenized CPU source tensors. It includes
host-to-device transfer, encoder computation, and autoregressive greedy decoder
computation. It excludes checkpoint loading, TSV reading/parsing, SentencePiece
text decoding, BLEU/chrF++ computation, and result-file I/O.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import sentencepiece as spm
import torch

from src.data.dataset import ParallelTranslationDataset
from src.models.transformer import TranslationTransformer
from src.utils.config import load_config


_SUPPORTED_STAGES = {"base", "finetuned"}
_SUPPORTED_DOMAINS = {"biblical", "conversational"}
_FIXED_CONDITIONS = {
    ("base", "biblical"),
    ("finetuned", "conversational"),
}


# ==============================================================================
# GENERIC HELPERS
# ==============================================================================


def _sha256_file(file_path: str | Path) -> str:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_jsonable(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(file_path: str | Path) -> dict[str, Any]:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"JSON artifact not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must contain an object: {file_path}")
    return payload


def _atomic_json_dump(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=4)
            handle.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_csv_dump(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        dataframe.to_csv(temporary_path, index=False, encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _get_git_metadata(repository_dir: str | Path) -> dict[str, Any]:
    repository_dir = Path(repository_dir).resolve()
    metadata: dict[str, Any] = {
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


def _software_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "sentencepiece": getattr(spm, "__version__", None),
        "pandas": pd.__version__,
    }


def _cuda_metadata(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "device_index": int(device.index if device.index is not None else 0),
        "gpu_name": properties.name,
        "total_memory_mib": float(properties.total_memory / (1024**2)),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _architecture_label(encoder_layers: int, decoder_layers: int) -> str:
    return f"{int(encoder_layers)}E-{int(decoder_layers)}D"


def model_tag(
    src: str,
    tgt: str,
    encoder_layers: int,
    decoder_layers: int,
    seed: int,
) -> str:
    """Builds the exact model tag used throughout the training pipeline."""
    return (
        f"{src}_to_{tgt}"
        f"_Enc{int(encoder_layers)}"
        f"_Dec{int(decoder_layers)}"
        f"_Seed{int(seed)}"
    )


# ==============================================================================
# CONFIGURATION / EXPERIMENT PLAN
# ==============================================================================


def read_benchmark_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    """Reads and strictly validates the fixed inference-benchmark protocol."""

    if "inference_benchmark" not in cfg:
        raise KeyError("config.yaml is missing the 'inference_benchmark' section.")

    raw = cfg["inference_benchmark"]
    if not isinstance(raw, dict):
        raise TypeError("inference_benchmark must be a YAML mapping.")

    required = {
        "conditions",
        "batch_size",
        "warmup_examples",
        "repetitions",
        "max_len",
        "precision",
        "cuda_device_index",
        "require_cuda",
        "output_subdir",
        "output_prefix",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise KeyError(f"Missing inference benchmark setting(s): {missing}")

    conditions_raw = raw["conditions"]
    if not isinstance(conditions_raw, list) or not conditions_raw:
        raise ValueError("inference_benchmark.conditions must be a non-empty list.")

    conditions: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in conditions_raw:
        if not isinstance(item, dict):
            raise TypeError("Each inference benchmark condition must be a mapping.")
        stage = str(item.get("stage", "")).strip().lower()
        domain = str(item.get("domain", "")).strip().lower()
        if stage not in _SUPPORTED_STAGES:
            raise ValueError(f"Unsupported benchmark stage: {stage!r}")
        if domain not in _SUPPORTED_DOMAINS:
            raise ValueError(f"Unsupported benchmark domain: {domain!r}")
        pair = (stage, domain)
        if pair in seen:
            raise ValueError(f"Duplicate benchmark condition: {pair}")
        seen.add(pair)
        conditions.append({"stage": stage, "domain": domain})

    if seen != _FIXED_CONDITIONS:
        raise ValueError(
            "The definitive inference benchmark must contain exactly "
            "Base/Biblical and Fine-tuned/Conversational conditions."
        )

    batch_size = int(raw["batch_size"])
    if batch_size != 1:
        raise ValueError("The definitive inference benchmark requires batch_size=1.")

    warmup_examples = int(raw["warmup_examples"])
    if warmup_examples < 0:
        raise ValueError("warmup_examples cannot be negative.")

    repetitions = int(raw["repetitions"])
    if repetitions != 1:
        raise ValueError("The definitive protocol uses exactly one measured full-test pass.")

    max_len = int(raw["max_len"])
    if max_len <= 0:
        raise ValueError("inference_benchmark.max_len must be greater than zero.")

    if max_len != int(cfg["base_test_max_len"]) or max_len != int(cfg["ft_test_max_len"]):
        raise ValueError(
            "Inference benchmark max_len must match both base_test_max_len and "
            "ft_test_max_len."
        )

    precision = str(raw["precision"]).strip().lower()
    if precision != "fp32":
        raise ValueError("The definitive inference benchmark uses FP32 only.")

    cuda_device_index = int(raw["cuda_device_index"])
    if cuda_device_index < 0:
        raise ValueError("cuda_device_index cannot be negative.")

    require_cuda = bool(raw["require_cuda"])
    if not require_cuda:
        raise ValueError("The definitive inference benchmark requires CUDA.")

    output_subdir = str(raw["output_subdir"]).strip()
    output_prefix = str(raw["output_prefix"]).strip()
    if not output_subdir or not output_prefix:
        raise ValueError("Benchmark output_subdir/output_prefix cannot be empty.")

    return {
        "conditions": conditions,
        "batch_size": batch_size,
        "warmup_examples": warmup_examples,
        "repetitions": repetitions,
        "max_len": max_len,
        "precision": precision,
        "cuda_device_index": cuda_device_index,
        "require_cuda": require_cuda,
        "output_subdir": output_subdir,
        "output_prefix": output_prefix,
        "timing_scope": (
            "pretokenized_cpu_source_tensor_to_generated_token_ids; includes "
            "CPU-to-GPU source transfer, encoder, and autoregressive greedy decoder; "
            "excludes checkpoint loading, TSV parsing, detokenization, metrics, and I/O"
        ),
        "generated_token_count_policy": (
            "exclude BOS and PAD control IDs; include EOS when generated"
        ),
        "primary_metrics": [
            "latency_ms_per_sentence",
            "generated_tokens_per_second",
            "peak_gpu_memory_mib",
        ],
        "run_order_policy": (
            "within each stage/direction, rotate architecture order across the "
            "three model seeds so each architecture occupies each run-order "
            "position once"
        ),
    }


def _load_grid(
    cfg: dict[str, Any],
) -> tuple[list[int], list[dict[str, str]], list[tuple[int, int]]]:
    seeds = [int(seed) for seed in cfg["experiment_seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("experiment_seeds must contain unique values.")

    directions: list[dict[str, str]] = []
    for item in cfg["experiment_directions"]:
        src = str(item["src"]).strip()
        tgt = str(item["tgt"]).strip()
        if not src or not tgt or src == tgt:
            raise ValueError(f"Invalid experiment direction: {item!r}")
        directions.append({"src": src, "tgt": tgt})
    if len({(item["src"], item["tgt"]) for item in directions}) != len(directions):
        raise ValueError("experiment_directions contains duplicates.")

    architectures: list[tuple[int, int]] = []
    for item in cfg["experiment_architectures"]:
        enc = int(item["encoder_layers"])
        dec = int(item["decoder_layers"])
        if enc <= 0 or dec <= 0 or enc + dec != 8:
            raise ValueError("Every benchmark architecture must preserve the eight-layer budget.")
        architectures.append((enc, dec))
    if len(architectures) != len(set(architectures)):
        raise ValueError("experiment_architectures contains duplicates.")

    return seeds, directions, architectures


def _build_plan(cfg: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    seeds, directions, architectures = _load_grid(cfg)
    plan: list[dict[str, Any]] = []

    for condition in protocol["conditions"]:
        for direction in directions:
            for seed_index, seed in enumerate(seeds):
                # Deterministic rotation balances first/middle/last benchmark
                # position across the three architectures and three seeds.
                offset = seed_index % len(architectures)
                ordered_architectures = architectures[offset:] + architectures[:offset]
                for enc, dec in ordered_architectures:
                    plan.append(
                        {
                            "stage": condition["stage"],
                            "domain": condition["domain"],
                            "src": direction["src"],
                            "tgt": direction["tgt"],
                            "encoder_layers": enc,
                            "decoder_layers": dec,
                            "architecture": _architecture_label(enc, dec),
                            "seed": int(seed),
                            "model_tag": model_tag(
                                direction["src"],
                                direction["tgt"],
                                enc,
                                dec,
                                seed,
                            ),
                        }
                    )

    return plan


# ==============================================================================
# ARTIFACT RESOLUTION / CHECKPOINT VALIDATION
# ==============================================================================


def _checkpoint_path(project_root: Path, cfg: dict[str, Any], item: dict[str, Any]) -> Path:
    models_dir = project_root / str(cfg["models_dir"])
    prefix = "base_checkpoint" if item["stage"] == "base" else "finetuned_checkpoint"
    return models_dir / f"{prefix}_{item['model_tag']}.pt"


def _test_path(project_root: Path, cfg: dict[str, Any], domain: str) -> Path:
    tokenized_dir = project_root / str(cfg["tokenized_dir"])
    if domain == "biblical":
        return tokenized_dir / "test_bible_tokenized.tsv"
    if domain == "conversational":
        return tokenized_dir / "test_dialogue_tokenized.tsv"
    raise ValueError(f"Unsupported domain: {domain}")


def _metrics_path(project_root: Path, cfg: dict[str, Any], item: dict[str, Any]) -> Path:
    results_dir = project_root / str(cfg["results_dir"])
    prefix = "base_test_metrics" if item["stage"] == "base" else "finetuned_test_metrics"
    return results_dir / f"{prefix}_{item['model_tag']}.json"


def _checkpoint_tokenizer_hash(checkpoint: dict[str, Any], stage: str) -> str | None:
    if stage == "base":
        return checkpoint.get("input_hashes", {}).get("sentencepiece_model_sha256")
    if stage == "finetuned":
        return checkpoint.get("fine_tuning_input_hashes", {}).get(
            "sentencepiece_model_sha256"
        )
    raise ValueError(f"Unsupported checkpoint stage: {stage}")


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    item: dict[str, Any],
    cfg: dict[str, Any],
    tokenizer_vocab_size: int,
    tokenizer_pad_id: int,
    tokenizer_sha256: str,
) -> None:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint for {item['model_tag']} must be a dictionary payload.")

    required = {
        "checkpoint_type",
        "model_tag",
        "experiment_seed",
        "model_state_dict",
        "vocab_size",
        "pad_id",
        "src",
        "tgt",
        "architecture",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise KeyError(f"Checkpoint {item['model_tag']} is missing field(s): {missing}")

    expected_type = "base_training" if item["stage"] == "base" else "fine_tuning"
    if checkpoint["checkpoint_type"] != expected_type:
        raise ValueError(
            f"Checkpoint type mismatch for {item['model_tag']}: "
            f"{checkpoint['checkpoint_type']!r} != {expected_type!r}"
        )

    scalar_checks = {
        "model_tag": (checkpoint["model_tag"], item["model_tag"]),
        "experiment_seed": (int(checkpoint["experiment_seed"]), int(item["seed"])),
        "src": (str(checkpoint["src"]), str(item["src"])),
        "tgt": (str(checkpoint["tgt"]), str(item["tgt"])),
        "vocab_size": (int(checkpoint["vocab_size"]), int(tokenizer_vocab_size)),
        "pad_id": (int(checkpoint["pad_id"]), int(tokenizer_pad_id)),
    }
    for label, (stored, expected) in scalar_checks.items():
        if stored != expected:
            raise ValueError(
                f"Checkpoint {item['model_tag']} mismatch for {label}: "
                f"{stored!r} != {expected!r}"
            )

    checkpoint_tokenizer_sha = _checkpoint_tokenizer_hash(checkpoint, item["stage"])
    if checkpoint_tokenizer_sha is not None and checkpoint_tokenizer_sha != tokenizer_sha256:
        raise ValueError(
            f"Checkpoint {item['model_tag']} was trained with a different tokenizer."
        )

    architecture = checkpoint["architecture"]
    expected_architecture = {
        "d_model": int(cfg["d_model"]),
        "n_heads": int(cfg["n_heads"]),
        "encoder_layers": int(item["encoder_layers"]),
        "decoder_layers": int(item["decoder_layers"]),
        "dim_feedforward": int(cfg["dim_feedforward"]),
        "dropout": float(cfg["base_dropout"]),
        "positional_encoding_max_len": int(cfg["positional_encoding_max_len"]),
    }
    for key, expected in expected_architecture.items():
        if key not in architecture:
            raise KeyError(f"Checkpoint architecture is missing field {key!r}.")
        if architecture[key] != expected:
            raise ValueError(
                f"Checkpoint {item['model_tag']} architecture mismatch for {key}: "
                f"{architecture[key]!r} != {expected!r}"
            )


def _build_model(checkpoint: dict[str, Any], device: torch.device) -> TranslationTransformer:
    architecture = checkpoint["architecture"]
    model = TranslationTransformer(
        vocab_size=int(checkpoint["vocab_size"]),
        d_model=int(architecture["d_model"]),
        n_heads=int(architecture["n_heads"]),
        num_encoder_layers=int(architecture["encoder_layers"]),
        num_decoder_layers=int(architecture["decoder_layers"]),
        dim_feedforward=int(architecture["dim_feedforward"]),
        dropout=float(architecture["dropout"]),
        pad_id=int(checkpoint["pad_id"]),
        positional_encoding_max_len=int(architecture["positional_encoding_max_len"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def _count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _load_sources(
    test_path: Path,
    src: str,
    tgt: str,
) -> list[torch.Tensor]:
    dataset = ParallelTranslationDataset(
        tsv_path=test_path,
        src_col=f"{src}_ids",
        tgt_col=f"{tgt}_ids",
    )
    return [dataset[index][0] for index in range(len(dataset))]


# ==============================================================================
# BENCHMARK CORE
# ==============================================================================


def count_generated_tokens(
    token_ids: list[int],
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> int:
    """Counts generated non-BOS/non-PAD IDs and includes EOS when present."""

    count = 0
    for index, token_id in enumerate(token_ids):
        token_id = int(token_id)
        if index == 0 and token_id == bos_id:
            continue
        if token_id == bos_id or token_id == pad_id:
            continue
        count += 1
        if token_id == eos_id:
            break
    return count


def _benchmark_loaded_model(
    model: TranslationTransformer,
    sources: list[torch.Tensor],
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    max_len: int,
    warmup_examples: int,
) -> dict[str, Any]:
    if not sources:
        raise ValueError("Inference benchmark received an empty test set.")
    if device.type != "cuda":
        raise ValueError("Peak-GPU-memory benchmarking requires a CUDA device.")

    # Lazy import keeps the protocol/aggregation helpers independently testable
    # without loading SacreBLEU until actual generation is requested.
    from src.utils.metrics import greedy_decode

    model.eval()
    bos_id = int(sp.bos_id())
    eos_id = int(sp.eos_id())
    pad_id = int(sp.pad_id())
    if min(bos_id, eos_id, pad_id) < 0:
        raise ValueError("SentencePiece BOS/EOS/PAD IDs must all be defined.")

    n_warmup = min(int(warmup_examples), len(sources))

    with torch.inference_mode():
        for source in sources[:n_warmup]:
            greedy_decode(
                model=model,
                src=source,
                sp=sp,
                max_len=max_len,
                device=device,
            )

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        generated_tokens = 0
        start = time.perf_counter()

        for source in sources:
            prediction_ids = greedy_decode(
                model=model,
                src=source,
                sp=sp,
                max_len=max_len,
                device=device,
            )
            generated_tokens += count_generated_tokens(
                token_ids=prediction_ids,
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
            )

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak_memory_mib = float(torch.cuda.max_memory_allocated(device) / (1024**2))

    if elapsed <= 0.0:
        raise RuntimeError("Measured inference time must be positive.")
    if generated_tokens <= 0:
        raise RuntimeError("No generated tokens were counted during benchmarking.")

    n_sentences = len(sources)
    return {
        "n_sentences": int(n_sentences),
        "n_generated_tokens": int(generated_tokens),
        "generation_seconds": float(elapsed),
        "latency_ms_per_sentence": float(1000.0 * elapsed / n_sentences),
        "generated_tokens_per_second": float(generated_tokens / elapsed),
        "peak_gpu_memory_mib": peak_memory_mib,
    }


def summarize_runs(
    rows: list[dict[str, Any]],
    expected_seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Aggregates the three primary benchmark metrics as mean ± sample SD."""

    if not rows:
        return []

    group_fields = [
        "stage",
        "domain",
        "direction",
        "architecture",
        "encoder_layers",
        "decoder_layers",
        "n_parameters",
        "n_sentences",
        "gpu_name",
    ]
    metric_fields = [
        "latency_ms_per_sentence",
        "generated_tokens_per_second",
        "peak_gpu_memory_mib",
    ]

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in grouped.items():
        base = dict(zip(group_fields, key))
        seeds = sorted(int(row["seed"]) for row in group)
        if expected_seeds is not None and seeds != sorted(int(seed) for seed in expected_seeds):
            raise ValueError(
                f"Benchmark group {base['stage']}/{base['direction']}/{base['architecture']} "
                f"has seeds {seeds}, expected {sorted(expected_seeds)}."
            )

        summary: dict[str, Any] = {
            **base,
            "n_seeds": len(group),
            "seeds": ";".join(str(seed) for seed in seeds),
        }
        for field in metric_fields:
            values = [float(row[field]) for row in group]
            summary[f"{field}_mean"] = float(statistics.fmean(values))
            summary[f"{field}_sd"] = (
                float(statistics.stdev(values)) if len(values) >= 2 else 0.0
            )
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            row["stage"],
            row["direction"],
            row["encoder_layers"],
            row["decoder_layers"],
        )
    )
    return summaries


# ==============================================================================
# PRE-FLIGHT / SIGNATURE
# ==============================================================================


def _preflight(
    project_root: Path,
    config_path: Path,
    cfg: dict[str, Any],
    plan: list[dict[str, Any]],
) -> tuple[dict[str, Any], spm.SentencePieceProcessor]:
    tokenizer_path = project_root / str(cfg["tokenizer_dir"]) / "SentencePiece.model"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"SentencePiece model not found: {tokenizer_path}")

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if min(sp.pad_id(), sp.bos_id(), sp.eos_id()) < 0:
        raise ValueError("SentencePiece PAD/BOS/EOS IDs must all be defined.")

    tokenizer_sha = _sha256_file(tokenizer_path)
    test_files: dict[str, dict[str, Any]] = {}
    for domain in sorted(_SUPPORTED_DOMAINS):
        path = _test_path(project_root, cfg, domain)
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark test file not found: {path}")
        test_files[domain] = {
            "path": os.path.relpath(path, project_root),
            "sha256": _sha256_file(path),
        }

    checkpoint_manifest: list[dict[str, Any]] = []
    for item in plan:
        checkpoint_path = _checkpoint_path(project_root, cfg, item)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Benchmark checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _validate_checkpoint(
            checkpoint=checkpoint,
            item=item,
            cfg=cfg,
            tokenizer_vocab_size=int(sp.vocab_size()),
            tokenizer_pad_id=int(sp.pad_id()),
            tokenizer_sha256=tokenizer_sha,
        )

        checkpoint_sha = _sha256_file(checkpoint_path)
        metrics_path = _metrics_path(project_root, cfg, item)
        metrics_payload = _load_json(metrics_path)
        if metrics_payload.get("model_tag") != item["model_tag"]:
            raise ValueError(
                f"Evaluation JSON model_tag mismatch: {metrics_path}"
            )
        recorded_checkpoint_sha = (
            metrics_payload.get("evaluation_hashes", {}).get("checkpoint_sha256")
        )
        if recorded_checkpoint_sha != checkpoint_sha:
            raise ValueError(
                f"Checkpoint {checkpoint_path.name} does not match the exact "
                f"checkpoint hash recorded by {metrics_path.name}."
            )
        test_hash_key = (
            "test_bible_tokenized_sha256"
            if item["domain"] == "biblical"
            else "test_dialogue_tokenized_sha256"
        )
        recorded_test_sha = metrics_payload.get("evaluation_hashes", {}).get(test_hash_key)
        if recorded_test_sha != test_files[item["domain"]]["sha256"]:
            raise ValueError(
                f"Test-set hash in {metrics_path.name} does not match the current "
                f"{item['domain']} tokenized test file."
            )

        checkpoint_manifest.append(
            {
                "stage": item["stage"],
                "domain": item["domain"],
                "src": item["src"],
                "tgt": item["tgt"],
                "architecture": item["architecture"],
                "seed": item["seed"],
                "model_tag": item["model_tag"],
                "checkpoint_file": os.path.relpath(checkpoint_path, project_root),
                "checkpoint_sha256": checkpoint_sha,
                "evaluation_metrics_file": os.path.relpath(metrics_path, project_root),
                "evaluation_metrics_sha256": _sha256_file(metrics_path),
            }
        )
        del checkpoint

    manifest = {
        "config_path": os.path.relpath(config_path, project_root),
        "config_sha256": _sha256_file(config_path),
        "tokenizer": {
            "path": os.path.relpath(tokenizer_path, project_root),
            "sha256": tokenizer_sha,
            "vocab_size": int(sp.vocab_size()),
            "pad_id": int(sp.pad_id()),
            "bos_id": int(sp.bos_id()),
            "eos_id": int(sp.eos_id()),
        },
        "test_files": test_files,
        "checkpoints": checkpoint_manifest,
    }
    return manifest, sp


def _analysis_signature_payload(
    protocol: dict[str, Any],
    plan: list[dict[str, Any]],
    manifest: dict[str, Any],
    git_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "plan": plan,
        "input_artifacts": manifest,
        "git": {
            "commit": git_metadata.get("commit"),
            "exact_tag": git_metadata.get("exact_tag"),
            "dirty_worktree": git_metadata.get("dirty_worktree"),
        },
    }


def _existing_outputs_are_compatible(
    json_path: Path,
    runs_csv_path: Path,
    summary_csv_path: Path,
    signature_sha256: str,
) -> bool:
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    if payload.get("analysis_signature_sha256") != signature_sha256:
        return False

    hashes = payload.get("output_hashes", {})
    return (
        hashes.get("runs_csv_sha256") == _sha256_file(runs_csv_path)
        and hashes.get("summary_csv_sha256") == _sha256_file(summary_csv_path)
    )


# ==============================================================================
# PUBLIC ENTRY POINT
# ==============================================================================


def run_inference_benchmark(
    config_path: str | Path = "config.yaml",
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Runs the complete, prespecified post-hoc inference benchmark."""

    config_path = Path(config_path).resolve()
    project_root = config_path.parent
    cfg = load_config(config_path)
    protocol = read_benchmark_protocol(cfg)
    plan = _build_plan(cfg, protocol)
    expected_seeds, _, _ = _load_grid(cfg)

    git_metadata = _get_git_metadata(project_root)
    manifest, sp = _preflight(
        project_root=project_root,
        config_path=config_path,
        cfg=cfg,
        plan=plan,
    )

    signature_payload = _analysis_signature_payload(
        protocol=protocol,
        plan=plan,
        manifest=manifest,
        git_metadata=git_metadata,
    )
    signature_sha256 = _sha256_jsonable(signature_payload)

    output_dir = project_root / str(cfg["results_dir"]) / protocol["output_subdir"]
    prefix = protocol["output_prefix"]
    runs_csv_path = output_dir / f"{prefix}_runs.csv"
    summary_csv_path = output_dir / f"{prefix}_summary.csv"
    summary_json_path = output_dir / f"{prefix}_summary.json"

    outputs = [runs_csv_path, summary_csv_path, summary_json_path]
    existing = [path for path in outputs if path.exists()]
    if existing and len(existing) != len(outputs) and not force:
        raise RuntimeError(
            "Partial inference-benchmark outputs already exist. Inspect/remove them or "
            "use --force only for intentional regeneration."
        )
    if len(existing) == len(outputs) and not force:
        if _existing_outputs_are_compatible(
            summary_json_path,
            runs_csv_path,
            summary_csv_path,
            signature_sha256,
        ):
            print("[SKIP] Compatible inference-benchmark outputs already exist.")
            with summary_json_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        raise RuntimeError(
            "Complete inference-benchmark outputs exist but are incompatible with "
            "the current code/config/checkpoint hashes. Use --force only if the "
            "regeneration is intentional."
        )

    print("\n" + "=" * 80)
    print("INFERENCE BENCHMARK — FIXED EIGHT-LAYER DEPTH ALLOCATIONS")
    print("=" * 80)
    print(f"Model seeds             : {expected_seeds}")
    print(f"Planned benchmark runs  : {len(plan)}")
    print(f"Conditions              : Base/Biblical; Fine-tuned/Conversational")
    print(f"Batch size              : {protocol['batch_size']}")
    print(f"Warm-up examples        : {protocol['warmup_examples']}")
    print(f"Measured repetitions    : {protocol['repetitions']}")
    print(f"Greedy max_len          : {protocol['max_len']}")
    print(f"Precision               : {protocol['precision'].upper()}")
    print(f"CUDA device index       : {protocol['cuda_device_index']}")
    print(f"Output runs CSV         : {runs_csv_path}")
    print(f"Output summary CSV      : {summary_csv_path}")
    print(f"Output summary JSON     : {summary_json_path}")
    print(f"Force                   : {force}")
    print(f"Dry run                 : {dry_run}")
    print("=" * 80)

    if dry_run:
        for index, item in enumerate(plan, start=1):
            print(
                f"[DRY RUN] {index:02d}/{len(plan):02d} "
                f"{item['stage']}|{item['domain']}|{item['model_tag']}"
            )
        return None

    if protocol["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by the definitive inference benchmark, but "
            "torch.cuda.is_available() is False."
        )

    device_index = int(protocol["cuda_device_index"])
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested CUDA device index {device_index}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible."
        )

    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    cuda_metadata = _cuda_metadata(device)
    print(f"GPU                     : {cuda_metadata['gpu_name']}")

    # Pretokenized source tensors are materialized once per domain/direction.
    # TSV parsing is intentionally outside the timed region.
    source_cache: dict[tuple[str, str, str], list[torch.Tensor]] = {}
    for item in plan:
        cache_key = (item["domain"], item["src"], item["tgt"])
        if cache_key not in source_cache:
            source_cache[cache_key] = _load_sources(
                test_path=_test_path(project_root, cfg, item["domain"]),
                src=item["src"],
                tgt=item["tgt"],
            )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []

    checkpoint_hash_by_tag = {
        (entry["stage"], entry["model_tag"]): entry["checkpoint_sha256"]
        for entry in manifest["checkpoints"]
    }

    for index, item in enumerate(plan, start=1):
        checkpoint_path = _checkpoint_path(project_root, cfg, item)
        test_path = _test_path(project_root, cfg, item["domain"])
        sources = source_cache[(item["domain"], item["src"], item["tgt"])]

        print("\n" + "-" * 80)
        print(f"BENCHMARK {index}/{len(plan)}: {item['stage']} | {item['domain']} | {item['model_tag']}")
        print("-" * 80)
        print(f"Checkpoint              : {checkpoint_path}")
        print(f"Test set                : {test_path}")
        print(f"Sentences               : {len(sources)}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _validate_checkpoint(
            checkpoint=checkpoint,
            item=item,
            cfg=cfg,
            tokenizer_vocab_size=int(sp.vocab_size()),
            tokenizer_pad_id=int(sp.pad_id()),
            tokenizer_sha256=manifest["tokenizer"]["sha256"],
        )

        model = _build_model(checkpoint, device)
        # model.parameters() correctly counts tied embedding/output weights once.
        n_parameters = _count_parameters(model)

        measurement = _benchmark_loaded_model(
            model=model,
            sources=sources,
            sp=sp,
            device=device,
            max_len=int(protocol["max_len"]),
            warmup_examples=int(protocol["warmup_examples"]),
        )

        row = {
            "stage": item["stage"],
            "domain": item["domain"],
            "direction": f"{item['src']}->{item['tgt']}",
            "src": item["src"],
            "tgt": item["tgt"],
            "architecture": item["architecture"],
            "encoder_layers": int(item["encoder_layers"]),
            "decoder_layers": int(item["decoder_layers"]),
            "seed": int(item["seed"]),
            "model_tag": item["model_tag"],
            "n_parameters": int(n_parameters),
            **measurement,
            "checkpoint_sha256": checkpoint_hash_by_tag[(item["stage"], item["model_tag"])],
            "test_tokenized_sha256": manifest["test_files"][item["domain"]]["sha256"],
            "gpu_name": cuda_metadata["gpu_name"],
        }
        rows.append(row)

        print(f"Latency                : {row['latency_ms_per_sentence']:.3f} ms/sentence")
        print(f"Generated throughput    : {row['generated_tokens_per_second']:.3f} tokens/s")
        print(f"Peak allocated GPU mem  : {row['peak_gpu_memory_mib']:.1f} MiB")
        print(f"Generated tokens        : {row['n_generated_tokens']}")
        print(f"Timed generation        : {row['generation_seconds']:.2f} s")

        del model
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    summaries = summarize_runs(rows, expected_seeds=expected_seeds)

    runs_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summaries)
    _atomic_csv_dump(runs_df, runs_csv_path)
    _atomic_csv_dump(summary_df, summary_csv_path)

    output_hashes = {
        "runs_csv_sha256": _sha256_file(runs_csv_path),
        "summary_csv_sha256": _sha256_file(summary_csv_path),
    }

    payload: dict[str, Any] = {
        "analysis_type": "Inference-Benchmark",
        "objective": (
            "post-hoc inference-cost characterization of fixed eight-layer "
            "encoder-decoder depth allocations"
        ),
        "analysis_signature_sha256": signature_sha256,
        "protocol": protocol,
        "grid": {
            "model_seeds": expected_seeds,
            "n_benchmark_runs": len(plan),
            "n_summary_groups": len(summaries),
        },
        "input_artifacts": manifest,
        "run_results": rows,
        "aggregate_results": summaries,
        "output_files": {
            "runs_csv": os.path.relpath(runs_csv_path, project_root),
            "summary_csv": os.path.relpath(summary_csv_path, project_root),
            "summary_json": os.path.relpath(summary_json_path, project_root),
        },
        "output_hashes": output_hashes,
        "timing": {
            "benchmark_wall_seconds": float(elapsed),
            "benchmark_wall_hours": float(elapsed / 3600.0),
        },
        "hardware": cuda_metadata,
        "software": _software_metadata(),
        "git": git_metadata,
        "config": cfg,
    }
    _atomic_json_dump(payload, summary_json_path)

    print("\n" + "=" * 80)
    print("INFERENCE BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Benchmark runs          : {len(rows)}")
    print(f"Summary groups          : {len(summaries)}")
    print(f"Runs CSV                : {runs_csv_path}")
    print(f"Summary CSV             : {summary_csv_path}")
    print(f"Summary JSON            : {summary_json_path}")
    print(f"Execution time          : {elapsed / 3600.0:.2f} h")
    print("=" * 80)

    return payload
