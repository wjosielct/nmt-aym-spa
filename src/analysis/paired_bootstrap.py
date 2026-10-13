# ======================================
# ./src/analysis/paired_bootstrap.py
# ======================================

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sacrebleu

from src.utils.config import load_config
from src.utils.metrics import _build_bleu_metric, _build_chrf_metric


# ==============================================================================
# CONSTANTS
# ==============================================================================

_STAGE_PREFIX = {
    "base": "base_predictions",
    "finetuned": "finetuned_predictions",
    "continued_control": "continued_control_predictions",
}

_STAGE_METRICS_PREFIX = {
    "base": "base_test_metrics",
    "finetuned": "finetuned_test_metrics",
    "continued_control": "continued_control_test_metrics",
}

_SUPPORTED_DOMAINS = ("biblical", "conversational")
_SUPPORTED_METRICS = ("bleu", "chrf++")


# ==============================================================================
# BASIC UTILITIES
# ==============================================================================


def _model_tag(
    src: str,
    tgt: str,
    encoder_layers: int,
    decoder_layers: int,
    seed: int,
) -> str:
    return (
        f"{src}_to_{tgt}"
        f"_Enc{int(encoder_layers)}"
        f"_Dec{int(decoder_layers)}"
        f"_Seed{int(seed)}"
    )


def _architecture_label(encoder_layers: int, decoder_layers: int) -> str:
    return f"{int(encoder_layers)}E-{int(decoder_layers)}D"


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _sha256_file(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return payload


def _atomic_json_dump(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4, ensure_ascii=False)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_csv_dump(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        dataframe.to_csv(temporary_path, index=False, encoding="utf-8")
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _float_close(a: float, b: float, tolerance: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=float(tolerance))


# ==============================================================================
# CONFIGURATION / PLAN
# ==============================================================================


def _validate_architecture(item: dict[str, Any]) -> tuple[int, int]:
    enc = int(item["encoder_layers"])
    dec = int(item["decoder_layers"])
    if enc <= 0 or dec <= 0 or enc + dec != 8:
        raise ValueError(
            "Every bootstrap architecture must preserve the fixed eight-layer budget."
        )
    return enc, dec


def _read_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    required = (
        "bootstrap_replicates",
        "bootstrap_confidence_level",
        "bootstrap_seed",
        "bootstrap_expected_sacrebleu_version",
        "bootstrap_metric_validation_tolerance",
        "bootstrap_output_subdir",
        "bootstrap_output_prefix",
        "bootstrap_resampling_units",
        "bootstrap_adaptation_domains",
        "bootstrap_enable_adaptation_contrasts",
        "bootstrap_enable_architecture_contrasts",
        "bootstrap_architecture_reference",
        "bootstrap_architecture_comparators",
        "bootstrap_architecture_conditions",
    )
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"Missing E8 bootstrap configuration parameter(s): {missing}")

    replicates = int(cfg["bootstrap_replicates"])
    if replicates < 1000:
        # Lower replicate counts remain permitted for explicitly overridden
        # validation configurations.
        pass
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be greater than zero.")

    confidence = float(cfg["bootstrap_confidence_level"])
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap_confidence_level must be between 0 and 1.")

    units = cfg["bootstrap_resampling_units"]
    if not isinstance(units, dict):
        raise TypeError("bootstrap_resampling_units must be a mapping.")
    for domain in _SUPPORTED_DOMAINS:
        if domain not in units or not str(units[domain]).strip():
            raise ValueError(
                f"bootstrap_resampling_units must define a non-empty unit for {domain}."
            )

    adaptation_domains = [str(x).strip() for x in cfg["bootstrap_adaptation_domains"]]
    if not adaptation_domains:
        raise ValueError("bootstrap_adaptation_domains cannot be empty.")
    invalid_domains = [x for x in adaptation_domains if x not in _SUPPORTED_DOMAINS]
    if invalid_domains:
        raise ValueError(f"Unsupported bootstrap domain(s): {invalid_domains}")

    ref_enc, ref_dec = _validate_architecture(cfg["bootstrap_architecture_reference"])
    comparators = []
    for item in cfg["bootstrap_architecture_comparators"]:
        enc, dec = _validate_architecture(item)
        if (enc, dec) == (ref_enc, ref_dec):
            raise ValueError("Architecture comparator cannot equal the reference architecture.")
        comparators.append((enc, dec))
    if len(comparators) != len(set(comparators)):
        raise ValueError("bootstrap_architecture_comparators contains duplicates.")

    conditions = []
    for item in cfg["bootstrap_architecture_conditions"]:
        stage = str(item["stage"]).strip()
        domain = str(item["domain"]).strip()
        if stage not in ("base", "finetuned", "continued_control"):
            raise ValueError(f"Unsupported architecture-comparison stage: {stage}")
        if domain not in _SUPPORTED_DOMAINS:
            raise ValueError(f"Unsupported architecture-comparison domain: {domain}")
        conditions.append({"stage": stage, "domain": domain})

    return {
        "replicates": replicates,
        "confidence_level": confidence,
        "seed": int(cfg["bootstrap_seed"]),
        "expected_sacrebleu_version": str(cfg["bootstrap_expected_sacrebleu_version"]),
        "metric_validation_tolerance": float(cfg["bootstrap_metric_validation_tolerance"]),
        "output_subdir": str(cfg["bootstrap_output_subdir"]),
        "output_prefix": str(cfg["bootstrap_output_prefix"]),
        "resampling_units": {k: str(v) for k, v in units.items()},
        "adaptation_domains": adaptation_domains,
        "enable_adaptation": bool(cfg["bootstrap_enable_adaptation_contrasts"]),
        "enable_architecture": bool(cfg["bootstrap_enable_architecture_contrasts"]),
        "architecture_reference": (ref_enc, ref_dec),
        "architecture_comparators": comparators,
        "architecture_conditions": conditions,
    }


def _load_grid(cfg: dict[str, Any]) -> tuple[list[int], list[dict[str, str]], list[tuple[int, int]]]:
    seeds = [int(x) for x in cfg["experiment_seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("experiment_seeds must contain unique values.")

    directions = []
    for item in cfg["experiment_directions"]:
        src = str(item["src"]).strip()
        tgt = str(item["tgt"]).strip()
        if not src or not tgt or src == tgt:
            raise ValueError("Invalid experiment direction.")
        directions.append({"src": src, "tgt": tgt})
    if len({(x["src"], x["tgt"]) for x in directions}) != len(directions):
        raise ValueError("experiment_directions contains duplicates.")

    architectures = [_validate_architecture(x) for x in cfg["experiment_architectures"]]
    if len(architectures) != len(set(architectures)):
        raise ValueError("experiment_architectures contains duplicates.")

    return seeds, directions, architectures


def _system_spec(stage: str, architecture: tuple[int, int]) -> dict[str, Any]:
    enc, dec = architecture
    return {
        "stage": stage,
        "encoder_layers": int(enc),
        "decoder_layers": int(dec),
        "architecture": _architecture_label(enc, dec),
    }


def build_primary_contrasts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Builds the prespecified E8 contrast families.

    The analysis deliberately avoids all-pairs post-hoc architecture testing.
    Architecture contrasts use 4E-4D (or the configured reference) against
    only the prespecified alternatives, and only in the configured principal
    stage/domain conditions.
    """
    protocol = _read_protocol(cfg)
    seeds, directions, architectures = _load_grid(cfg)
    _ = seeds

    contrasts: list[dict[str, Any]] = []

    if protocol["enable_adaptation"]:
        for direction in directions:
            for arch in architectures:
                for domain in protocol["adaptation_domains"]:
                    enc, dec = arch
                    stem = (
                        f"{direction['src']}_to_{direction['tgt']}|"
                        f"{_architecture_label(enc, dec)}|{domain}"
                    )
                    contrasts.append({
                        "contrast_family": "adaptation_vs_base",
                        "contrast_id": f"adaptation_vs_base|{stem}",
                        "src": direction["src"],
                        "tgt": direction["tgt"],
                        "domain": domain,
                        "system_a": _system_spec("base", arch),
                        "system_b": _system_spec("finetuned", arch),
                        "delta_definition": "finetuned_minus_base",
                    })
                    contrasts.append({
                        "contrast_family": "adaptation_vs_continued_control",
                        "contrast_id": f"adaptation_vs_continued_control|{stem}",
                        "src": direction["src"],
                        "tgt": direction["tgt"],
                        "domain": domain,
                        "system_a": _system_spec("continued_control", arch),
                        "system_b": _system_spec("finetuned", arch),
                        "delta_definition": "finetuned_minus_continued_control",
                    })

    if protocol["enable_architecture"]:
        ref_arch = protocol["architecture_reference"]
        if ref_arch not in architectures:
            raise ValueError(
                "bootstrap_architecture_reference is not present in experiment_architectures."
            )
        for comp_arch in protocol["architecture_comparators"]:
            if comp_arch not in architectures:
                raise ValueError(
                    f"Architecture comparator {comp_arch} is not present in experiment_architectures."
                )

        for direction in directions:
            for condition in protocol["architecture_conditions"]:
                for comp_arch in protocol["architecture_comparators"]:
                    stage = condition["stage"]
                    domain = condition["domain"]
                    stem = (
                        f"{stage}|{direction['src']}_to_{direction['tgt']}|{domain}|"
                        f"{_architecture_label(*comp_arch)}_vs_{_architecture_label(*ref_arch)}"
                    )
                    contrasts.append({
                        "contrast_family": "architecture_vs_4E4D_reference",
                        "contrast_id": f"architecture_vs_reference|{stem}",
                        "src": direction["src"],
                        "tgt": direction["tgt"],
                        "domain": domain,
                        "system_a": _system_spec(stage, ref_arch),
                        "system_b": _system_spec(stage, comp_arch),
                        "delta_definition": (
                            f"{_architecture_label(*comp_arch)}_minus_"
                            f"{_architecture_label(*ref_arch)}"
                        ),
                    })

    if not contrasts:
        raise ValueError("E8 contrast plan is empty.")

    ids = [x["contrast_id"] for x in contrasts]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Internal error: duplicate E8 contrast identifiers.")
    return contrasts


# ==============================================================================
# TEST-SPLIT BOOTSTRAP UNITS
# ==============================================================================


@dataclass(frozen=True)
class BootstrapUnits:
    domain: str
    split_path: Path
    split_sha256: str
    unit_column: str
    labels: np.ndarray
    unique_labels: np.ndarray
    codes: np.ndarray

    @property
    def n_examples(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.unique_labels.shape[0])


def _load_bootstrap_units(
    project_root: Path,
    cfg: dict[str, Any],
    domain: str,
    unit_column: str,
) -> BootstrapUnits:
    splits_dir = _resolve_project_path(project_root, cfg["splits_dir"])
    if domain == "biblical":
        split_path = splits_dir / "test_bible.tsv"
    elif domain == "conversational":
        split_path = splits_dir / "test_dialogue.tsv"
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    if not split_path.exists():
        raise FileNotFoundError(f"Test split required by E8 is missing: {split_path}")

    split_df = pd.read_csv(split_path, sep="\t")
    if unit_column == "example_index":
        labels = np.arange(len(split_df), dtype=np.int64).astype(str)
    else:
        if unit_column not in split_df.columns:
            raise KeyError(
                f"Bootstrap unit column {unit_column!r} not found in {split_path}. "
                f"Available columns: {list(split_df.columns)}"
            )
        if split_df[unit_column].isna().any():
            raise ValueError(
                f"Bootstrap unit column {unit_column!r} contains missing values in {split_path}."
            )
        labels = split_df[unit_column].astype(str).to_numpy()

    codes, uniques = pd.factorize(labels, sort=False)
    if len(uniques) < 2:
        raise ValueError(
            f"Bootstrap requires at least two resampling units; found {len(uniques)} "
            f"for {domain}."
        )

    return BootstrapUnits(
        domain=domain,
        split_path=split_path,
        split_sha256=_sha256_file(split_path),
        unit_column=unit_column,
        labels=np.asarray(labels),
        unique_labels=np.asarray(uniques),
        codes=np.asarray(codes, dtype=np.int64),
    )


# ==============================================================================
# PREDICTION ARTIFACTS
# ==============================================================================


@dataclass
class PredictionArtifact:
    stage: str
    domain: str
    src: str
    tgt: str
    encoder_layers: int
    decoder_layers: int
    seed: int
    prediction_path: Path
    metrics_path: Path
    prediction_sha256: str
    metrics_sha256: str
    references: list[str]
    hypotheses: list[str]
    metrics_payload: dict[str, Any]
    observed_scores: dict[str, float]
    _segment_stats: dict[str, np.ndarray] = field(default_factory=dict)
    _group_stats: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    @property
    def model_tag(self) -> str:
        return _model_tag(
            self.src,
            self.tgt,
            self.encoder_layers,
            self.decoder_layers,
            self.seed,
        )

    @property
    def architecture(self) -> str:
        return _architecture_label(self.encoder_layers, self.decoder_layers)

    def segment_stats(self, metric_name: str) -> np.ndarray:
        if metric_name not in self._segment_stats:
            metric = _build_metric(metric_name)
            try:
                raw_stats = metric._extract_corpus_statistics(  # noqa: SLF001
                    self.hypotheses,
                    [self.references],
                )
            except AttributeError as exc:
                raise RuntimeError(
                    "SacreBLEU does not expose the expected internal corpus-statistics "
                    "API. E8 is pinned to SacreBLEU 2.6.0; verify the environment."
                ) from exc
            stats = np.asarray(raw_stats, dtype=np.int64)
            if stats.ndim != 2 or stats.shape[0] != len(self.references):
                raise RuntimeError(
                    f"Unexpected {metric_name} statistics shape for {self.prediction_path}: "
                    f"{stats.shape}"
                )

            # One-time consistency check: the sufficient statistics must exactly
            # reproduce the public corpus score used in evaluation.
            aggregated = stats.sum(axis=0)
            reconstructed = float(
                metric._compute_score_from_stats(aggregated.tolist()).score  # noqa: SLF001
            )
            public = float(self.observed_scores[metric_name])
            if not _float_close(reconstructed, public, 1e-10):
                raise RuntimeError(
                    f"Internal-statistics score mismatch for {self.prediction_path} "
                    f"[{metric_name}]: reconstructed={reconstructed}, public={public}."
                )
            self._segment_stats[metric_name] = stats
        return self._segment_stats[metric_name]

    def group_stats(self, metric_name: str, units: BootstrapUnits) -> np.ndarray:
        key = (metric_name, units.split_sha256)
        if key not in self._group_stats:
            stats = self.segment_stats(metric_name)
            if stats.shape[0] != units.n_examples:
                raise RuntimeError(
                    f"Prediction/test-split length mismatch for {self.prediction_path}: "
                    f"predictions={stats.shape[0]}, split={units.n_examples}."
                )
            grouped = np.zeros(
                (units.n_units, stats.shape[1]),
                dtype=np.int64,
            )
            np.add.at(grouped, units.codes, stats)
            self._group_stats[key] = grouped
        return self._group_stats[key]


def _build_metric(metric_name: str):
    if metric_name == "bleu":
        return _build_bleu_metric()
    if metric_name == "chrf++":
        return _build_chrf_metric()
    raise ValueError(f"Unsupported metric: {metric_name}")


class ArtifactRepository:
    def __init__(
        self,
        project_root: Path,
        cfg: dict[str, Any],
        units_by_domain: dict[str, BootstrapUnits],
        validation_tolerance: float,
    ) -> None:
        self.project_root = project_root
        self.cfg = cfg
        self.results_dir = _resolve_project_path(project_root, cfg["results_dir"])
        self.units_by_domain = units_by_domain
        self.validation_tolerance = float(validation_tolerance)
        self._cache: dict[tuple[Any, ...], PredictionArtifact] = {}

    def _paths(
        self,
        stage: str,
        domain: str,
        src: str,
        tgt: str,
        architecture: tuple[int, int],
        seed: int,
    ) -> tuple[Path, Path]:
        if stage not in _STAGE_PREFIX:
            raise ValueError(f"Unsupported stage: {stage}")
        if domain not in _SUPPORTED_DOMAINS:
            raise ValueError(f"Unsupported domain: {domain}")
        enc, dec = architecture
        tag = _model_tag(src, tgt, enc, dec, seed)
        prediction = self.results_dir / f"{_STAGE_PREFIX[stage]}_{domain}_{tag}.tsv"
        metrics = self.results_dir / f"{_STAGE_METRICS_PREFIX[stage]}_{tag}.json"
        return prediction, metrics

    def get(
        self,
        stage: str,
        domain: str,
        src: str,
        tgt: str,
        architecture: tuple[int, int],
        seed: int,
        validate_scores: bool = True,
    ) -> PredictionArtifact:
        key = (stage, domain, src, tgt, architecture[0], architecture[1], int(seed))
        if key in self._cache:
            return self._cache[key]

        prediction_path, metrics_path = self._paths(
            stage, domain, src, tgt, architecture, seed
        )
        if not prediction_path.exists():
            raise FileNotFoundError(f"E8 prediction artifact is missing: {prediction_path}")
        if not metrics_path.exists():
            raise FileNotFoundError(f"E8 metrics artifact is missing: {metrics_path}")

        prediction_sha = _sha256_file(prediction_path)
        metrics_sha = _sha256_file(metrics_path)
        metrics_payload = _load_json(metrics_path)
        enc, dec = architecture
        tag = _model_tag(src, tgt, enc, dec, seed)

        mismatches: list[str] = []
        scalar_checks = {
            "model_tag": (metrics_payload.get("model_tag"), tag),
            "src": (metrics_payload.get("src"), src),
            "tgt": (metrics_payload.get("tgt"), tgt),
            "experiment_seed": (metrics_payload.get("experiment_seed"), int(seed)),
            "encoder_layers": (
                metrics_payload.get("model", {}).get("architecture", {}).get("encoder_layers"),
                int(enc),
            ),
            "decoder_layers": (
                metrics_payload.get("model", {}).get("architecture", {}).get("decoder_layers"),
                int(dec),
            ),
        }
        for label, (stored, expected) in scalar_checks.items():
            if stored != expected:
                mismatches.append(f"{label}: stored={stored!r}, expected={expected!r}")

        hash_key = f"{domain}_predictions_sha256"
        stored_prediction_hash = metrics_payload.get("evaluation_hashes", {}).get(hash_key)
        if stored_prediction_hash != prediction_sha:
            mismatches.append(
                f"evaluation_hashes.{hash_key}: stored={stored_prediction_hash!r}, "
                f"actual={prediction_sha!r}"
            )

        result_key = "biblical_results" if domain == "biblical" else "conversational_results"
        result_payload = metrics_payload.get(result_key, {})

        frame = pd.read_csv(
            prediction_path,
            sep="\t",
            dtype={"reference": str, "hypothesis": str},
            keep_default_na=False,
        )
        required_columns = ["example_index", "reference", "hypothesis"]
        if list(frame.columns) != required_columns:
            mismatches.append(
                f"prediction columns: stored={list(frame.columns)!r}, "
                f"expected={required_columns!r}"
            )
        else:
            expected_index = np.arange(len(frame), dtype=np.int64)
            actual_index = frame["example_index"].to_numpy(dtype=np.int64)
            if not np.array_equal(actual_index, expected_index):
                mismatches.append("example_index is not the exact sequence 0..N-1")
            if frame["reference"].eq("").any():
                mismatches.append("prediction TSV contains empty references")
            if frame["hypothesis"].eq("").any():
                mismatches.append("prediction TSV contains empty hypotheses")

        expected_pairs = result_payload.get("test_pairs")
        if expected_pairs != len(frame):
            mismatches.append(
                f"test_pairs: stored={expected_pairs!r}, TSV_rows={len(frame)!r}"
            )

        units = self.units_by_domain[domain]
        if len(frame) != units.n_examples:
            mismatches.append(
                f"test split rows={units.n_examples}, prediction rows={len(frame)}"
            )

        if mismatches:
            text = "\n".join(f"  - {item}" for item in mismatches)
            raise RuntimeError(
                f"E8 rejected incompatible artifact {prediction_path}.\nMismatches:\n{text}"
            )

        references = frame["reference"].astype(str).tolist()
        hypotheses = frame["hypothesis"].astype(str).tolist()

        stored_scores = {
            "bleu": float(result_payload["bleu_score"]),
            "chrf++": float(result_payload["chrf_score"]),
        }
        observed_scores = dict(stored_scores)

        if validate_scores:
            for metric_name in _SUPPORTED_METRICS:
                metric = _build_metric(metric_name)
                public_score = float(metric.corpus_score(hypotheses, [references]).score)
                if not _float_close(
                    public_score,
                    stored_scores[metric_name],
                    self.validation_tolerance,
                ):
                    raise RuntimeError(
                        f"Stored/public {metric_name} mismatch for {prediction_path}: "
                        f"stored={stored_scores[metric_name]}, recomputed={public_score}."
                    )
                observed_scores[metric_name] = public_score

        artifact = PredictionArtifact(
            stage=stage,
            domain=domain,
            src=src,
            tgt=tgt,
            encoder_layers=int(enc),
            decoder_layers=int(dec),
            seed=int(seed),
            prediction_path=prediction_path,
            metrics_path=metrics_path,
            prediction_sha256=prediction_sha,
            metrics_sha256=metrics_sha,
            references=references,
            hypotheses=hypotheses,
            metrics_payload=metrics_payload,
            observed_scores=observed_scores,
        )
        self._cache[key] = artifact
        return artifact

    def manifest(self) -> list[dict[str, Any]]:
        rows = []
        for artifact in sorted(
            self._cache.values(),
            key=lambda x: (
                x.stage,
                x.domain,
                x.src,
                x.tgt,
                x.encoder_layers,
                x.decoder_layers,
                x.seed,
            ),
        ):
            rows.append({
                "stage": artifact.stage,
                "domain": artifact.domain,
                "src": artifact.src,
                "tgt": artifact.tgt,
                "architecture": artifact.architecture,
                "seed": artifact.seed,
                "prediction_file": os.path.relpath(
                    artifact.prediction_path, self.project_root
                ),
                "prediction_sha256": artifact.prediction_sha256,
                "metrics_file": os.path.relpath(artifact.metrics_path, self.project_root),
                "metrics_sha256": artifact.metrics_sha256,
            })
        return rows


# ==============================================================================
# PAIRED GROUP BOOTSTRAP
# ==============================================================================


def _derived_seed(base_seed: int, contrast_id: str, metric_name: str) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}|{contrast_id}|{metric_name}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)


def _score_rows(metric, aggregated_stats: np.ndarray) -> np.ndarray:
    scores = np.empty(aggregated_stats.shape[0], dtype=np.float64)
    for idx, row in enumerate(aggregated_stats):
        scores[idx] = float(
            metric._compute_score_from_stats(row.tolist()).score  # noqa: SLF001
        )
    return scores


def paired_bootstrap(
    artifacts_a: list[PredictionArtifact],
    artifacts_b: list[PredictionArtifact],
    units: BootstrapUnits,
    metric_name: str,
    n_resamples: int,
    confidence_level: float,
    bootstrap_seed: int,
    contrast_id: str,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """
    Paired cluster/group bootstrap for a difference in corpus MT metrics.

    A single bootstrap sample of split groups is applied simultaneously to
    both systems and to every model seed. Therefore pairing is preserved at
    the test-data level. The bootstrap distribution is the mean metric
    difference across model seeds; seed-to-seed variability remains a
    separate quantity and is not conflated with test-set uncertainty.
    """
    if len(artifacts_a) != len(artifacts_b) or not artifacts_a:
        raise ValueError("Paired bootstrap requires the same non-zero number of seed runs.")

    seed_ids_a = [x.seed for x in artifacts_a]
    seed_ids_b = [x.seed for x in artifacts_b]
    if seed_ids_a != seed_ids_b:
        raise ValueError(
            f"Seed pairing mismatch: system_a={seed_ids_a}, system_b={seed_ids_b}."
        )

    # Exact reference pairing across systems and seeds.
    canonical_references = artifacts_a[0].references
    for artifact in artifacts_a[1:] + artifacts_b:
        if artifact.references != canonical_references:
            raise RuntimeError(
                f"Reference mismatch detected in paired bootstrap contrast {contrast_id}."
            )

    metric = _build_metric(metric_name)

    # SacreBLEU 2.6.0 only knows the number of references used by a metric
    # after at least one corpus evaluation has been performed with that metric
    # instance.  Initialize the metric here before requesting its signature.
    # This score is not used as a bootstrap statistic; it only initializes
    # SacreBLEU's reproducibility metadata (nrefs=1).
    metric.corpus_score(
        artifacts_a[0].hypotheses,
        [canonical_references],
    )
    metric_signature = str(metric.get_signature())

    group_stats_a = [x.group_stats(metric_name, units) for x in artifacts_a]
    group_stats_b = [x.group_stats(metric_name, units) for x in artifacts_b]

    observed_a_by_seed = [float(x.observed_scores[metric_name]) for x in artifacts_a]
    observed_b_by_seed = [float(x.observed_scores[metric_name]) for x in artifacts_b]
    observed_delta_by_seed = [
        b - a for a, b in zip(observed_a_by_seed, observed_b_by_seed)
    ]
    observed_a = float(np.mean(observed_a_by_seed))
    observed_b = float(np.mean(observed_b_by_seed))
    observed_delta = float(np.mean(observed_delta_by_seed))

    derived_seed = _derived_seed(bootstrap_seed, contrast_id, metric_name)
    rng = np.random.default_rng(derived_seed)
    probabilities = np.full(units.n_units, 1.0 / units.n_units, dtype=np.float64)
    bootstrap_deltas = np.empty(int(n_resamples), dtype=np.float64)

    write_index = 0
    while write_index < n_resamples:
        current = min(chunk_size, n_resamples - write_index)
        counts = rng.multinomial(
            n=units.n_units,
            pvals=probabilities,
            size=current,
        ).astype(np.int64, copy=False)

        mean_delta_chunk = np.zeros(current, dtype=np.float64)
        for stats_a, stats_b in zip(group_stats_a, group_stats_b):
            aggregate_a = counts @ stats_a
            aggregate_b = counts @ stats_b
            score_a = _score_rows(metric, aggregate_a)
            score_b = _score_rows(metric, aggregate_b)
            mean_delta_chunk += (score_b - score_a) / len(group_stats_a)

        bootstrap_deltas[write_index : write_index + current] = mean_delta_chunk
        write_index += current

    alpha = 1.0 - float(confidence_level)
    ci_low, ci_high = np.quantile(
        bootstrap_deltas,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )

    return {
        "metric": metric_name,
        "metric_signature": metric_signature,
        "n_seeds": len(artifacts_a),
        "seeds": seed_ids_a,
        "score_a_mean": observed_a,
        "score_b_mean": observed_b,
        "observed_delta_b_minus_a": observed_delta,
        "observed_delta_by_seed": {
            str(seed): float(delta)
            for seed, delta in zip(seed_ids_a, observed_delta_by_seed)
        },
        "score_a_by_seed": {
            str(seed): float(score)
            for seed, score in zip(seed_ids_a, observed_a_by_seed)
        },
        "score_b_by_seed": {
            str(seed): float(score)
            for seed, score in zip(seed_ids_b, observed_b_by_seed)
        },
        "bootstrap_replicates": int(n_resamples),
        "bootstrap_seed_base": int(bootstrap_seed),
        "bootstrap_seed_derived": int(derived_seed),
        "confidence_level": float(confidence_level),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_delta_mean": float(np.mean(bootstrap_deltas)),
        "bootstrap_delta_sd": float(np.std(bootstrap_deltas, ddof=1)),
        "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
        "resampling_unit_column": units.unit_column,
        "n_test_examples": units.n_examples,
        "n_resampling_units": units.n_units,
    }


# Backward-compatible alias for the earlier public function name.
paired_group_bootstrap = paired_bootstrap


# ==============================================================================
# COMPLETE E8 ANALYSIS
# ==============================================================================


def _spec_tuple(system: dict[str, Any]) -> tuple[str, tuple[int, int]]:
    return (
        str(system["stage"]),
        (int(system["encoder_layers"]), int(system["decoder_layers"])),
    )


def _preload_required_artifacts(
    repository: ArtifactRepository,
    contrasts: Iterable[dict[str, Any]],
    seeds: list[int],
    validate_scores: bool,
) -> None:
    requested: set[tuple[Any, ...]] = set()
    for contrast in contrasts:
        for side in ("system_a", "system_b"):
            stage, architecture = _spec_tuple(contrast[side])
            for seed in seeds:
                requested.add((
                    stage,
                    contrast["domain"],
                    contrast["src"],
                    contrast["tgt"],
                    architecture[0],
                    architecture[1],
                    seed,
                ))

    for item in sorted(requested):
        stage, domain, src, tgt, enc, dec, seed = item
        repository.get(
            stage=stage,
            domain=domain,
            src=src,
            tgt=tgt,
            architecture=(enc, dec),
            seed=seed,
            validate_scores=validate_scores,
        )


def _analysis_signature_payload(
    protocol: dict[str, Any],
    contrasts: list[dict[str, Any]],
    units_by_domain: dict[str, BootstrapUnits],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "protocol": {
            "bootstrap_replicates": protocol["replicates"],
            "confidence_level": protocol["confidence_level"],
            "bootstrap_seed": protocol["seed"],
            "expected_sacrebleu_version": protocol["expected_sacrebleu_version"],
            "resampling_units": protocol["resampling_units"],
            "enable_adaptation": protocol["enable_adaptation"],
            "enable_architecture": protocol["enable_architecture"],
            "architecture_reference": list(protocol["architecture_reference"]),
            "architecture_comparators": [list(x) for x in protocol["architecture_comparators"]],
            "architecture_conditions": protocol["architecture_conditions"],
            "adaptation_domains": protocol["adaptation_domains"],
        },
        "contrasts": contrasts,
        "test_splits": {
            domain: {
                "path": str(units.split_path),
                "sha256": units.split_sha256,
                "unit_column": units.unit_column,
                "n_examples": units.n_examples,
                "n_units": units.n_units,
            }
            for domain, units in units_by_domain.items()
        },
        "input_artifacts": manifest,
    }


def _existing_outputs_are_compatible(
    json_path: Path,
    csv_path: Path,
    signature_sha256: str,
) -> bool:
    payload = _load_json(json_path)
    if payload.get("analysis_signature_sha256") != signature_sha256:
        return False
    stored_csv_hash = payload.get("output_hashes", {}).get("summary_csv_sha256")
    if stored_csv_hash != _sha256_file(csv_path):
        return False
    return True


def run_bootstrap_analysis(
    config_path: str | Path = "config.yaml",
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """
    Runs the complete E8 paired-bootstrap analysis.

    Final interpretation uses percentile 95% confidence intervals for
    prespecified score differences. The bootstrap resamples the domain-
    specific test units defined in config.yaml and applies exactly the same
    sampled units to the two compared systems and to all model seeds.
    """
    config_path = Path(config_path).resolve()
    project_root = config_path.parent
    cfg = load_config(config_path)
    protocol = _read_protocol(cfg)
    seeds, directions, architectures = _load_grid(cfg)
    _ = directions, architectures

    if sacrebleu.__version__ != protocol["expected_sacrebleu_version"]:
        raise RuntimeError(
            "E8 SacreBLEU version mismatch: "
            f"expected={protocol['expected_sacrebleu_version']}, "
            f"installed={sacrebleu.__version__}."
        )

    contrasts = build_primary_contrasts(cfg)
    units_by_domain = {
        domain: _load_bootstrap_units(
            project_root,
            cfg,
            domain,
            protocol["resampling_units"][domain],
        )
        for domain in _SUPPORTED_DOMAINS
    }

    repository = ArtifactRepository(
        project_root=project_root,
        cfg=cfg,
        units_by_domain=units_by_domain,
        validation_tolerance=protocol["metric_validation_tolerance"],
    )

    # Preflight is intentionally strict: E8 never silently analyzes partial,
    # stale, or hash-incompatible prediction files.
    _preload_required_artifacts(
        repository,
        contrasts,
        seeds,
        validate_scores=not dry_run,
    )

    manifest = repository.manifest()
    signature_payload = _analysis_signature_payload(
        protocol,
        contrasts,
        units_by_domain,
        manifest,
    )
    signature_sha256 = _sha256_jsonable(signature_payload)

    results_dir = _resolve_project_path(project_root, cfg["results_dir"])
    output_dir = results_dir / protocol["output_subdir"]
    output_prefix = protocol["output_prefix"]
    csv_path = output_dir / f"{output_prefix}.csv"
    json_path = output_dir / f"{output_prefix}.json"

    existing = [path for path in (csv_path, json_path) if path.exists()]
    if existing and len(existing) != 2 and not force:
        raise RuntimeError(
            "Partial E8 output artifacts detected. Inspect them before continuing. "
            "Use --force only for intentional regeneration."
        )
    if len(existing) == 2 and not force:
        if _existing_outputs_are_compatible(
            json_path,
            csv_path,
            signature_sha256,
        ):
            print("[SKIP] Compatible E8 bootstrap outputs already exist.")
            return _load_json(json_path)
        raise RuntimeError(
            "Complete E8 bootstrap outputs exist, but they are incompatible with "
            "the current configuration or prediction hashes. Use --force only "
            "when intentional regeneration is required."
        )

    print("\n" + "=" * 80)
    print("E8 — PAIRED BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("=" * 80)
    print(f"Seeds                  : {seeds}")
    print(f"Bootstrap replicates   : {protocol['replicates']}")
    print(f"Confidence level       : {protocol['confidence_level']:.3f}")
    print(f"Bootstrap seed         : {protocol['seed']}")
    print(f"SacreBLEU              : {sacrebleu.__version__}")
    print(f"Primary contrasts      : {len(contrasts)}")
    for domain in _SUPPORTED_DOMAINS:
        units = units_by_domain[domain]
        print(
            f"{domain.capitalize():<23}: {units.n_examples} examples | "
            f"{units.n_units} {units.unit_column} units"
        )
    print(f"Output CSV             : {csv_path}")
    print(f"Output JSON            : {json_path}")
    print(f"Force                  : {force}")
    print(f"Dry run                : {dry_run}")
    print("=" * 80)

    if dry_run:
        for index, contrast in enumerate(contrasts, start=1):
            print(
                f"[DRY RUN] {index:02d}/{len(contrasts):02d} "
                f"{contrast['contrast_id']}"
            )
        return None

    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    json_results: list[dict[str, Any]] = []

    for index, contrast in enumerate(contrasts, start=1):
        stage_a, arch_a = _spec_tuple(contrast["system_a"])
        stage_b, arch_b = _spec_tuple(contrast["system_b"])
        units = units_by_domain[contrast["domain"]]

        artifacts_a = [
            repository.get(
                stage=stage_a,
                domain=contrast["domain"],
                src=contrast["src"],
                tgt=contrast["tgt"],
                architecture=arch_a,
                seed=seed,
                validate_scores=True,
            )
            for seed in seeds
        ]
        artifacts_b = [
            repository.get(
                stage=stage_b,
                domain=contrast["domain"],
                src=contrast["src"],
                tgt=contrast["tgt"],
                architecture=arch_b,
                seed=seed,
                validate_scores=True,
            )
            for seed in seeds
        ]

        print(
            f"[{index:02d}/{len(contrasts):02d}] {contrast['contrast_id']}"
        )

        metric_results = []
        for metric_name in _SUPPORTED_METRICS:
            result = paired_bootstrap(
                artifacts_a=artifacts_a,
                artifacts_b=artifacts_b,
                units=units,
                metric_name=metric_name,
                n_resamples=protocol["replicates"],
                confidence_level=protocol["confidence_level"],
                bootstrap_seed=protocol["seed"],
                contrast_id=contrast["contrast_id"],
            )
            metric_results.append(result)

            row = {
                "contrast_family": contrast["contrast_family"],
                "contrast_id": contrast["contrast_id"],
                "direction": f"{contrast['src']}->{contrast['tgt']}",
                "domain": contrast["domain"],
                "system_a_stage": stage_a,
                "system_a_architecture": _architecture_label(*arch_a),
                "system_b_stage": stage_b,
                "system_b_architecture": _architecture_label(*arch_b),
                "delta_definition": contrast["delta_definition"],
                "metric": metric_name,
                "n_seeds": result["n_seeds"],
                "seeds": ";".join(str(x) for x in result["seeds"]),
                "score_a_mean": result["score_a_mean"],
                "score_b_mean": result["score_b_mean"],
                "observed_delta_b_minus_a": result["observed_delta_b_minus_a"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "confidence_level": result["confidence_level"],
                "ci_excludes_zero": result["ci_excludes_zero"],
                "bootstrap_replicates": result["bootstrap_replicates"],
                "bootstrap_seed_derived": result["bootstrap_seed_derived"],
                "resampling_unit_column": result["resampling_unit_column"],
                "n_test_examples": result["n_test_examples"],
                "n_resampling_units": result["n_resampling_units"],
                "metric_signature": result["metric_signature"],
            }
            rows.append(row)

        json_results.append({
            **contrast,
            "metrics": metric_results,
        })

    elapsed = time.perf_counter() - start
    summary_df = pd.DataFrame(rows)
    _atomic_csv_dump(summary_df, csv_path)
    csv_sha256 = _sha256_file(csv_path)

    payload: dict[str, Any] = {
        "analysis_type": "E8-Paired-Bootstrap",
        "objective": "paired-95pct-CI-for-prespecified-primary-contrasts",
        "analysis_signature_sha256": signature_sha256,
        "protocol": {
            "bootstrap_replicates": protocol["replicates"],
            "confidence_level": protocol["confidence_level"],
            "bootstrap_seed": protocol["seed"],
            "resampling_method": "paired_domain_specific_bootstrap_with_replacement",
            "seed_aggregation": (
                "same resampled test units applied to both systems and all model seeds; "
                "bootstrap statistic is mean score difference across seeds"
            ),
            "resampling_unit_policy": {
                "biblical": (
                    "cluster bootstrap over usfm_code so aligned edition variants of "
                    "the same underlying verse remain together"
                ),
                "conversational": (
                    "paired example-level bootstrap over example_index; the six dialogue "
                    "clusters are retained for data splitting but are too few for a "
                    "stable percentile cluster-bootstrap interval"
                ),
            },
            "seed_variability_policy": (
                "model-seed variability remains reported separately as mean±SD; "
                "E8 intervals quantify test-set resampling uncertainty"
            ),
            "delta_convention": "system_b_minus_system_a",
            "interval": "percentile",
            "metrics": list(_SUPPORTED_METRICS),
            "sacrebleu_version": sacrebleu.__version__,
            "resampling_units": protocol["resampling_units"],
            "prespecified_contrasts_only": True,
            "multiplicity_note": (
                "No exhaustive all-pairs architecture search is performed; only "
                "prespecified contrasts are generated."
            ),
        },
        "test_splits": {
            domain: {
                "path": os.path.relpath(units.split_path, project_root),
                "sha256": units.split_sha256,
                "unit_column": units.unit_column,
                "n_examples": units.n_examples,
                "n_units": units.n_units,
            }
            for domain, units in units_by_domain.items()
        },
        "input_artifacts": repository.manifest(),
        "primary_contrasts": contrasts,
        "results": json_results,
        "output_files": {
            "summary_csv": os.path.relpath(csv_path, project_root),
            "summary_json": os.path.relpath(json_path, project_root),
        },
        "output_hashes": {
            "summary_csv_sha256": csv_sha256,
        },
        "timing": {
            "analysis_seconds": elapsed,
            "analysis_hours": elapsed / 3600.0,
        },
        "git": _get_git_metadata(project_root),
        "config": cfg,
    }
    _atomic_json_dump(payload, json_path)

    print("\n" + "=" * 80)
    print("E8 PAIRED BOOTSTRAP COMPLETE")
    print("=" * 80)
    print(f"Contrast families       : {len(contrasts)}")
    print(f"Metric rows             : {len(summary_df)}")
    print(f"Summary CSV             : {csv_path}")
    print(f"Summary JSON            : {json_path}")
    print(f"Execution time          : {elapsed:.2f} s")
    print("=" * 80)
    return payload
