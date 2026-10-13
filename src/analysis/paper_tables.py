# ================================
# ./src/analysis/paper_tables.py
# ================================

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.utils.config import load_config


GENERATOR_VERSION = "1.1"

_TABLE_FILES = (
    "table_1_dataset_partitions.tex",
    "table_2_model_training_protocol.tex",
    "table_3_translation_performance.tex",
    "table_4_primary_bootstrap_contrasts.tex",
    "table_5_inference_efficiency.tex",
)

_METRIC_FILE_RE = re.compile(
    r"^(?P<stage>base|finetuned)_test_metrics_"
    r"(?P<direction>aym_to_spa|spa_to_aym)_"
    r"Enc(?P<enc>\d+)_Dec(?P<dec>\d+)_Seed(?P<seed>\d+)\.json$"
)

_BASE_SUMMARY_RE = re.compile(
    r"^base_training_summary_"
    r"(?P<direction>aym_to_spa|spa_to_aym)_"
    r"Enc(?P<enc>\d+)_Dec(?P<dec>\d+)_Seed(?P<seed>\d+)\.json$"
)

_FT_SUMMARY_RE = re.compile(
    r"^finetuning_summary_"
    r"(?P<direction>aym_to_spa|spa_to_aym)_"
    r"Enc(?P<enc>\d+)_Dec(?P<dec>\d+)_Seed(?P<seed>\d+)\.json$"
)

_CT_SUMMARY_RE = re.compile(
    r"^continued_control_summary_"
    r"(?P<direction>aym_to_spa|spa_to_aym)_"
    r"Enc(?P<enc>\d+)_Dec(?P<dec>\d+)_Seed(?P<seed>\d+)\.json$"
)


class PaperTableError(RuntimeError):
    """Raised when final table inputs fail scientific consistency checks."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PaperTableError(f"Expected JSON object in {path}.")
    return value


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _arch_label(enc: int, dec: int) -> str:
    return f"{enc}E-{dec}D"


def _latex_arch(label: str) -> str:
    return label.replace("-", "--")


def _direction_key(src: str, tgt: str) -> str:
    return f"{src}_to_{tgt}"


def _direction_arrow(direction: str) -> str:
    mapping = {
        "aym_to_spa": r"Aym$\rightarrow$Spa",
        "spa_to_aym": r"Spa$\rightarrow$Aym",
        "aym->spa": r"Aym$\rightarrow$Spa",
        "spa->aym": r"Spa$\rightarrow$Aym",
    }
    try:
        return mapping[direction]
    except KeyError as exc:
        raise PaperTableError(f"Unexpected direction: {direction}") from exc


def _expected_grid(cfg: dict[str, Any]) -> set[tuple[str, str, int]]:
    expected: set[tuple[str, str, int]] = set()
    for direction in cfg["experiment_directions"]:
        direction_key = _direction_key(direction["src"], direction["tgt"])
        for architecture in cfg["experiment_architectures"]:
            arch = _arch_label(
                int(architecture["encoder_layers"]),
                int(architecture["decoder_layers"]),
            )
            for seed in cfg["experiment_seeds"]:
                expected.add((direction_key, arch, int(seed)))
    return expected


def _architecture_order(cfg: dict[str, Any]) -> list[str]:
    return [
        _arch_label(
            int(item["encoder_layers"]),
            int(item["decoder_layers"]),
        )
        for item in cfg["experiment_architectures"]
    ]


def _direction_order(cfg: dict[str, Any]) -> list[str]:
    return [
        _direction_key(item["src"], item["tgt"])
        for item in cfg["experiment_directions"]
    ]


def _require_finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise PaperTableError(f"Non-finite value for {label}: {value}")
    return value


def _sample_mean_sd(values: Iterable[float]) -> tuple[float, float]:
    values = [float(value) for value in values]
    if len(values) < 2:
        raise PaperTableError("At least two values are required for sample SD.")
    for value in values:
        _require_finite(value, "sample value")
    return statistics.mean(values), statistics.stdev(values)


def _fmt_mean_sd(mean: float, sd: float, decimals: int = 2) -> str:
    return rf"${mean:.{decimals}f}\pm{sd:.{decimals}f}$"


def _fmt_sci_latex(value: float) -> str:
    if value <= 0:
        raise PaperTableError(f"Learning rate must be positive: {value}")
    exponent = int(math.floor(math.log10(value)))
    coefficient = value / (10 ** exponent)
    if math.isclose(coefficient, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return rf"$10^{{{exponent}}}$"
    return rf"${coefficient:g}\times10^{{{exponent}}}$"


def _fmt_delta_ci(delta: float, low: float, high: float, decimals: int = 2) -> str:
    sign = "+" if delta > 0 else ""
    return (
        rf"${sign}{delta:.{decimals}f}\;"
        rf"[{low:.{decimals}f},\,{high:.{decimals}f}]$"
    )


def _relative_to_root(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _split_stats(path: Path, group_col: str) -> tuple[int, set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required split file not found: {path}")

    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        usecols=[group_col],
    )
    if frame[group_col].eq("").any():
        raise PaperTableError(f"Empty grouping identifier in {path}: {group_col}")
    groups = set(frame[group_col].tolist())
    return len(frame), groups


def _dataset_partition_summary(
    project_root: Path,
    cfg: dict[str, Any],
) -> tuple[dict[str, dict[str, list[int]]], list[Path]]:
    splits_dir = project_root / cfg["splits_dir"]
    inputs: list[Path] = []

    specifications = {
        "biblical": {
            "group_col": "usfm_code",
            "files": [
                splits_dir / "train_bible.tsv",
                splits_dir / "val_bible.tsv",
                splits_dir / "test_bible.tsv",
            ],
        },
        "conversational": {
            "group_col": "dialogue_id",
            "files": [
                splits_dir / "train_dialogue.tsv",
                splits_dir / "val_dialogue.tsv",
                splits_dir / "test_dialogue.tsv",
            ],
        },
    }

    summary: dict[str, dict[str, list[int]]] = {}
    for corpus, spec in specifications.items():
        pair_counts: list[int] = []
        group_sets: list[set[str]] = []
        for path in spec["files"]:
            pairs, groups = _split_stats(path, spec["group_col"])
            pair_counts.append(pairs)
            group_sets.append(groups)
            inputs.append(path)

        train_groups, val_groups, test_groups = group_sets
        if train_groups & val_groups:
            raise PaperTableError(f"{corpus}: grouping leakage train/validation.")
        if train_groups & test_groups:
            raise PaperTableError(f"{corpus}: grouping leakage train/test.")
        if val_groups & test_groups:
            raise PaperTableError(f"{corpus}: grouping leakage validation/test.")

        group_counts = [len(groups) for groups in group_sets]
        summary[corpus] = {
            "pairs": pair_counts + [sum(pair_counts)],
            "groups": group_counts + [sum(group_counts)],
        }

    return summary, inputs


def _load_summaries(
    results_dir: Path,
    pattern: str,
    regex: re.Pattern[str],
    cfg: dict[str, Any],
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[Path]]:
    expected = _expected_grid(cfg)
    loaded: dict[tuple[str, str, int], dict[str, Any]] = {}
    inputs: list[Path] = []

    for path in sorted(results_dir.glob(pattern)):
        match = regex.match(path.name)
        if match is None:
            continue
        key = (
            match.group("direction"),
            _arch_label(int(match.group("enc")), int(match.group("dec"))),
            int(match.group("seed")),
        )
        if key in loaded:
            raise PaperTableError(f"Duplicate summary for {key}: {path}")
        loaded[key] = _load_json(path)
        inputs.append(path)

    missing = expected - set(loaded)
    extra = set(loaded) - expected
    if missing or extra:
        raise PaperTableError(
            f"Incomplete summary grid for {pattern}. Missing={sorted(missing)}; "
            f"extra={sorted(extra)}"
        )
    return loaded, inputs


def _parameter_counts_and_protocol_checks(
    results_dir: Path,
    cfg: dict[str, Any],
    dataset_summary: dict[str, dict[str, list[int]]],
) -> tuple[dict[str, int], list[Path]]:
    base, base_paths = _load_summaries(
        results_dir,
        "base_training_summary_*.json",
        _BASE_SUMMARY_RE,
        cfg,
    )
    ft, ft_paths = _load_summaries(
        results_dir,
        "finetuning_summary_*.json",
        _FT_SUMMARY_RE,
        cfg,
    )
    control, control_paths = _load_summaries(
        results_dir,
        "continued_control_summary_*.json",
        _CT_SUMMARY_RE,
        cfg,
    )

    counts_by_arch: dict[str, set[int]] = defaultdict(set)
    expected_base_train, expected_base_val = dataset_summary["biblical"]["pairs"][:2]
    expected_ft_train, expected_ft_val = dataset_summary["conversational"]["pairs"][:2]
    expected_ft_train_dialogues, expected_ft_val_dialogues = (
        dataset_summary["conversational"]["groups"][:2]
    )

    for key, summary in base.items():
        _, arch, _ = key
        counts_by_arch[arch].add(int(summary["model"]["total_parameters"]))
        dataset = summary["dataset"]
        if int(dataset["train_pairs"]) != expected_base_train:
            raise PaperTableError(f"Base train-pair mismatch for {key}.")
        if int(dataset["validation_pairs"]) != expected_base_val:
            raise PaperTableError(f"Base validation-pair mismatch for {key}.")

    for key, summary in ft.items():
        dataset = summary["dataset"]
        if int(dataset["train_pairs"]) != expected_ft_train:
            raise PaperTableError(f"Fine-tuning train-pair mismatch for {key}.")
        if int(dataset["validation_pairs"]) != expected_ft_val:
            raise PaperTableError(f"Fine-tuning validation-pair mismatch for {key}.")
        dialogue_split = dataset["dialogue_split"]
        if int(dialogue_split["train_dialogues"]) != expected_ft_train_dialogues:
            raise PaperTableError(f"Fine-tuning train-dialogue mismatch for {key}.")
        if int(dialogue_split["validation_dialogues"]) != expected_ft_val_dialogues:
            raise PaperTableError(f"Fine-tuning validation-dialogue mismatch for {key}.")

    for key, summary in control.items():
        subset = summary["control_subset"]
        optimization = summary["optimization"]
        ft_summary = ft[key]
        budget = ft_summary["optimization_budget"]

        if int(subset["subset_pairs"]) != expected_ft_train:
            raise PaperTableError(f"Control subset-size mismatch for {key}.")
        if int(subset["matched_dialogue_train_pairs"]) != expected_ft_train:
            raise PaperTableError(f"Control matched-size mismatch for {key}.")
        if int(optimization["matched_optimizer_steps"]) != int(
            budget["optimizer_step_at_best_checkpoint"]
        ):
            raise PaperTableError(f"Control optimizer-step mismatch for {key}.")
        if int(optimization["scheduler_horizon_steps"]) != int(
            budget["planned_total_optimizer_steps"]
        ):
            raise PaperTableError(f"Control scheduler-horizon mismatch for {key}.")

    parameter_counts: dict[str, int] = {}
    for arch, counts in counts_by_arch.items():
        if len(counts) != 1:
            raise PaperTableError(
                f"Parameter count is not constant across runs for {arch}: {sorted(counts)}"
            )
        parameter_counts[arch] = next(iter(counts))

    expected_architectures = set(_architecture_order(cfg))
    if set(parameter_counts) != expected_architectures:
        raise PaperTableError(
            "Parameter-count architecture set does not match experiment grid."
        )

    return parameter_counts, base_paths + ft_paths + control_paths


def _load_metric_runs(
    results_dir: Path,
    cfg: dict[str, Any],
) -> tuple[dict[tuple[str, str, str, int], dict[str, Any]], list[Path]]:
    expected_grid = _expected_grid(cfg)
    runs: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    inputs: list[Path] = []

    for pattern in ("base_test_metrics_*.json", "finetuned_test_metrics_*.json"):
        for path in sorted(results_dir.glob(pattern)):
            match = _METRIC_FILE_RE.match(path.name)
            if match is None:
                continue
            stage = match.group("stage")
            direction = match.group("direction")
            arch = _arch_label(int(match.group("enc")), int(match.group("dec")))
            seed = int(match.group("seed"))
            key = (stage, direction, arch, seed)
            if key in runs:
                raise PaperTableError(f"Duplicate metric run: {key}")
            runs[key] = _load_json(path)
            inputs.append(path)

    for stage in ("base", "finetuned"):
        actual = {
            (direction, arch, seed)
            for s, direction, arch, seed in runs
            if s == stage
        }
        if actual != expected_grid:
            missing = expected_grid - actual
            extra = actual - expected_grid
            raise PaperTableError(
                f"Incomplete {stage} metric grid. Missing={sorted(missing)}; "
                f"extra={sorted(extra)}"
            )

    return runs, inputs


def _aggregate_translation_performance(
    runs: dict[tuple[str, str, str, int], dict[str, Any]],
    cfg: dict[str, Any],
    dataset_summary: dict[str, dict[str, list[int]]],
) -> dict[tuple[str, str, str, str], dict[str, tuple[float, float]]]:
    values: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"chrf++": [], "bleu": []}
    )
    expected_test_pairs = {
        "biblical": dataset_summary["biblical"]["pairs"][2],
        "conversational": dataset_summary["conversational"]["pairs"][2],
    }

    for (stage, direction, arch, seed), run in runs.items():
        if int(run["experiment_seed"]) != seed:
            raise PaperTableError(f"Seed metadata mismatch for {(stage, direction, arch, seed)}")

        for domain, result_key in (
            ("biblical", "biblical_results"),
            ("conversational", "conversational_results"),
        ):
            result = run[result_key]
            if int(result["test_pairs"]) != expected_test_pairs[domain]:
                raise PaperTableError(
                    f"Test-pair mismatch for {(stage, direction, arch, seed, domain)}"
                )
            bleu = _require_finite(float(result["bleu_score"]), "BLEU")
            chrf = _require_finite(float(result["chrf_score"]), "chrF++")
            values[(stage, direction, arch, domain)]["bleu"].append(bleu)
            values[(stage, direction, arch, domain)]["chrf++"].append(chrf)

    expected_seeds = len(cfg["experiment_seeds"])
    aggregate: dict[tuple[str, str, str, str], dict[str, tuple[float, float]]] = {}
    for key, metrics in values.items():
        if any(len(metric_values) != expected_seeds for metric_values in metrics.values()):
            raise PaperTableError(f"Expected {expected_seeds} seeds for {key}.")
        aggregate[key] = {
            metric: _sample_mean_sd(metric_values)
            for metric, metric_values in metrics.items()
        }
    return aggregate


def _load_primary_bootstrap_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], list[Path]]:
    if not path.is_file():
        raise FileNotFoundError(f"Bootstrap summary not found: {path}")
    rows = _read_csv_dicts(path)
    if len(rows) != 64:
        raise PaperTableError(
            f"Expected 64 metric rows in final bootstrap summary; found {len(rows)}."
        )

    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        metric = row["metric"]
        if metric not in {"bleu", "chrf++"}:
            raise PaperTableError(f"Unexpected bootstrap metric: {metric}")
        grouped[row["contrast_id"]][metric] = row

    selected_ids: list[str] = []
    # Adaptation and retention: compact reference architecture only.
    for direction in ("aym_to_spa", "spa_to_aym"):
        for family in ("adaptation_vs_base", "adaptation_vs_continued_control"):
            for domain in ("biblical", "conversational"):
                selected_ids.append(f"{family}|{direction}|4E-4D|{domain}")

    # Architecture contrasts: all prespecified comparator-vs-reference contrasts.
    architecture_rows = [
        contrast_id
        for contrast_id in grouped
        if contrast_id.startswith("architecture_vs_reference|")
    ]

    stage_rank = {"base": 0, "finetuned": 1}
    direction_rank = {"aym_to_spa": 0, "spa_to_aym": 1}
    arch_rank = {"6E-2D": 0, "2E-6D": 1}

    def architecture_sort_key(contrast_id: str) -> tuple[int, int, int]:
        sample = grouped[contrast_id]["chrf++"]
        return (
            stage_rank[sample["system_b_stage"]],
            direction_rank[sample["direction"].replace("->", "_to_")],
            arch_rank[sample["system_b_architecture"]],
        )

    selected_ids.extend(sorted(architecture_rows, key=architecture_sort_key))

    if len(selected_ids) != 16 or len(set(selected_ids)) != 16:
        raise PaperTableError(
            f"Expected 16 selected manuscript bootstrap contrasts; got {len(selected_ids)}."
        )

    output: list[dict[str, Any]] = []
    for contrast_id in selected_ids:
        if contrast_id not in grouped:
            raise PaperTableError(f"Missing bootstrap contrast: {contrast_id}")
        metric_rows = grouped[contrast_id]
        if set(metric_rows) != {"bleu", "chrf++"}:
            raise PaperTableError(f"Missing BLEU/chrF++ pair for {contrast_id}.")

        sample = metric_rows["chrf++"]
        family = sample["contrast_family"]
        if family == "adaptation_vs_base":
            comparison = "FT - Base (4E-4D)"
            condition = sample["domain"].capitalize()
        elif family == "adaptation_vs_continued_control":
            comparison = "FT - Control (4E-4D)"
            condition = sample["domain"].capitalize()
        elif family == "architecture_vs_4E4D_reference":
            comparison = f"{sample['system_b_architecture']} - 4E-4D"
            condition = (
                "Base/Biblical"
                if sample["system_b_stage"] == "base"
                else "FT/Conversational"
            )
        else:
            raise PaperTableError(f"Unexpected contrast family: {family}")

        record: dict[str, Any] = {
            "family": family,
            "comparison": comparison,
            "direction": sample["direction"],
            "condition": condition,
        }
        for metric in ("bleu", "chrf++"):
            row = metric_rows[metric]
            delta = _require_finite(float(row["observed_delta_b_minus_a"]), "delta")
            low = _require_finite(float(row["ci_low"]), "ci_low")
            high = _require_finite(float(row["ci_high"]), "ci_high")
            if not low <= delta <= high:
                raise PaperTableError(f"Bootstrap CI does not contain delta: {contrast_id}/{metric}")
            if float(row["confidence_level"]) != 0.95:
                raise PaperTableError("Bootstrap confidence level is not 0.95.")
            if int(row["bootstrap_replicates"]) != 5000:
                raise PaperTableError("Bootstrap replicate count is not 5000.")
            record[metric] = (delta, low, high)
        output.append(record)

    return output, [path]


def _load_inference_summary(path: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    if not path.is_file():
        raise FileNotFoundError(f"Inference summary not found: {path}")
    rows = _read_csv_dicts(path)
    if len(rows) != 12:
        raise PaperTableError(f"Expected 12 inference summary rows; found {len(rows)}.")

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (row["stage"], row["domain"], row["direction"], row["architecture"])
        if key in seen:
            raise PaperTableError(f"Duplicate inference summary row: {key}")
        seen.add(key)

        if (row["stage"], row["domain"]) not in {
            ("base", "biblical"),
            ("finetuned", "conversational"),
        }:
            raise PaperTableError(f"Unexpected inference condition: {key[:2]}")
        if int(row["n_seeds"]) != 3 or row["seeds"] != "111;222;333":
            raise PaperTableError(f"Unexpected seed summary in inference row: {key}")

        converted: dict[str, Any] = dict(row)
        for field in (
            "latency_ms_per_sentence_mean",
            "latency_ms_per_sentence_sd",
            "generated_tokens_per_second_mean",
            "generated_tokens_per_second_sd",
            "peak_gpu_memory_mib_mean",
            "peak_gpu_memory_mib_sd",
        ):
            converted[field] = _require_finite(float(row[field]), field)
        output.append(converted)

    expected_arch = {"4E-4D", "6E-2D", "2E-6D"}
    for stage, domain in (("base", "biblical"), ("finetuned", "conversational")):
        for direction in ("aym->spa", "spa->aym"):
            actual = {
                row["architecture"]
                for row in output
                if row["stage"] == stage
                and row["domain"] == domain
                and row["direction"] == direction
            }
            if actual != expected_arch:
                raise PaperTableError(
                    f"Incomplete inference architecture set for {(stage, domain, direction)}"
                )

    return output, [path]


def _render_table_1(summary: dict[str, dict[str, list[int]]]) -> str:
    b = summary["biblical"]
    c = summary["conversational"]
    rows = [
        ("Biblical", "Parallel pairs", b["pairs"]),
        ("Biblical", "USFM verses", b["groups"]),
        ("Conversational", "Parallel pairs", c["pairs"]),
        ("Conversational", "Dialogues", c["groups"]),
    ]
    body = "\n".join(
        f"{corpus} & {unit} & {values[0]:,} & {values[1]:,} & {values[2]:,} & {values[3]:,} \\\\"
        for corpus, unit, values in rows
    )
    return rf"""% AUTO-GENERATED by 19_generate_paper_tables.py. Do not edit manually.
\begin{{table}}[!t]
\centering
\caption{{Corpus partitions under grouped leakage-aware splitting.}}
\label{{tab:dataset_partitions}}
\small
\setlength{{\tabcolsep}}{{3.4pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrr}}
\hline
Corpus & Unit & Train & Validation & Test & Total \\
\hline
{body}
\hline
\end{{tabular}}%
}}
\end{{table}}
"""


def _render_table_2(
    cfg: dict[str, Any],
    parameter_counts: dict[str, int],
    dataset_summary: dict[str, dict[str, list[int]]],
) -> str:
    architecture_order = _architecture_order(cfg)
    reference = parameter_counts["4E-4D"]
    arch_rows = []
    for arch in architecture_order:
        enc, dec = [int(value) for value in re.findall(r"\d+", arch)]
        params = parameter_counts[arch]
        if arch == "4E-4D":
            delta = "--"
        else:
            pct = 100.0 * (params - reference) / reference
            delta = f"{pct:+.2f}\\%"
        arch_rows.append(
            f"{_latex_arch(arch)} & {enc} & {dec} & {params:,} & {delta} \\\\"
        )

    base_pairs = dataset_summary["biblical"]["pairs"][0]
    ft_pairs = dataset_summary["conversational"]["pairs"][0]
    protocol_rows = [
        ("Optimization data", "Biblical", "Conversational", "Biblical subset"),
        ("Training pairs", f"{base_pairs:,}", f"{ft_pairs:,}", f"{ft_pairs:,}"),
        (
            "Budget",
            f"{int(cfg['base_epochs'])} epochs",
            f"up to {int(cfg['ft_epochs'])} epochs",
            "matched selected FT steps",
        ),
        (
            "CUDA batch size",
            str(cfg["base_batch_size_cuda"]),
            str(cfg["ft_batch_size_cuda"]),
            str(cfg["ct_batch_size_cuda"]),
        ),
        ("Optimizer", "AdamW", "AdamW", "AdamW"),
        (
            "Learning rate",
            _fmt_sci_latex(float(cfg['base_max_learning_rate'])) + " (max)",
            _fmt_sci_latex(float(cfg['ft_learning_rate'])) + " (initial)",
            _fmt_sci_latex(float(cfg['ct_learning_rate'])) + " (initial)",
        ),
        ("Scheduler", "OneCycleLR", "CosineAnnealingLR", "same cosine prefix as FT"),
        (
            "Checkpoint policy",
            "best biblical val chrF++",
            "best conversational val chrF++",
            "exact matched FT optimizer step",
        ),
    ]
    protocol_body = "\n".join(
        f"{setting} & {base} & {ft} & {control} \\\\"
        for setting, base, ft, control in protocol_rows
    )

    return rf"""% AUTO-GENERATED by 19_generate_paper_tables.py. Do not edit manually.
\begin{{table*}}[!t]
\centering
\caption{{Transformer depth allocations and training protocol. All architectures use $d_{{model}}=256$, 8 attention heads, $d_{{ff}}=1024$, dropout 0.2, shared source/target embeddings, and tied output projection.}}
\label{{tab:model_training_protocol}}
\small
\renewcommand{{\arraystretch}}{{1.05}}
\textbf{{(a) Fixed eight-layer depth allocations}}\\[2pt]
\begin{{tabular}}{{lrrrr}}
\hline
Architecture & Encoder & Decoder & Parameters & $\Delta$ vs. 4E--4D \\
\hline
{chr(10).join(arch_rows)}
\hline
\end{{tabular}}

\vspace{{0.35em}}
\textbf{{(b) Optimization protocol}}\\[2pt]
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llll}}
\hline
Setting & Base training & Fine-tuning & Continued-training control \\
\hline
{protocol_body}
\hline
\end{{tabular}}%
}}
\end{{table*}}
"""


def _render_table_3(
    aggregate: dict[tuple[str, str, str, str], dict[str, tuple[float, float]]],
    cfg: dict[str, Any],
) -> str:
    rows: list[str] = []
    for direction in _direction_order(cfg):
        for arch in _architecture_order(cfg):
            cells = [_direction_arrow(direction), _latex_arch(arch)]
            for stage, domain in (
                ("base", "biblical"),
                ("base", "conversational"),
                ("finetuned", "biblical"),
                ("finetuned", "conversational"),
            ):
                metrics = aggregate[(stage, direction, arch, domain)]
                cells.append(_fmt_mean_sd(*metrics["chrf++"]))
                cells.append(_fmt_mean_sd(*metrics["bleu"]))
            rows.append(" & ".join(cells) + r" \\")
        if direction != _direction_order(cfg)[-1]:
            rows.append(r"\hline")

    return rf"""% AUTO-GENERATED by 19_generate_paper_tables.py. Do not edit manually.
\begin{{table*}}[!t]
\centering
\caption{{Translation performance before and after conversational fine-tuning. Values are mean $\pm$ sample standard deviation across seeds 111, 222, and 333.}}
\label{{tab:translation_performance}}
\scriptsize
\setlength{{\tabcolsep}}{{2.7pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrrrr}}
\hline
& & \multicolumn{{4}}{{c}}{{Base model}} & \multicolumn{{4}}{{c}}{{Fine-tuned model}} \\
\cline{{3-10}}
& & \multicolumn{{2}}{{c}}{{Biblical}} & \multicolumn{{2}}{{c}}{{Conversational}} & \multicolumn{{2}}{{c}}{{Biblical}} & \multicolumn{{2}}{{c}}{{Conversational}} \\
Direction & Architecture & chrF++ & BLEU & chrF++ & BLEU & chrF++ & BLEU & chrF++ & BLEU \\
\hline
{chr(10).join(rows)}
\hline
\end{{tabular}}%
}}
\end{{table*}}
"""


def _render_table_4(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    previous_family: str | None = None
    for row in rows:
        if previous_family is not None and row["family"] != previous_family:
            rendered.append(r"\hline")
        rendered.append(
            " & ".join(
                [
                    (
                        row["comparison"]
                        .replace("4E-4D", "4E--4D")
                        .replace("6E-2D", "6E--2D")
                        .replace("2E-6D", "2E--6D")
                        .replace("FT - Base", r"FT $-$ Base")
                        .replace("FT - Control", r"FT $-$ Control")
                        .replace(" - 4E--4D", r" $-$ 4E--4D")
                    ),
                    _direction_arrow(row["direction"]),
                    row["condition"],
                    _fmt_delta_ci(*row["bleu"]),
                    _fmt_delta_ci(*row["chrf++"]),
                ]
            )
            + r" \\"
        )
        previous_family = row["family"]

    return rf"""% AUTO-GENERATED by 19_generate_paper_tables.py. Do not edit manually.
\begin{{table*}}[!t]
\centering
\caption{{Selected prespecified paired-bootstrap contrasts reported in the manuscript. Each cell reports the observed mean metric difference across three seeds followed by its paired 95\% percentile confidence interval. For adaptation rows, $\Delta$ is fine-tuned minus the comparison system; for architecture rows, $\Delta$ is the asymmetric model minus 4E--4D. The complete set of prespecified bootstrap contrasts is available in the public repository.}}
\label{{tab:bootstrap_primary}}
\scriptsize
\setlength{{\tabcolsep}}{{3.2pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{tabular}}{{lllcc}}
\hline
Comparison & Direction & Condition/domain & $\Delta$ BLEU [95\% CI] & $\Delta$ chrF++ [95\% CI] \\
\hline
{chr(10).join(rendered)}
\hline
\end{{tabular}}
\end{{table*}}
"""


def _render_table_5(rows: list[dict[str, Any]]) -> str:
    condition_rank = {("base", "biblical"): 0, ("finetuned", "conversational"): 1}
    direction_rank = {"aym->spa": 0, "spa->aym": 1}
    arch_rank = {"4E-4D": 0, "6E-2D": 1, "2E-6D": 2}
    rows = sorted(
        rows,
        key=lambda row: (
            condition_rank[(row["stage"], row["domain"])],
            direction_rank[row["direction"]],
            arch_rank[row["architecture"]],
        ),
    )

    rendered: list[str] = []
    previous_condition: tuple[str, str] | None = None
    for row in rows:
        condition = (row["stage"], row["domain"])
        if previous_condition is not None and condition != previous_condition:
            rendered.append(r"\hline")
        condition_label = (
            "Base/Biblical"
            if condition == ("base", "biblical")
            else "FT/Conversational"
        )
        rendered.append(
            " & ".join(
                [
                    condition_label,
                    _direction_arrow(row["direction"]),
                    _latex_arch(row["architecture"]),
                    _fmt_mean_sd(
                        row["latency_ms_per_sentence_mean"],
                        row["latency_ms_per_sentence_sd"],
                    ),
                    _fmt_mean_sd(
                        row["generated_tokens_per_second_mean"],
                        row["generated_tokens_per_second_sd"],
                    ),
                    _fmt_mean_sd(
                        row["peak_gpu_memory_mib_mean"],
                        row["peak_gpu_memory_mib_sd"],
                    ),
                ]
            )
            + r" \\"
        )
        previous_condition = condition

    return rf"""% AUTO-GENERATED by 19_generate_paper_tables.py. Do not edit manually.
\begin{{table*}}[!t]
\centering
\caption{{Inference efficiency under batch-size-1 FP32 greedy decoding with a maximum generation length of 150 tokens. Values are mean $\pm$ sample standard deviation across three model seeds on an NVIDIA GeForce RTX 2050.}}
\label{{tab:inference_efficiency}}
\scriptsize
\setlength{{\tabcolsep}}{{3.4pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{tabular}}{{lllccc}}
\hline
Condition & Direction & Architecture & Latency (ms/sentence) & Generated tokens/s & Peak VRAM (MiB) \\
\hline
{chr(10).join(rendered)}
\hline
\end{{tabular}}
\end{{table*}}
"""


def _manifest(
    project_root: Path,
    cfg: dict[str, Any],
    input_paths: Iterable[Path],
    output_contents: dict[str, str],
) -> dict[str, Any]:
    unique_inputs = sorted({path.resolve() for path in input_paths})
    return {
        "generator": "19_generate_paper_tables.py",
        "generator_version": GENERATOR_VERSION,
        "experiment_seeds": [int(seed) for seed in cfg["experiment_seeds"]],
        "directions": [
            _direction_key(item["src"], item["tgt"])
            for item in cfg["experiment_directions"]
        ],
        "architectures": _architecture_order(cfg),
        "bootstrap_table_policy": {
            "adaptation_architecture": "4E-4D",
            "adaptation_families": [
                "fine-tuned minus base",
                "fine-tuned minus continued-training control",
            ],
            "adaptation_domains": ["biblical", "conversational"],
            "architecture_reference": "4E-4D",
            "architecture_conditions": ["base/biblical", "finetuned/conversational"],
            "full_bootstrap_rows_remain_available": "results/bootstrap/bootstrap_summary.csv",
        },
        "inputs": {
            _relative_to_root(path, project_root): _sha256(path)
            for path in unique_inputs
        },
        "outputs": {
            filename: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for filename, content in sorted(output_contents.items())
        },
    }


def _write_outputs(
    output_dir: Path,
    outputs: dict[str, str],
    manifest: dict[str, Any],
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_paths = [output_dir / name for name in outputs]
    manifest_path = output_dir / "table_generation_manifest.json"
    target_paths.append(manifest_path)

    existing = [path for path in target_paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Paper-table outputs already exist. Re-run with --force only after "
            "confirming that regeneration is intentional. Existing: "
            + ", ".join(str(path) for path in existing)
        )

    for filename, content in outputs.items():
        (output_dir / filename).write_text(content, encoding="utf-8", newline="\n")

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_paper_tables(
    config_path: str | Path,
    output_dir: str | Path,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Validate final experimental artifacts and generate the five LaTeX tables
    used by the IJ-AI manuscript.

    The generator intentionally reads only numerical/metadata result artifacts
    plus the six local split TSV files required to count grouped partitions. It
    never reads or writes prediction TSV files and never exports corpus text.
    """
    config_path = Path(config_path).resolve()
    project_root = config_path.parent
    output_dir = Path(output_dir).resolve()
    cfg = load_config(config_path)
    results_dir = project_root / cfg["results_dir"]

    dataset_summary, split_inputs = _dataset_partition_summary(project_root, cfg)
    parameter_counts, summary_inputs = _parameter_counts_and_protocol_checks(
        results_dir,
        cfg,
        dataset_summary,
    )
    metric_runs, metric_inputs = _load_metric_runs(results_dir, cfg)
    performance = _aggregate_translation_performance(
        metric_runs,
        cfg,
        dataset_summary,
    )
    bootstrap_rows, bootstrap_inputs = _load_primary_bootstrap_rows(
        results_dir / "bootstrap" / "bootstrap_summary.csv"
    )
    inference_rows, inference_inputs = _load_inference_summary(
        results_dir / "inference_benchmark" / "inference_benchmark_summary.csv"
    )

    outputs = {
        "table_1_dataset_partitions.tex": _render_table_1(dataset_summary),
        "table_2_model_training_protocol.tex": _render_table_2(
            cfg,
            parameter_counts,
            dataset_summary,
        ),
        "table_3_translation_performance.tex": _render_table_3(performance, cfg),
        "table_4_primary_bootstrap_contrasts.tex": _render_table_4(bootstrap_rows),
        "table_5_inference_efficiency.tex": _render_table_5(inference_rows),
    }

    all_inputs = (
        [config_path]
        + split_inputs
        + summary_inputs
        + metric_inputs
        + bootstrap_inputs
        + inference_inputs
    )
    manifest = _manifest(project_root, cfg, all_inputs, outputs)

    print("=" * 80)
    print("IJ-AI PAPER TABLE GENERATION")
    print("=" * 80)
    print(f"Config                  : {config_path}")
    print(f"Output directory        : {output_dir}")
    print(f"Model seeds             : {cfg['experiment_seeds']}")
    print(f"Table 1 data rows       : 4")
    print(f"Table 2 architectures   : {len(parameter_counts)}")
    print(f"Table 3 model rows      : {len(_direction_order(cfg)) * len(_architecture_order(cfg))}")
    print(f"Table 4 contrasts       : {len(bootstrap_rows)}")
    print(f"Table 5 benchmark rows  : {len(inference_rows)}")
    print(f"Source files validated  : {len(set(path.resolve() for path in all_inputs))}")
    print(f"Dry run                 : {dry_run}")
    print("=" * 80)

    if dry_run:
        for filename in _TABLE_FILES:
            print(f"[DRY RUN] would write {output_dir / filename}")
        print(f"[DRY RUN] would write {output_dir / 'table_generation_manifest.json'}")
        return manifest

    _write_outputs(output_dir, outputs, manifest, force=force)
    for filename in _TABLE_FILES:
        print(f"[Saved] {output_dir / filename}")
    print(f"[Saved] {output_dir / 'table_generation_manifest.json'}")
    return manifest
