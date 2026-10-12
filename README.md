# nmt-aym-spa

Reproducible research code for low-resource neural machine translation (NMT) between **Aymara (aym)** and **Spanish (spa)**.

The project implements a compact Transformer encoder-decoder pipeline trained from scratch and studies three controlled factors:

1. **translation direction**: Aymara→Spanish and Spanish→Aymara;
2. **encoder-decoder depth allocation** under a fixed total depth of eight Transformer layers;
3. **domain adaptation** from a biblical source domain to a pedagogical conversational target domain.

The repository also includes a size-matched continued-training control and paired bootstrap confidence intervals for prespecified system comparisons.

> **Repository status.** The code path has been validated end to end. The definitive experimental results, exact Git commit/tag, execution evidence, and final result summaries will be added after the final paper runs are completed.

## Research scope

This repository is intended to support a **controlled characterization study of compact Transformers trained from scratch** for Aymara–Spanish NMT. It is not designed as a state-of-the-art benchmark against large pretrained multilingual translation systems.

The configured model grid contains three encoder-decoder depth allocations while preserving a fixed total depth of eight layers:

| Architecture | Encoder layers | Decoder layers |
|---|---:|---:|
| 4E–4D | 4 | 4 |
| 6E–2D | 6 | 2 |
| 2E–6D | 2 | 6 |

Because encoder and decoder layers do not have identical parameter costs, these configurations should be interpreted as **fixed total-depth / fixed eight-layer-budget allocations**, not as exactly parameter-matched models.

The definitive experiment grid uses:

- model seeds: `111`, `222`, `333`;
- translation directions: `aym -> spa`, `spa -> aym`;
- architectures: `4E-4D`, `6E-2D`, `2E-6D`;
- base training domain: biblical parallel text;
- adaptation domain: pedagogical Aymara–Spanish conversations.

## Data availability

The research datasets are **not redistributed in this repository**.

The code required to reconstruct the datasets is provided. Users must obtain the original source materials independently and comply with the corresponding rights statements, licenses, website terms, and access conditions.

Detailed source information and reconstruction instructions are provided in:

- [`data/README.md`](data/README.md)

In brief:

- the biblical corpus is reconstructed from four Bible.com editions identified by the IDs in `config.yaml`;
- the conversational corpus is reconstructed from *AYMARA ARUSKIPAWINAKA: Conversaciones en aimara* after downloading the source PDF from its public source page.

## Repository structure

```text
nmt-aym-spa/
├── 1_scrap_bibles.py
├── 2_scrap_book.py
├── 3_align_verses.py
├── 4_preprocess_bible.py
├── 5_preprocess_book.py
├── 6_split_bible.py
├── 7_split_book.py
├── 8_train_tokenizer.py
├── 9_encode_bible.py
├── 10_encode_book.py
├── 11_run_base_training.py
├── 12_run_base_evaluation.py
├── 13_run_finetuning.py
├── 14_run_finetuned_evaluation.py
├── 15_run_continued_training_control.py
├── 16_run_continued_control_evaluation.py
├── 17_bootstrap_analysis.py
├── 18_benchmark_inference.py
├── config.yaml
├── requirements.txt
├── LICENSE
├── data/
│   └── README.md
├── src/
│   ├── analysis/
│   ├── benchmarking/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── tokenization/
│   ├── training/
│   └── utils/
└── tokenizer/
    ├── SentencePiece.model
    └── SentencePiece.vocab
```

Generated corpora, checkpoints, smoke-test artifacts, prediction TSV files, and internal simulation outputs are intentionally excluded from version control.

## Environment

The end-to-end pipeline was validated in the following local environment:

- Python `3.10.21`;
- PyTorch `2.5.1+cu121`;
- CUDA runtime used by PyTorch: `12.1`;
- SacreBLEU `2.6.0`;
- SentencePiece `0.2.2`;
- NumPy `2.2.6`;
- pandas `2.3.3`;
- PyYAML `6.0.3`;
- Windows 11;
- NVIDIA GeForce RTX 2050, 4 GB VRAM.

The definitive paper runs may be executed on different compatible hardware. Each training/evaluation artifact records environment, configuration, hashes, and Git metadata when available.

### Installation

A clean Python 3.10 environment is recommended.

```bash
python -m venv .venv
```

Activate the environment using the command appropriate for your operating system and then install the direct project dependencies:

```bash
pip install -r requirements.txt
```

For GPU execution, install a PyTorch build compatible with the CUDA stack available on the target system. The validated development environment used PyTorch `2.5.1+cu121`.

## Configuration

All experimental settings are centralized in:

```text
config.yaml
```

The current definitive configuration specifies:

- data split seed: `1989`;
- model seeds: `111`, `222`, `333`;
- Transformer depth allocations: `4E-4D`, `6E-2D`, `2E-6D`;
- shared SentencePiece BPE vocabulary size: `8000`;
- base training epochs: `150`;
- conversational fine-tuning epochs: `30`;
- greedy decoding with maximum generation length `150`;
- paired bootstrap confidence intervals with `5000` replicates and 95% confidence level;
- post-hoc inference benchmarking with batch size `1`, FP32, greedy decoding, and `max_len=150`.

Do not modify the definitive configuration while final experimental runs are in progress if exact Git/configuration provenance is required.

## Data preparation

After obtaining the external source materials described in [`data/README.md`](data/README.md), run the data pipeline in order:

```bash
python 1_scrap_bibles.py
python 2_scrap_book.py
python 3_align_verses.py
python 4_preprocess_bible.py
python 5_preprocess_book.py
python 6_split_bible.py
python 7_split_book.py
python 8_train_tokenizer.py
python 9_encode_bible.py
python 10_encode_book.py
```

The tokenizer is trained **only on the biblical training partition**, using both languages from that partition. The conversational corpus is excluded from tokenizer training to preserve source-domain/target-domain separation.

The tracked tokenizer is a SentencePiece BPE model with:

- vocabulary size: `8000`;
- model type: `bpe`;
- character coverage: `1.0`;
- user-defined symbols: `0123456789/`;
- `split_digits=True`.

Tracked tokenizer SHA-256:

```text
SentencePiece.model
5508521388f095f47a905b24586e647b703a780805d80c0bc043256a3ca1cf21
```

## Experimental workflow

### 1. Base training

```bash
python 11_run_base_training.py
```

The runner executes the configured Cartesian grid of directions, architectures, and model seeds. The definitive configuration contains 18 base runs.

The base model is trained on the biblical training partition. Checkpoint selection uses validation `chrF++`.

A planning-only check can be run with:

```bash
python 11_run_base_training.py --dry-run
```

### 2. Base-model evaluation

```bash
python 12_run_base_evaluation.py
```

Each selected base checkpoint is evaluated on:

- the biblical test set;
- the conversational test set.

### 3. Conversational fine-tuning

```bash
python 13_run_finetuning.py
```

Fine-tuning:

- starts from the corresponding selected base checkpoint;
- updates the full model;
- uses only the conversational training partition for optimization;
- uses only the conversational validation partition for checkpoint selection;
- does not use either test set for model selection.

### 4. Fine-tuned-model evaluation

```bash
python 14_run_finetuned_evaluation.py
```

Each selected adapted checkpoint is evaluated on both test domains.

### 5. Size-matched continued-training control

```bash
python 15_run_continued_training_control.py
```

The control starts from the **same selected base checkpoint** as conversational fine-tuning but continues optimization on a deterministic, size-matched subset of biblical training pairs.

The control matches the selected fine-tuning checkpoint's optimizer-step budget and cosine-scheduler horizon. This provides a control for the effect of additional optimization without conversational-domain training data.

### 6. Continued-control evaluation

```bash
python 16_run_continued_control_evaluation.py
```

The continued-training control is evaluated on both biblical and conversational test sets.

### 7. Paired bootstrap analysis

```bash
python 17_bootstrap_analysis.py
```

The final statistical analysis uses paired bootstrap resampling with:

- `5000` replicates;
- 95% percentile confidence intervals;
- fixed bootstrap seed `1989`;
- SacreBLEU `2.6.0`.

The resampling unit is domain-specific:

- **biblical test set:** cluster bootstrap over `usfm_code`, keeping aligned edition variants of the same underlying verse together;
- **conversational test set:** paired example-level bootstrap over `example_index`.

The analysis uses prespecified contrasts rather than an exhaustive post-hoc all-pairs search.

Model-seed variability and test-set resampling uncertainty are treated separately:

- model-seed variability is summarized across seeds using mean ± sample standard deviation;
- paired bootstrap intervals quantify uncertainty associated with test-set resampling.

### 8. Post-hoc inference benchmark

```bash
python -m unittest tests.test_inference_benchmark -v
python 18_benchmark_inference.py --dry-run
python 18_benchmark_inference.py
```

The benchmark does **not** retrain any model. It loads the already selected
checkpoints and measures only the two prespecified architecture-analysis
conditions:

- **Base → Biblical test**;
- **Fine-tuned → Conversational test**.

For each condition, both translation directions, all three depth allocations,
and seeds `111`, `222`, and `333` are benchmarked, for 36 checkpoint/test
measurements in total.

The definitive protocol uses batch size `1`, FP32, greedy decoding,
`max_len=150`, 10 warm-up examples, and one complete measured test-set pass per
checkpoint. Source tensors are materialized from the tokenized TSV files before
timing. Architecture order is deterministically rotated across seeds so each
architecture occupies the first, middle, and last timing position once within
each stage/direction. Immediately before the measured pass, CUDA is synchronized
and peak allocated-memory statistics are reset. CUDA is synchronized again
before the wall-clock timer is stopped.

The timed region includes CPU-to-GPU transfer of each already-tokenized source
tensor, encoder computation, and autoregressive greedy decoding. It excludes
checkpoint loading, TSV parsing, SentencePiece detokenization, BLEU/chrF++
calculation, and output-file writing.

Only three primary inference quantities are reported:

- latency in milliseconds per sentence;
- generated-token throughput in tokens per second;
- peak allocated CUDA memory in MiB.

Results are summarized as mean ± sample standard deviation across the three
model seeds. The benchmark writes only lightweight numerical metadata (no
reference or hypothesis text):

```text
results/inference_benchmark/inference_benchmark_runs.csv
results/inference_benchmark/inference_benchmark_summary.csv
results/inference_benchmark/inference_benchmark_summary.json
```

For a scientifically comparable run, execute all 36 measurements in the same
process on the same GPU while avoiding other GPU-heavy workloads. Commit/tag the
benchmark code before the definitive run so the output JSON records a clean and
exact Git state.

## Evaluation

All reported final evaluations use greedy autoregressive decoding.

Corpus-level metrics are computed with SacreBLEU `2.6.0`.

### BLEU

- case-sensitive;
- tokenizer: `13a`;
- smoothing: exponential;
- maximum n-gram order: 4;
- effective order: disabled.

Expected signature:

```text
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
```

### chrF++

- character order: 6;
- word order: 2;
- beta: 2;
- case-sensitive;
- whitespace excluded;
- epsilon smoothing disabled.

Expected signature:

```text
nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0
```

## Reproducibility safeguards

The training, evaluation, continued-training, and bootstrap code includes safeguards intended to make the public workflow auditable:

- deterministic experiment seeds;
- grouped data splitting to prevent cross-partition leakage;
- input/output SHA-256 hashes stored in result metadata;
- explicit optimization and decoding configuration;
- compatibility checks before skipping existing artifacts;
- Git commit, branch, exact tag, and dirty-worktree metadata when the project is run from a Git repository;
- per-example prediction exports used locally for paired statistical analysis.

Prediction TSV files are not intended for public versioning because they include reference text derived from the external research corpora.

## Results

The definitive paper results are intentionally not hard-coded into this README yet.

After the final experiment grid is completed, this section will be updated with:

- the exact public Git commit and release tag;
- the definitive execution environment;
- final aggregate BLEU and chrF++ results;
- seed variability summaries;
- paired bootstrap 95% confidence intervals;
- links to the lightweight JSON/CSV result artifacts used to support the paper tables.

## License

The source code in this repository is released under the MIT License. See [`LICENSE`](LICENSE).

**Important:** the MIT License applies to the repository's original source code only. It does not grant rights to Bible.com content, the conversational source book, reconstructed datasets, or other third-party materials. Those materials remain subject to their own rights statements, licenses, and terms.

## Citation

A formal citation will be added when the associated paper is published or publicly archived.
