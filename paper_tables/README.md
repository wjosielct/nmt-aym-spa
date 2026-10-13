# Generated IJ-AI paper tables

This directory is the default output of:

```bash
python 19_generate_paper_tables.py
```

The generator validates the frozen local dataset splits and the final public
result artifacts before producing five LaTeX tables:

1. `table_1_dataset_partitions.tex` — pair and grouping-unit counts for the
   biblical and conversational partitions;
2. `table_2_model_training_protocol.tex` — fixed eight-layer allocations,
   parameter counts, and Base/Fine-tuning/Continued-control protocol;
3. `table_3_translation_performance.tex` — Base and Fine-tuned BLEU/chrF++
   results on both domains, reported as mean ± sample SD across three seeds;
4. `table_4_primary_bootstrap_contrasts.tex` — a selected set of 16 prespecified
   paired-bootstrap contrasts reported compactly in the manuscript; the complete
   bootstrap output remains available under `results/bootstrap/`;
5. `table_5_inference_efficiency.tex` — latency, generated-token throughput,
   and peak allocated GPU memory.

A `table_generation_manifest.json` file is generated alongside the tables. It
records SHA-256 hashes for every numerical/source input used and for every
produced `.tex` file.

The generator **never reads prediction TSV files** and does not export corpus
text. Table 1 reads only the grouping columns (`usfm_code` and `dialogue_id`)
from the six split files in order to count pairs/groups and verify that grouping
units do not cross train/validation/test boundaries.

Before definitive generation, run:

```bash
python -m unittest tests.test_paper_tables -v
python 19_generate_paper_tables.py --dry-run
```

Then, after reviewing the dry run:

```bash
python 19_generate_paper_tables.py
```

If the files already exist, the script stops rather than overwriting them.
Regeneration must be explicit:

```bash
python 19_generate_paper_tables.py --force
```

## Overleaf integration

The generated tables use only LaTeX facilities/packages already present in the
current IJ-AI manuscript (`graphicx` is used for `\resizebox`). Copy the
`paper_tables/` directory into the Overleaf project and include the tables at
the desired locations:

```latex
\input{paper_tables/table_1_dataset_partitions.tex}
\input{paper_tables/table_2_model_training_protocol.tex}
\input{paper_tables/table_3_translation_performance.tex}
\input{paper_tables/table_4_primary_bootstrap_contrasts.tex}
\input{paper_tables/table_5_inference_efficiency.tex}
```

Do not edit the generated numeric tables manually. If source results change,
regenerate them from the repository artifacts instead.
