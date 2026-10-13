# Generated LaTeX result tables

This directory is the default output of:

```bash
python 19_generate_paper_tables.py
```

The generator validates the local dataset splits and public result artifacts before producing five LaTeX tables:

1. `table_1_dataset_partitions.tex` — pair and grouping-unit counts for the biblical and conversational partitions;
2. `table_2_model_training_protocol.tex` — fixed eight-layer allocations, parameter counts, and Base/Fine-tuning/Continued-control protocol;
3. `table_3_translation_performance.tex` — Base and Fine-tuned BLEU/chrF++ results on both domains, reported as mean ± sample SD across three seeds;
4. `table_4_primary_bootstrap_contrasts.tex` — a selected set of 16 prespecified paired-bootstrap contrasts; the complete bootstrap output remains available under `results/bootstrap/`;
5. `table_5_inference_efficiency.tex` — latency, generated-token throughput, and peak allocated GPU memory.

A `table_generation_manifest.json` file is generated alongside the tables. It records SHA-256 hashes for every numerical/source input used and for every produced `.tex` file.

The generator **never reads prediction TSV files** and does not export corpus text. Table 1 reads only the grouping columns (`usfm_code` and `dialogue_id`) from the six split files in order to count pairs/groups and verify that grouping units do not cross train/validation/test boundaries.

Validate the inputs and generation plan with:

```bash
python -m unittest tests.test_paper_tables -v
python 19_generate_paper_tables.py --dry-run
```

Generate the tables with:

```bash
python 19_generate_paper_tables.py
```

If the files already exist, the script stops rather than overwriting them. Intentional regeneration must be explicit:

```bash
python 19_generate_paper_tables.py --force
```

Generated numeric tables should not be edited manually. If source result artifacts change, regenerate the tables so that the LaTeX outputs and manifest remain synchronized with their inputs.
