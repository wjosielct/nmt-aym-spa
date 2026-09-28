# Data reconstruction

The datasets used by this project are **not redistributed in this repository**.

This directory contains documentation only. The public code reconstructs the research data from the original external sources. Users are responsible for obtaining those sources lawfully and for complying with the applicable licenses, rights statements, website terms, and access conditions.

## Expected local directory layout

After source acquisition and execution of the preprocessing pipeline, the project expects the following structure:

```text
data/
├── raw/
│   ├── bible/
│   │   ├── aym/
│   │   │   ├── 293.json
│   │   │   └── 2250.json
│   │   └── spa/
│   │       ├── 146.json
│   │       └── 4278.json
│   └── book/
│       ├── book.pdf
│       └── book.tsv
├── interim/
│   └── verses_aligned_raw.tsv
├── clean/
│   ├── verses_aligned_clean.tsv
│   └── book_clean.tsv
├── splits/
│   ├── train_bible.tsv
│   ├── val_bible.tsv
│   ├── test_bible.tsv
│   ├── train_dialogue.tsv
│   ├── val_dialogue.tsv
│   └── test_dialogue.tsv
└── tokenized/
    ├── train_bible_tokenized.tsv
    ├── val_bible_tokenized.tsv
    ├── test_bible_tokenized.tsv
    ├── train_dialogue_tokenized.tsv
    ├── val_dialogue_tokenized.tsv
    └── test_dialogue_tokenized.tsv
```

These generated files are intentionally ignored by Git.

## Biblical parallel corpus

The biblical source domain is reconstructed from Bible.com using the edition identifiers configured in `config.yaml`.

| Language | Bible ID | Edition | Rights holder / notice |
|---|---:|---|---|
| Aymara | 293 | *Qullan Arunaca* | Sociedad Bíblica Boliviana |
| Aymara | 2250 | *Qullan Arunaka DC* | Sociedad Bíblica Boliviana |
| Spanish | 4278 | *Dios Habla Hoy*, 4th edition | Sociedades Bíblicas Unidas |
| Spanish | 146 | *Reina Valera Contemporánea* | Sociedades Bíblicas Unidas |

The scraper implemented in `src/data/web_scraper.py` requests chapters using the pattern:

```text
https://www.bible.com/bible/<BIBLE_ID>/<USFM_BOOK>.<CHAPTER>
```

For example, edition ID `293` is addressed through Bible.com with that identifier.

### Important rights note

The Bible texts are not licensed by this repository and are not redistributed here. The rights remain with the corresponding publishers/rights holders. Users must independently verify that their intended access, scraping, storage, and use comply with the source website's current terms and the relevant rights restrictions.

## Conversational parallel corpus

The target-domain conversational corpus is reconstructed from:

**AYMARA ARUSKIPAWINAKA: Conversaciones en aimara**
Editors: **Román Pairumani Ajacopa** and **Alejandra Bertha Carrasco Lima**
First electronic edition, La Paz, Bolivia, January 2022.

Public source page:

https://www.researchgate.net/publication/362907049_Aymara_Aruskipawinaka_-_Conversaciones_en_aimara

The source credits state **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

Although the source identifies a CC BY-NC 4.0 license, this repository deliberately does **not** redistribute the PDF or the derived conversational corpus. This keeps the data-distribution policy consistent across both research domains.

### Required local placement

Download the PDF from the source page and save it locally as:

```text
data/raw/book/book.pdf
```

The current extractor expects this exact path.

The parser in `src/data/pdf_scraper.py` extracts the dialogue section from PDF pages 13 through 146 as configured by `2_scrap_book.py`. It does not perform OCR; the PDF must contain extractable text.

## Data reconstruction pipeline

Run all commands from the repository root.

### 1. Acquire the Bible editions

```bash
python 1_scrap_bibles.py
```

This script:

- reads the Bible edition IDs from `config.yaml`;
- retrieves verses from Bible.com;
- preserves USFM verse identifiers;
- saves one JSON file per edition under `data/raw/bible/<language>/`.

### 2. Extract the conversational PDF

After placing `book.pdf` in `data/raw/book/`, run:

```bash
python 2_scrap_book.py
```

The extractor:

- reads PDF text in page order;
- groups turns by numbered `ARUSKIPAWI` headings;
- identifies speaker turns;
- preserves dialogue provenance;
- separates the Aymara and Spanish portions of bilingual turns;
- writes `data/raw/book/book.tsv`.

### 3. Align biblical editions

```bash
python 3_align_verses.py
```

This aligns the two Aymara and two Spanish editions by shared USFM verse identifier and writes:

```text
data/interim/verses_aligned_raw.tsv
```

### 4. Clean the biblical corpus

```bash
python 4_preprocess_bible.py
```

The cleaning stage:

- normalizes text according to the project preprocessing rules;
- removes Bible-specific reference markers;
- removes empty pairs;
- removes duplicate bilingual pairs;
- applies the configured length and source/target ratio filters.

Output:

```text
data/clean/verses_aligned_clean.tsv
```

### 5. Clean the conversational corpus

```bash
python 5_preprocess_book.py
```

The conversational cleaning stage applies the corresponding normalization, duplicate removal, empty-pair removal, and configured length/ratio filtering without Bible-specific reference-marker removal.

Output:

```text
data/clean/book_clean.tsv
```

### 6. Split the biblical corpus

```bash
python 6_split_bible.py
```

The biblical corpus is split by **complete USFM verse group** rather than individual aligned rows.

Current configuration:

- train: `0.90`;
- validation: `0.05`;
- test: `0.05`;
- split seed: `1989`.

All edition-pair variants associated with the same `usfm_code` remain in exactly one partition.

### 7. Split the conversational corpus

```bash
python 7_split_book.py
```

The conversational corpus is split by **complete dialogue**, not by individual turn.

Current configuration:

- train: `0.70`;
- validation: `0.15`;
- test: `0.15`;
- split seed: `1989`.

This prevents turns from the same dialogue from leaking across train, validation, and test partitions.

## Reference preparation counts

The following values were obtained from the validated data-preparation snapshot used while preparing this repository. They provide useful integrity checks when reconstructing the data from the external sources.

### Raw sources

| Artifact | Count |
|---|---:|
| Aymara Bible ID 293 | 30,816 verses |
| Aymara Bible ID 2250 | 30,700 verses |
| Spanish Bible ID 146 | 31,104 verses |
| Spanish Bible ID 4278 | 30,720 verses |
| Raw aligned biblical pairs | 122,154 |
| Raw conversational pairs extracted from the PDF | 1,912 |

### Clean corpora

| Corpus | Parallel pairs | Grouping units |
|---|---:|---:|
| Biblical | 120,904 | 30,788 unique USFM verses |
| Conversational | 1,623 | 40 dialogues |

### Final splits

| Corpus | Split | Groups | Parallel pairs |
|---|---|---:|---:|
| Biblical | train | 27,709 USFM verses | 108,823 |
| Biblical | validation | 1,539 USFM verses | 6,048 |
| Biblical | test | 1,540 USFM verses | 6,033 |
| Conversational | train | 28 dialogues | 1,128 |
| Conversational | validation | 6 dialogues | 254 |
| Conversational | test | 6 dialogues | 241 |

External websites and files can change over time. A mismatch in these counts should be investigated before training rather than silently accepted.

## SentencePiece tokenizer

Train the shared tokenizer with:

```bash
python 8_train_tokenizer.py
```

The tokenizer is trained **exclusively on `train_bible.tsv`**, using the Aymara and Spanish text from that training partition.

Current tokenizer policy:

- SentencePiece model: BPE;
- vocabulary size: `8000`;
- character coverage: `1.0`;
- user-defined symbols: `0123456789/`;
- digit splitting: enabled.

The conversational corpus is intentionally excluded from tokenizer training.

The repository tracks the frozen tokenizer used by the study:

```text
tokenizer/SentencePiece.model
tokenizer/SentencePiece.vocab
```

Reference SHA-256 for the tracked model:

```text
5508521388f095f47a905b24586e647b703a780805d80c0bc043256a3ca1cf21
```

## Encode the final splits

Biblical splits:

```bash
python 9_encode_bible.py
```

Conversational splits:

```bash
python 10_encode_book.py
```

Both scripts add SentencePiece token-ID representations while retaining the original row metadata.

## Reference split hashes

The following SHA-256 values correspond to the validated data-preparation snapshot. They are provided only as integrity checks; the underlying data files are not redistributed.

### Plain-text splits

```text
train_bible.tsv
3c17cc3ed729ad76cc8b503de5196459fc8fb375fbe5b300bca8990bfa5b04a4

val_bible.tsv
0a2ef2ace4f551d85b9de13f3b4ae0259eb1d91427658d14281d2cc70d8f0cf5

test_bible.tsv
4072b4d61cf0bd591af68c4d5afa96f70f201f46f0bf564fd6b76072f15c6720

train_dialogue.tsv
5e8b7a87d678212f41b4a010aa15621d682137bf2659eb2166687de6ad7d009a

val_dialogue.tsv
384f4b3580b2310a8eb195865fcc30ccec7e04d3f7ee75675475bb06796a23ac

test_dialogue.tsv
462cdcc22aba43d9721887381021d59d54bd80b385152fbc95b1eba879627174
```

### Tokenized splits

```text
train_bible_tokenized.tsv
3be6fef96d5d8121a4fd1d2459b3ceb3dfed55358456e08a8e4ebe53febcea94

val_bible_tokenized.tsv
cdc8f634185eb3ec7425c6bd858dc7952990b329491da3d27db1235f1058c716

test_bible_tokenized.tsv
92806cbebaea0d01d5a9f2527a1b020bde2fd4f4c0ad4c5a3e90a85aae9d9afa

train_dialogue_tokenized.tsv
547e9a9c06eb9c842b3880d552bbbad493b3913f45f118d08dbce37cb646a3d3

val_dialogue_tokenized.tsv
e00f41e192c1764a670607d5503f2482e732475ddb124675d13e779291cd55b9

test_dialogue_tokenized.tsv
f3aa86048bc1e03009b92d74b70ef029bed3e32ea96ed97fe132d5f0f49d995b
```

## Data licensing and repository license

The repository's MIT license applies to the project's original source code only.

It does **not** relicense:

- Bible.com content;
- Bible editions used as source material;
- *AYMARA ARUSKIPAWINAKA: Conversaciones en aimara*;
- reconstructed biblical or conversational datasets;
- any other third-party material.

Users remain responsible for complying with the licenses, rights statements, and terms applicable to the source materials.
