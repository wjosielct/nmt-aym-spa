# ==========================
# ./src/data/preprocessor.py
# ==========================

import re
import unicodedata
import pandas as pd


# Matches reference markers such as "10 (11)", "20-21 (5-6)", etc.
PATTERN_1 = re.compile(r'\S*\d\S*\s+\(\S*\d\S*\)')

# Matches parenthesized sequences containing at least one digit,
# optionally preceded by digits, e.g. "(21.1)", "(5-6)", "1(2)".
PATTERN_2 = re.compile(r'\d*\(\S*\d\S*\)')

# Matches redundant whitespaces
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_unicode(
        text: str
) -> str:

    """
    Normalizes string encoding to Unicode NFC (Canonical Composition).
    """

    return unicodedata.normalize("NFC", text)


def normalize_quotes(
        text: str
) -> str:

    """
    Standardizes typographic quotes to standard ASCII single and double quotes.
    """

    text = text.replace("«", '"').replace("»", '"')
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("`", "'").replace("´", "'")

    return text


def normalize_punctuation(
        text: str
) -> str:

    """
    Normalizes em dashes and ellipses into standard ASCII characters.
    """

    text = text.replace("—", "-")
    text = text.replace("…", "...")

    return text


def clean_reference_markers(text: str) -> str:

    """
    Remove reference markers from a text string.
    """

    text = PATTERN_1.sub("", text)
    text = PATTERN_2.sub("", text)

    return text.strip()


def clean_spaces(
        text: str
) -> str:

    """
    Collapses consecutive whitespace characters into a single space and strips boundaries.
    """

    return WHITESPACE_PATTERN.sub(" ", text).strip()


def lowercase(
        text: str
) -> str:

    """
    Converts string characters to lowercase.
    """

    return text.lower()


def preprocess_text(
        text: str,
        remove_reference_markers: bool = False
) -> str:

    """
    Applies the sequential preprocessing pipeline to an individual text string.

    Args:
        text: Input string to be preprocessed.
        remove_reference_markers:
            If True, removes Bible-specific reference markers before
            whitespace normalization. Default is False.

    Returns:
        str: Preprocessed text string.
    """

    text = normalize_unicode(text)
    text = normalize_quotes(text)
    text = normalize_punctuation(text)

    if remove_reference_markers:
        text = clean_reference_markers(text)

    text = clean_spaces(text)
    text = lowercase(text)

    return text

def remove_empty_pairs(
        df: pd.DataFrame,
        source_col: str = "aym_text",
        target_col: str = "spa_text"
):

    initial_count = len(df)
    df_filtered = df[(df[source_col] != "") & (df[target_col] != "")].reset_index(drop=True)
    removed_count = initial_count - len(df_filtered)

    print(f"<<Empty Pairs>> --> Dropped {removed_count:,} pairs.")

    return df_filtered

def remove_duplicate_pairs(
        df: pd.DataFrame,
        source_col: str = "aym_text",
        target_col: str = "spa_text"
) -> pd.DataFrame:

    """
    Removes identical (source, target) sentence pairs from the DataFrame.

    Args:
        df: Input DataFrame containing parallel text columns.
        source_col: Column name for source sentences.
        target_col: Column name for target sentences.

    Returns:
        pd.DataFrame: Deduplicated DataFrame.
    """

    initial_count = len(df)
    df_dedup = df.drop_duplicates(subset=[source_col, target_col]).reset_index(drop=True)
    removed_count = initial_count - len(df_dedup)

    print(f"<<Duplicate Pairs>> --> Dropped {removed_count:,} pairs.")

    return df_dedup

def filter_by_length_and_ratio(
        df: pd.DataFrame,
        source_col: str = "aym_text",
        target_col: str = "spa_text",
        min_words: int = 2,
        max_words: int = 90,
        max_ratio: float = 3.5
) -> pd.DataFrame:
    """
    Filters parallel sentence pairs based on word counts and length ratios.

    Heuristics applied:
        1. Word bounds: min_words <= count <= max_words (for both src and tgt).
        2. Length ratio: max(len(src), len(tgt)) / min(len(src), len(tgt)) <= max_ratio.

    Note: Aymara is polysynthetic/agglutinative, so a relaxed max_ratio (e.g., 3.0-4.0)
    accommodates high morphological density without discarding valid pairs.

    Args:
        df: Input parallel DataFrame.
        source_col: Source text column name.
        target_col: Target text column name.
        min_words: Minimum allowed words per sentence.
        max_words: Maximum allowed words per sentence.
        max_ratio: Maximum allowed ratio between longer and shorter sentences.

    Returns:
        pd.DataFrame: Filtered DataFrame with valid parallel sentence pairs.
    """
    initial_count = len(df)

    src_lens = [len(text.split()) for text in df[source_col].astype(str)]
    tgt_lens = [len(text.split()) for text in df[target_col].astype(str)]

    valid_mask = []
    for s_len, t_len in zip(src_lens, tgt_lens):
        # Check minimum and maximum word boundaries
        if s_len < min_words or t_len < min_words:
            valid_mask.append(False)
            continue
        if s_len > max_words or t_len > max_words:
            valid_mask.append(False)
            continue

        # Check source/target length ratio
        min_len = min(s_len, t_len)
        max_len = max(s_len, t_len)
        if min_len == 0 or (max_len / min_len) > max_ratio:
            valid_mask.append(False)
            continue

        valid_mask.append(True)

    df_filtered = df[valid_mask].reset_index(drop=True)
    dropped_count = initial_count - len(df_filtered)

    print(f"<<Length & Ratio Filter>> --> Dropped {dropped_count:,} pairs.")

    return df_filtered
