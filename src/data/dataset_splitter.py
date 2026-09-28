# ==============================
# ./src/data/dataset_splitter.py
# ==============================

import numpy as np
import pandas as pd
from typing import Tuple


def split_corpus_by_usfm(
        df: pd.DataFrame,
        group_col: str = "usfm_code",
        train_ratio: float = 0.90,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 1989
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a parallel biblical corpus into Train, Validation, and Test sets
    by grouping complete USFM verse codes.

    All parallel pairs associated with the same USFM verse code are assigned
    to exactly one partition, preventing cross-edition data leakage across
    train, validation, and test sets.

    Args:
        df: Clean aligned biblical DataFrame.
        group_col: Column containing verse identifiers
            (e.g. "usfm_code").
        train_ratio: Proportion of unique verse codes for training.
        val_ratio: Proportion of unique verse codes for validation.
        test_ratio: Proportion of unique verse codes for testing.
        seed: Random seed for reproducible splitting.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (train_df, val_df, test_df).
    """

    # Verify split ratios
    if not np.isclose(
        train_ratio + val_ratio + test_ratio,
        1.0
    ):
        raise ValueError(
            "Split ratios must sum to 1.0."
        )

    # Verify grouping column
    if group_col not in df.columns:
        raise KeyError(
            f"Column '{group_col}' was not found in the DataFrame."
        )

    # Verify that grouping identifiers are complete
    if df[group_col].isna().any():
        raise ValueError(
            f"Column '{group_col}' contains missing values."
        )

    # Extract unique verse codes
    unique_usfm_codes = np.sort(
        df[group_col].unique()
    )

    # Shuffle verse codes reproducibly
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_usfm_codes)

    n_total = len(unique_usfm_codes)

    # Number of verse codes assigned to each split
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_keys = set(
        unique_usfm_codes[:n_train]
    )

    val_keys = set(
        unique_usfm_codes[
            n_train:n_train + n_val
        ]
    )

    test_keys = set(
        unique_usfm_codes[
            n_train + n_val:
        ]
    )

    # Assign complete USFM verse groups to each split
    train_df = (
        df[df[group_col].isin(train_keys)]
        .reset_index(drop=True)
    )

    val_df = (
        df[df[group_col].isin(val_keys)]
        .reset_index(drop=True)
    )

    test_df = (
        df[df[group_col].isin(test_keys)]
        .reset_index(drop=True)
    )

    # Verify that no USFM verse code appears in more than one split
    if train_keys & val_keys:
        raise RuntimeError(
            "USFM leakage detected between train and validation."
        )

    if train_keys & test_keys:
        raise RuntimeError(
            "USFM leakage detected between train and test."
        )

    if val_keys & test_keys:
        raise RuntimeError(
            "USFM leakage detected between validation and test."
        )

    # Verify that all unique USFM verse codes were assigned
    if (
        len(train_keys)
        + len(val_keys)
        + len(test_keys)
        != n_total
    ):
        raise RuntimeError(
            "Some USFM verse codes were lost during splitting."
        )

    # Verify that no parallel pairs were lost
    if (
        len(train_df)
        + len(val_df)
        + len(test_df)
        != len(df)
    ):
        raise RuntimeError(
            "Some biblical parallel pairs were lost during splitting."
        )

    print("\n" + "=" * 60)
    print("BIBLE DATASET SPLIT SUMMARY")
    print("=" * 60)

    print(f"Unique USFM Verses : {n_total:,}")

    print(
        f" -> Train Verses : "
        f"{len(train_keys):,} "
        f"({len(train_df):,} parallel pairs)"
    )

    print(
        f" -> Val Verses   : "
        f"{len(val_keys):,} "
        f"({len(val_df):,} parallel pairs)"
    )

    print(
        f" -> Test Verses  : "
        f"{len(test_keys):,} "
        f"({len(test_df):,} parallel pairs)"
    )

    print("=" * 60 + "\n")

    return train_df, val_df, test_df



def split_dialogue_corpus(
        df: pd.DataFrame,
        group_col: str = "dialogue_id",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 1989
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a parallel conversational corpus into Train, Validation, and Test
    sets by grouping complete dialogues.

    All turns belonging to the same dialogue are assigned to exactly one
    partition, preventing dialogue-level data leakage across splits.

    Args:
        df: Clean dialogue DataFrame.
        group_col: Column containing dialogue identifiers
            (e.g. "dialogue_id").
        train_ratio: Proportion of unique dialogues for training.
        val_ratio: Proportion of unique dialogues for validation.
        test_ratio: Proportion of unique dialogues for testing.
        seed: Random seed for reproducible splitting.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (train_df, val_df, test_df).
    """

    if not np.isclose(
        train_ratio + val_ratio + test_ratio,
        1.0
    ):
        raise ValueError(
            "Split ratios must sum to 1.0."
        )

    if group_col not in df.columns:
        raise KeyError(
            f"Column '{group_col}' was not found in the DataFrame."
        )

    if df[group_col].isna().any():
        raise ValueError(
            f"Column '{group_col}' contains missing values."
        )

    # Extract unique dialogue IDs
    unique_dialogues = np.array(
        sorted(df[group_col].unique())
    )

    # Shuffle dialogue IDs reproducibly
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_dialogues)

    n_total = len(unique_dialogues)

    # Number of dialogues assigned to each split
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_dialogues = set(
        unique_dialogues[:n_train]
    )

    val_dialogues = set(
        unique_dialogues[
            n_train:n_train + n_val
        ]
    )

    test_dialogues = set(
        unique_dialogues[
            n_train + n_val:
        ]
    )

    # Assign complete dialogues to each split
    train_df = (
        df[df[group_col].isin(train_dialogues)]
        .reset_index(drop=True)
    )

    val_df = (
        df[df[group_col].isin(val_dialogues)]
        .reset_index(drop=True)
    )

    test_df = (
        df[df[group_col].isin(test_dialogues)]
        .reset_index(drop=True)
    )

    # Verify that no dialogue appears in more than one split
    if train_dialogues & val_dialogues:
        raise RuntimeError(
            "Dialogue leakage detected between train and validation."
        )

    if train_dialogues & test_dialogues:
        raise RuntimeError(
            "Dialogue leakage detected between train and test."
        )

    if val_dialogues & test_dialogues:
        raise RuntimeError(
            "Dialogue leakage detected between validation and test."
        )

    # Verify that no rows were lost
    if (
        len(train_df)
        + len(val_df)
        + len(test_df)
        != len(df)
    ):
        raise RuntimeError(
            "Some dialogue pairs were lost during splitting."
        )

    print("\n" + "=" * 60)
    print("DIALOGUE-LEVEL DATASET SPLIT SUMMARY")
    print("=" * 60)

    print(f"Unique Dialogues : {n_total:,}")

    print(
        f" -> Train Dialogues : "
        f"{len(train_dialogues):,} "
        f"({len(train_df):,} parallel pairs)"
    )

    print(
        f" -> Val Dialogues   : "
        f"{len(val_dialogues):,} "
        f"({len(val_df):,} parallel pairs)"
    )

    print(
        f" -> Test Dialogues  : "
        f"{len(test_dialogues):,} "
        f"({len(test_df):,} parallel pairs)"
    )

    print("=" * 60 + "\n")

    return train_df, val_df, test_df