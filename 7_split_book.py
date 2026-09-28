# =================
# ./7_split_book.py
# =================

from pathlib import Path
import pandas as pd
from src.data.dataset_splitter import split_dialogue_corpus
from src.utils.config import load_config


if __name__ == "__main__":

    cfg = load_config("config.yaml")

    input_path = (
        Path(cfg["clean_data_dir"])
        / "book_clean.tsv"
    )

    splits_dir = Path(cfg["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Load clean conversational dataset
    df = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8"
    )

    print(f"▪ Total clean dialogue pairs: {len(df):,}")

    # Split at dialogue level
    train_df, val_df, test_df = split_dialogue_corpus(
        df=df,
        group_col="dialogue_id",
        train_ratio=cfg["book_train_ratio"],
        val_ratio=cfg["book_val_ratio"],
        test_ratio=cfg["book_test_ratio"],
        seed=cfg["data_split_seed"],
    )

    # Save splits
    train_df.to_csv(
        splits_dir / "train_dialogue.tsv",
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    val_df.to_csv(
        splits_dir / "val_dialogue.tsv",
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    test_df.to_csv(
        splits_dir / "test_dialogue.tsv",
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    print(
        f"[Saved] Dialogue-level splits exported successfully "
        f"to --> {splits_dir}"
    )