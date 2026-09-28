# ================
# ./6_split_bible.py
# ================

from pathlib import Path
import pandas as pd
from src.data.dataset_splitter import split_corpus_by_usfm
from src.utils.config import load_config


if __name__ == "__main__":

    cfg = load_config("config.yaml")

    input_path = Path(cfg["clean_data_dir"]) / "verses_aligned_clean.tsv"
    splits_dir = Path(cfg["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Load clean dataset
    df = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8"
    )

    # Perform split
    train_df, val_df, test_df = split_corpus_by_usfm(
        df=df,
        group_col="usfm_code",
        train_ratio=cfg["bible_train_ratio"],
        val_ratio=cfg["bible_val_ratio"],
        test_ratio=cfg["bible_test_ratio"],
        seed=cfg["data_split_seed"]
    )

    # Save split TSV files
    train_df.to_csv(splits_dir / "train_bible.tsv", sep="\t", index=False, encoding="utf-8")
    val_df.to_csv(splits_dir / "val_bible.tsv", sep="\t", index=False, encoding="utf-8")
    test_df.to_csv(splits_dir / "test_bible.tsv", sep="\t", index=False, encoding="utf-8")

    print(f"[Saved] Splits exported successfully to --> {splits_dir}")
