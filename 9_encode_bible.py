# =================
# ./9_encode_bible.py
# =================

from pathlib import Path

import pandas as pd
import sentencepiece as spm

from src.tokenization.tokenizer import encode_dataframe
from src.utils.config import load_config


if __name__ == "__main__":

    cfg = load_config("config.yaml")

    splits_dir = Path(cfg["splits_dir"])
    tokenizer_dir = Path(cfg["tokenizer_dir"])
    tokenized_dir = Path(cfg["tokenized_dir"])

    tokenized_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load trained SentencePiece tokenizer
    sp_model_path = tokenizer_dir / "SentencePiece.model"

    sp = spm.SentencePieceProcessor(
        model_file=str(sp_model_path)
    )

    # Tokenize each biblical split
    for split_name in [
        "train_bible",
        "val_bible",
        "test_bible",
    ]:

        input_tsv = splits_dir / f"{split_name}.tsv"
        output_tsv = tokenized_dir / f"{split_name}_tokenized.tsv"

        df_split = pd.read_csv(
            input_tsv,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )

        print(
            f"▪ Encoding '{split_name}.tsv' dataset "
            f"({len(df_split):,} pairs)..."
        )

        df_encoded = encode_dataframe(
            df=df_split,
            sp_model=sp,
            source_col="aym_text",
            target_col="spa_text",
        )

        df_encoded.to_csv(
            output_tsv,
            sep="\t",
            index=False,
            encoding="utf-8",
        )

    print(
        f"[Saved] Tokenized biblical splits to --> "
        f"{tokenized_dir}"
    )