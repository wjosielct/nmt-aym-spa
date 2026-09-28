# ========================
# ./8_train_tokenizer.py
# ========================

from itertools import chain
from pathlib import Path

import pandas as pd

from src.tokenization.tokenizer import train_sentencepiece_tokenizer
from src.utils.config import load_config


if __name__ == "__main__":

    cfg = load_config("config.yaml")

    # =========================================================
    # PATHS
    # =========================================================

    splits_dir = Path(
        cfg["splits_dir"]
    )

    tokenizer_dir = Path(
        cfg["tokenizer_dir"]
    )

    train_tsv_path = (
        splits_dir
        / "train_bible.tsv"
    )

    # =========================================================
    # LOAD TOKENIZER TRAINING DATA
    # =========================================================
    #
    # IMPORTANT:
    # The SentencePiece vocabulary is learned exclusively from
    # the biblical training partition.
    #
    # The conversational corpus is intentionally excluded from
    # tokenizer training to preserve source-domain / target-domain
    # separation in the domain-adaptation experiments.
    # =========================================================

    train_df = pd.read_csv(
        train_tsv_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )

    print(
        f"[Loaded] Biblical training data from: "
        f"{train_tsv_path} "
        f"(pairs: {len(train_df):,})"
    )

    # =========================================================
    # SENTENCEPIECE TRAINING STREAM
    # =========================================================
    #
    # Both languages from train_bible.tsv contribute to the
    # shared subword vocabulary.
    # =========================================================

    training_sentences = chain(
        train_df["aym_text"].astype(str),
        train_df["spa_text"].astype(str),
    )

    # =========================================================
    # PREDEFINED USER-DEFINED SYMBOLS
    # =========================================================
    #
    # Complete ASCII decimal digit class plus "/".
    #
    # These symbols are reserved explicitly and are NOT learned
    # from validation or test data.
    #
    # split_digits=True ensures that sequences such as "2026"
    # can be represented digit by digit.
    # =========================================================

    user_defined_symbols = list(
        "0123456789/"
    )

    print(
        "[Training] SentencePiece tokenizer..."
    )

    print(
        f"[Output] Tokenizer directory: "
        f"{tokenizer_dir}"
    )

    # =========================================================
    # TRAIN TOKENIZER
    # =========================================================

    train_sentencepiece_tokenizer(
        sentences=training_sentences,
        vocab_size=int(
            cfg["vocab_size"]
        ),
        model_type=str(
            cfg["model_type"]
        ),
        character_coverage=float(
            cfg["character_coverage"]
        ),
        user_defined_symbols=user_defined_symbols,
        split_digits=True,
        tokenizer_dir=tokenizer_dir,
    )