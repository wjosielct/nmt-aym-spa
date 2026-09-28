# ===============================
# ./src/tokenization/tokenizer.py
# ===============================

from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import sentencepiece as spm


def train_sentencepiece_tokenizer(
        sentences: Iterable[str] | Iterator[str],
        vocab_size: int = 8000,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
        user_defined_symbols: list[str] | None = None,
        split_digits: bool = True,
        tokenizer_dir: str | Path = "tokenizer",
) -> None:
    """
    Trains a SentencePiece subword tokenizer directly from an iterable
    of training sentences.

    The learned subword vocabulary is obtained exclusively from the
    supplied training sentences. A small predefined set of
    user-defined symbols can additionally be reserved to guarantee
    that selected characters remain representable even when they are
    absent from the tokenizer training corpus.

    Args:
        sentences:
            Iterable containing the tokenizer training sentences.

        vocab_size:
            Target SentencePiece vocabulary size.

        model_type:
            SentencePiece model type, e.g. "bpe" or "unigram".

        character_coverage:
            Fraction of observed characters covered by the learned
            SentencePiece alphabet.

        user_defined_symbols:
            Symbols explicitly reserved in the vocabulary. These
            symbols do not need to occur in the tokenizer training
            corpus. By default, no additional symbols are reserved.

        split_digits:
            If True, numeric sequences are segmented into individual
            digits.

        tokenizer_dir:
            Directory where the SentencePiece model and vocabulary
            files will be stored.

    Special Token IDs:
        0: <pad>
        1: <s>    (BOS)
        2: </s>   (EOS)
        3: <unk>
    """

    if user_defined_symbols is None:
        user_defined_symbols = []

    tokenizer_dir = Path(tokenizer_dir)

    tokenizer_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sp_prefix = tokenizer_dir / "SentencePiece"

    # ---------------------------------------------------------
    # Train SentencePiece.
    # ---------------------------------------------------------
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences,
        model_prefix=str(sp_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        user_defined_symbols=user_defined_symbols,
        split_digits=split_digits,
        shuffle_input_sentence=True,
        normalization_rule_name="nmt_nfkc",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )

    model_path = tokenizer_dir / "SentencePiece.model"

    print(
        f"☑ Done! tokenizer saved to --> {tokenizer_dir}"
    )

    # ---------------------------------------------------------
    # Verify explicitly reserved symbols.
    # ---------------------------------------------------------
    if user_defined_symbols:

        sp_model = spm.SentencePieceProcessor(
            model_file=str(model_path)
        )

        unk_id = sp_model.unk_id()

        print("\n" + "=" * 60)
        print("TOKENIZER SYMBOL VERIFICATION")
        print("=" * 60)

        for symbol in user_defined_symbols:

            symbol_id = sp_model.piece_to_id(symbol)

            if symbol_id == unk_id:
                raise RuntimeError(
                    f"Reserved symbol '{symbol}' was mapped "
                    f"to <unk>."
                )

            print(
                f"'{symbol}' -> ID {symbol_id}"
            )

        # -----------------------------------------------------
        # Verify that encoding the complete reserved sequence
        # does not generate an unknown token.
        # -----------------------------------------------------
        test_string = "".join(user_defined_symbols)

        test_ids = sp_model.encode(
            test_string,
            out_type=int,
        )

        if unk_id in test_ids:
            raise RuntimeError(
                "The reserved-symbol verification string "
                "generated <unk>."
            )

        print("-" * 60)
        print(
            f"Test string     : {test_string}"
        )
        print(
            f"Encoded pieces  : "
            f"{sp_model.encode(test_string, out_type=str)}"
        )
        print(
            f"Encoded IDs     : {test_ids}"
        )
        print(
            "Contains <unk>  : False"
        )
        print("=" * 60 + "\n")


def encode_dataframe(
        df: pd.DataFrame,
        sp_model: spm.SentencePieceProcessor,
        source_col: str = "aym_text",
        target_col: str = "spa_text",
) -> pd.DataFrame:
    """
    Tokenizes parallel text columns in a DataFrame using a
    SentencePiece model.

    Args:
        df:
            Input DataFrame containing parallel text columns.

        sp_model:
            Loaded SentencePiece model.

        source_col:
            Name of the source text column.

        target_col:
            Name of the target text column.

    Returns:
        pd.DataFrame:
            Copy of the input DataFrame with encoded source and
            target ID columns.
    """

    bos_id = sp_model.bos_id()
    eos_id = sp_model.eos_id()

    def _encode_sequence(text: str) -> str:

        if not text.strip():
            ids = []

        else:
            ids = sp_model.encode(
                text.strip(),
                out_type=int,
            )

        ids = [bos_id] + ids + [eos_id]

        return " ".join(
            map(str, ids)
        )

    df_encoded = df.copy()

    src_target_name = (
        f"{source_col.split('_')[0]}_ids"
    )

    tgt_target_name = (
        f"{target_col.split('_')[0]}_ids"
    )

    print(
        f" --> Encoding column: '{source_col}'..."
    )

    df_encoded[src_target_name] = (
        df_encoded[source_col]
        .map(_encode_sequence)
    )

    print(
        f" --> Encoding column: '{target_col}'..."
    )

    df_encoded[tgt_target_name] = (
        df_encoded[target_col]
        .map(_encode_sequence)
    )

    return df_encoded