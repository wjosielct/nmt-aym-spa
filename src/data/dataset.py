# =====================
# ./src/data/dataset.py
# =====================

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ParallelTranslationDataset(Dataset):
    """
    PyTorch Dataset for parallel NMT tokenized TSV files.

    The dataset expects pre-tokenized source and target columns
    containing whitespace-separated integer token IDs.
    """

    def __init__(
            self,
            tsv_path: str | Path,
            src_col: str = "aym_ids",
            tgt_col: str = "spa_ids",
    ):

        tsv_path = Path(tsv_path)

        if not tsv_path.exists():
            raise FileNotFoundError(
                f"Tokenized dataset not found: {tsv_path}"
            )

        self.df = pd.read_csv(
            tsv_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )

        # -----------------------------------------------------
        # Validate expected token-ID columns.
        # -----------------------------------------------------
        missing_columns = [
            col
            for col in (src_col, tgt_col)
            if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required column(s) in {tsv_path}: "
                f"{missing_columns}"
            )

        # -----------------------------------------------------
        # Parse whitespace-separated token-ID sequences.
        # -----------------------------------------------------
        self.src_ids = [
            self._parse_token_ids(
                text=text,
                row_idx=row_idx,
                column=src_col,
                tsv_path=tsv_path,
            )
            for row_idx, text in enumerate(self.df[src_col])
        ]

        self.tgt_ids = [
            self._parse_token_ids(
                text=text,
                row_idx=row_idx,
                column=tgt_col,
                tsv_path=tsv_path,
            )
            for row_idx, text in enumerate(self.df[tgt_col])
        ]

        if len(self.src_ids) != len(self.tgt_ids):
            raise ValueError(
                "Source and target dataset lengths do not match: "
                f"{len(self.src_ids)} != {len(self.tgt_ids)}"
            )

    @staticmethod
    def _parse_token_ids(
            text: str,
            row_idx: int,
            column: str,
            tsv_path: Path,
    ) -> List[int]:
        """
        Parses a whitespace-separated sequence of integer token IDs.
        """

        text = str(text).strip()

        if not text:
            raise ValueError(
                f"Empty token sequence found in "
                f"{tsv_path}, row={row_idx}, column='{column}'."
            )

        try:
            token_ids = [
                int(token)
                for token in text.split()
            ]

        except ValueError as exc:
            raise ValueError(
                f"Invalid token ID found in "
                f"{tsv_path}, row={row_idx}, column='{column}'."
            ) from exc

        if any(token_id < 0 for token_id in token_ids):
            raise ValueError(
                f"Negative token ID found in "
                f"{tsv_path}, row={row_idx}, column='{column}'."
            )

        return token_ids

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(
            self,
            idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        src = torch.tensor(
            self.src_ids[idx],
            dtype=torch.long,
        )

        tgt = torch.tensor(
            self.tgt_ids[idx],
            dtype=torch.long,
        )

        return src, tgt


def collate_fn_pad(
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
        pad_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamically pads source and target sequences to the maximum
    sequence length within the current batch.
    """

    if not batch:
        raise ValueError(
            "collate_fn_pad received an empty batch."
        )

    src_batch, tgt_batch = zip(*batch)

    src_padded = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=pad_id,
    )

    tgt_padded = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=pad_id,
    )

    return src_padded, tgt_padded