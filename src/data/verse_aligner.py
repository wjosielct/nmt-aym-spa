# ===========================
# ./src/data/verse_aligner.py
# ===========================

import json
from itertools import product
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_bible_from_json(
        bible_code: int | str,
        lang: str,
        base_dir: str | Path
) -> Dict[str, str]:

    """
    Loads a raw Bible JSON file: {USFM_CODE: text}.

    Args:
        bible_code: Bible identifier on Bible.com (e.g., 293, 4278).
        lang: Subfolder name corresponding to language (e.g., 'aym', 'spa').
        base_dir: Parent folder.

    Returns:
        Dict[str, str]: Mapping of USFM verse codes to their raw extracted text.
    """

    json_file = Path(base_dir) / lang / f"{bible_code}.json"

    if not json_file.exists():
        print(f"[WARN] File not found --> {json_file}")
        return {}

    try:
        with open(json_file, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as e:
        print(f"[ERROR] Failed to read '{json_file}' --> {e}")
        return {}


def align_bible_verses(
        aym_codes: List[int | str],
        spa_codes: List[int | str],
        base_dir: str | Path
):

    """
    Cross-aligns multiple raw Aymara and Spanish Bible editions using USFM verse codes.

    Computes the Cartesian product across editions and matches common USFM keys.

    Args:
        aym_codes: Bible codes for the Aymara editions.
        spa_codes: Bible codes for the Spanish editions.
        base_dir: Directory containing the raw Bible JSON files.

    Returns:
        pd.DataFrame: Aligned verse records containing text pairs and metadata.
    """

    aym_bibles = {code: load_bible_from_json(code, "aym", base_dir) for code in aym_codes}
    spa_bibles = {code: load_bible_from_json(code, "spa", base_dir) for code in spa_codes}

    # Cartesian product alignment
    records = []

    for aym_code, spa_code in product(aym_codes, spa_codes):
        aym_bible = aym_bibles[aym_code]
        spa_bible = spa_bibles[spa_code]

        # Intersect keys present in both translations
        common_usfm_codes = set(aym_bible.keys()) & set(spa_bible.keys())
        matched_count = 0

        for usfm_code in sorted(common_usfm_codes):
            aym_text = aym_bible[usfm_code].strip()
            spa_text = spa_bible[usfm_code].strip()

            if aym_text and spa_text:
                records.append({
                    "usfm_code": usfm_code,
                    "aym_code": aym_code,
                    "spa_code": spa_code,
                    "aym_text": aym_text,
                    "spa_text": spa_text
                })
                matched_count += 1

        print(f"• Aligned (Aym Code: {aym_code} ↔ Spa Code: {spa_code}) --> {matched_count:,} pairs")

    print(f"→ Alignment completed: {len(records):,} total pairs generated.")

    return pd.DataFrame(records)
