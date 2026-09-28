# =================
# ./3_align_verses.py
# =================

from pathlib import Path
from src.data.verse_aligner import align_bible_verses
from src.utils.config import load_config


if __name__ == "__main__":

    cfg = load_config("config.yaml")

    aym_codes = [bible["code"] for bible in cfg["bibles"] if bible["lang"] == "aym"]
    spa_codes = [bible["code"] for bible in cfg["bibles"] if bible["lang"] == "spa"]

    output_dir = Path(cfg["interim_data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tsv = output_dir / "verses_aligned_raw.tsv"

    df = align_bible_verses(aym_codes=aym_codes, spa_codes=spa_codes, base_dir=cfg["raw_bible_data_dir"])

    if not df.empty:

        df.to_csv(output_tsv, sep="\t", index=False, encoding="utf-8")
        print(f"[SAVED] Raw Aligned Verses successfully exported to: {output_tsv}")

    else:
        print("[WARN] No aligned verses found to export.")
