# ===================
# ./2_scrap_book.py
# ===================

from pathlib import Path
from src.data.pdf_scraper import scrap_pdf
from src.utils.config import load_config


if __name__ == "__main__":

    cfg = load_config("config.yaml")

    pdf_dir = Path(cfg["raw_pdf_dir"])
    pdf_path = pdf_dir / "book.pdf"
    output_tsv = pdf_dir / "book.tsv"

    df = scrap_pdf(
        pdf_path=pdf_path,
        start_page=13,
        end_page=146,
    )

    if not df.empty:
        df.to_csv(
            output_tsv,
            sep="\t",
            index=False,
            encoding="utf-8",
        )

        print(
            f"[SAVED] Raw Book Pairs exported to --> {output_tsv}"
        )

    else:
        print("[WARN] No pairs were extracted from PDF.")