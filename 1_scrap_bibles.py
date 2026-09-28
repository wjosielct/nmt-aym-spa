# =================
# ./1_scrap_bibles.py
# =================

import json
from pathlib import Path
from src.data.web_scraper import scrap_bible
from src.utils.config import load_config


if __name__ == "__main__":

    cfg = load_config("config.yaml")

    base_output_dir = Path(cfg["raw_bible_data_dir"])

    for bible in cfg["bibles"]:
        code = bible["code"]
        lang = bible["lang"]

        # Extract bible
        bible_info = scrap_bible(bible_code=code, lang=lang)

        if not bible_info:
            print(f"[WARN] No verses were extracted for Bible with Code: {code}. Skipping file generation.")
            continue

        # Create target language directory
        output_dir = base_output_dir / lang
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save bible as a JSON file
        output_file = output_dir / f"{code}.json"

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(bible_info, file, ensure_ascii=False, indent=4)
