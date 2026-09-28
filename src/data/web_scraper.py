# =========================
# ./src/data/web_scraper.py
# =========================

import re
from time import sleep
from bs4 import BeautifulSoup
from curl_cffi import requests


BOOKS = {
    "GEN": 50, "EXO": 40, "LEV": 27, "NUM": 36, "DEU": 34, "JOS": 24, "JDG": 21, "RUT": 4, "1SA": 31, "2SA": 24,
    "1KI": 22, "2KI": 25, "1CH": 29, "2CH": 36, "EZR": 10, "NEH": 13, "EST": 10, "JOB": 42, "PSA": 150, "PRO": 31,
    "ECC": 12, "SNG": 8, "ISA": 66, "JER": 52, "LAM": 5, "EZK": 48, "DAN": 12, "HOS": 14, "JOL": 3, "AMO": 9,
    "OBA": 1, "JON": 4, "MIC": 7, "NAM": 3, "HAB": 3, "ZEP": 3, "HAG": 2, "ZEC": 14, "MAL": 4, "MAT": 28,
    "MRK": 16, "LUK": 24, "JHN": 21, "ACT": 28, "ROM": 16, "1CO": 16, "2CO": 13, "GAL": 6, "EPH": 6, "PHP": 4,
    "COL": 4, "1TH": 5, "2TH": 3, "1TI": 6, "2TI": 4, "TIT": 3, "PHM": 1, "HEB": 13, "JAS": 5, "1PE": 5,
    "2PE": 3, "1JN": 5, "2JN": 1, "3JN": 1, "JUD": 1, "REV": 22
}


def scrap_bible_chapter(bible_code, book_code, chapter_number):

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    url = f"https://www.bible.com/bible/{bible_code}/{book_code}.{chapter_number}"

    try:

        response = requests.get(url, headers=headers, impersonate="chrome120")
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Search for span elements with the data-usfm attribute
        verse_elems = soup.find_all("span", attrs={"data-usfm": True})

        if not verse_elems:
            print(f"No verses found for --> {url}")
            return {}

        chapter_info = {}

        for verse_elem in verse_elems:
            usfm_code = verse_elem.get("data-usfm", "").strip()

            # Skip empty USFM codes
            if not usfm_code:
                continue

            for content_elem in verse_elem.find_all(class_=re.compile(r"_content", re.IGNORECASE)):
                text = content_elem.get_text(" ", strip=True)
                text = re.sub(r"\s+", " ", text).strip()

                if not text:
                    continue

                # Concatenate text if the verse is divided into multiple blocks
                if usfm_code in chapter_info:
                    chapter_info[usfm_code] += f" {text}"
                else:
                    chapter_info[usfm_code] = text

        # Normalize whitespace in final verses
        for usfm_code in chapter_info:
            chapter_info[usfm_code] = re.sub(r"\s+", " ", chapter_info[usfm_code]).strip()

        return chapter_info

    except Exception as e:
        print(f"[ERROR] {url} --> {e}")
        return {}

    finally:
        sleep(0.25)


def scrap_bible(bible_code, lang):

    full_bible = {}

    print(f"\n==================================================")
    print(f"Starting Bible extraction with Code {bible_code} ({lang})\n")

    for book_idx, (book_code, total_chapters) in enumerate(BOOKS.items(), start=1):
        print(f"[{book_idx}/{len(BOOKS)}] Processing Bible {book_code} ({total_chapters} chapters)...")

        for chapter_number in range(1, total_chapters + 1):
            chapter_info = scrap_bible_chapter(bible_code, book_code, chapter_number)
            full_bible.update(chapter_info)

    print(f"\n¡Extraction completed successfully!")
    print(f"==================================================\n")

    return full_bible
