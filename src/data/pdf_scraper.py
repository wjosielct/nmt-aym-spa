"""Extract bilingual speaker turns from 'AYMARA ARUSKIPAWINAKA: Conversaciones en aimara (2022)'.

The extractor:

- Reads physical PDF lines in page order.
- Groups turns by numbered ARUSKIPAWI headings.
- Detects speaker markers such as A), A:, M), etc.
- Keeps turns open across page breaks.
- Splits bilingual text using the FIRST slash "/".
- Excludes unlabelled bold narrative notes.
- Preserves additional slashes inside the Spanish translation.
- Preserves original spelling, punctuation, case, numbers and duplicates.

No cleaning, deduplication, filtering or dataset splitting is performed here.
Those operations belong to subsequent processing stages.
"""

from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import pandas as pd
import pdfplumber


# ---------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------

# Speaker markers:
# A)
# B)
# A:
# M)
# etc.
_SPEAKER = re.compile(r"^([A-ZÁÉÍÓÚÜÑ])\s*([):])\s*")

# Dialogue headings such as:
# 1. ARUSKIPAWI
# 2° ARUSKIPAWI
# 3º - ARUSKIPAWI
_HEADING = re.compile(
    r"^(\d+)\s*[°º.]?\s*[-–—]?\s*.*\bARUSKIPAWI\b",
    re.I,
)

# Credits appearing before/inside dialogue sections.
_CREDITS = re.compile(r"^Qilqirinaka\s*:", re.I)

# Separator/divider lines.
_DIVIDER = re.compile(r"^[-–—\s&]+$")

# End of the dialogue section of the book.
_END_MATTER = re.compile(
    r"^ULLARAÑATAKI\s*/|^BIBLIOGRAF[IÍ]A\b",
    re.I,
)

# Running headers to remove.
_HEADERS = {
    "conversaciones en aimara",
    "aymara aruskipawinaka",
}


# ---------------------------------------------------------------------
# Output columns
# ---------------------------------------------------------------------

_COLUMNS = [
    "source",
    "pair_id",
    "dialogue_id",
    "dialogue_title",
    "turn_id",
    "speaker",
    "speaker_marker",
    "page_start",
    "page_end",
    "line_start",
    "line_end",
    "aym_text",
    "spa_text",
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _space(text: str) -> str:
    """Normalize consecutive whitespace without altering the text."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class _Line:
    """One physical text line extracted from the PDF."""

    text: str
    page: int
    number: int
    bold: bool = False


@dataclass
class _Turn:
    """One speaker turn."""

    dialogue_id: int
    dialogue_title: str
    turn_id: int
    speaker: str
    marker: str
    lines: list[_Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Return all physical lines belonging to the turn."""
        return _space(
            " ".join(line.text for line in self.lines)
        )

    def location(self) -> dict[str, Any]:
        """Return provenance information for the turn."""
        first = self.lines[0]
        last = self.lines[-1]

        return {
            "pair_id": (
                f"book_d{self.dialogue_id:03d}"
                f"_t{self.turn_id:04d}"
            ),
            "dialogue_id": self.dialogue_id,
            "dialogue_title": self.dialogue_title,
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "speaker_marker": self.marker,
            "page_start": first.page,
            "page_end": last.page,
            "line_start": first.number,
            "line_end": last.number,
        }


# ---------------------------------------------------------------------
# PDF line extraction
# ---------------------------------------------------------------------

def _page_lines(
    page: Any,
    number: int,
) -> list[_Line]:
    """Extract physical text lines from one PDF page."""

    extracted = page.extract_text_lines(
        layout=False,
        strip=True,
        return_chars=True,
    )

    if not extracted:
        raise ValueError(
            f"PDF page {number} has no extractable text. "
            "Check the PDF; this extractor does not perform OCR."
        )

    result: list[_Line] = []

    for index, item in enumerate(extracted, start=1):

        text = _space(item["text"])

        if not text:
            continue

        # -------------------------------------------------------------
        # Remove running headers appearing at the top of the page.
        # -------------------------------------------------------------
        if (
            text.casefold() in _HEADERS
            and item["top"] < page.height * 0.12
        ):
            continue

        # -------------------------------------------------------------
        # Remove page number appearing at the bottom of the page.
        # -------------------------------------------------------------
        if (
            text == str(number)
            and item["top"] > page.height * 0.88
        ):
            continue

        # -------------------------------------------------------------
        # Determine whether the line is written entirely in bold.
        # -------------------------------------------------------------
        letters = [
            char
            for char in item.get("chars", [])
            if char["text"].isalpha()
        ]

        bold = bool(letters) and all(
            "bold" in char.get("fontname", "").lower()
            for char in letters
        )

        result.append(
            _Line(
                text=text,
                page=number,
                number=index,
                bold=bold,
            )
        )

    return result


# ---------------------------------------------------------------------
# Dialogue parsing
# ---------------------------------------------------------------------

def _parse_lines(
    lines: Iterable[_Line],
    following_line: _Line | None = None,
) -> pd.DataFrame:
    """Parse PDF lines and return bilingual dialogue pairs."""

    records: list[dict[str, Any]] = []

    seen_dialogues: set[int] = set()

    dialogue_id: int | None = None
    dialogue_title = ""

    turn_id = 0

    current: _Turn | None = None

    # Used only to identify narrative/stage-direction blocks.
    note: list[_Line] = []

    in_preamble = True

    skipped_turns = 0

    # -----------------------------------------------------------------
    # Close the current speaker turn.
    # -----------------------------------------------------------------
    def finish_turn() -> None:

        nonlocal current
        nonlocal skipped_turns

        if current is None:
            return

        location = current.location()

        # Remove exactly one speaker marker from the beginning.
        #
        # Example:
        # A) Kamisaraki? / ¿Cómo estás?
        #
        # becomes:
        # Kamisaraki? / ¿Cómo estás?
        body = _SPEAKER.sub(
            "",
            current.text,
            count=1,
        )

        # Split ONLY at the first slash.
        #
        # Additional slashes in the Spanish translation are preserved.
        left, separator, right = body.partition("/")

        left = _space(left)
        right = _space(right)

        # A valid bilingual pair must contain:
        #
        # Aymara / Spanish
        if not separator or not left or not right:
            skipped_turns += 1
            current = None
            return

        records.append(
            {
                "source": "book",
                **location,
                "aym_text": left,
                "spa_text": right,
            }
        )

        current = None

    # -----------------------------------------------------------------
    # Finish an excluded narrative note.
    # -----------------------------------------------------------------
    def finish_note() -> None:

        nonlocal note

        note = []

    # -----------------------------------------------------------------
    # Process physical lines sequentially.
    # -----------------------------------------------------------------
    for line in lines:

        text = line.text

        # -------------------------------------------------------------
        # New dialogue heading.
        # -------------------------------------------------------------
        heading = _HEADING.match(text)

        if heading:

            finish_turn()
            finish_note()

            new_id = int(heading.group(1))

            # Dialogue IDs must be unique and ordered.
            if (
                new_id in seen_dialogues
                or (
                    dialogue_id is not None
                    and new_id <= dialogue_id
                )
            ):
                raise ValueError(
                    "Duplicate/out-of-order dialogue heading "
                    f"on page {line.page}: {text}"
                )

            seen_dialogues.add(new_id)

            dialogue_id = new_id
            dialogue_title = text
            turn_id = 0

            in_preamble = True

            continue

        # -------------------------------------------------------------
        # Stop extraction once end matter begins.
        # -------------------------------------------------------------
        if _END_MATTER.match(text):

            finish_turn()
            finish_note()

            break

        # -------------------------------------------------------------
        # Credits or divider lines.
        # -------------------------------------------------------------
        if (
            _CREDITS.match(text)
            or _DIVIDER.fullmatch(text)
        ):

            finish_turn()
            finish_note()

            in_preamble = True

            continue

        # -------------------------------------------------------------
        # New speaker turn.
        # -------------------------------------------------------------
        speaker = _SPEAKER.match(text)

        if speaker:

            if dialogue_id is None:
                raise ValueError(
                    f"Speaker on page {line.page} has no "
                    "dialogue heading. "
                    "Start at the first page of a complete "
                    "dialogue (default: 13)."
                )

            finish_turn()
            finish_note()

            turn_id += 1

            current = _Turn(
                dialogue_id=dialogue_id,
                dialogue_title=dialogue_title,
                turn_id=turn_id,
                speaker=speaker.group(1),
                marker=speaker.group(2),
                lines=[line],
            )

            in_preamble = False

            continue

        # -------------------------------------------------------------
        # Ignore text before the first speaker of a dialogue.
        # This normally contains title continuations, authors, etc.
        # -------------------------------------------------------------
        if in_preamble:
            continue

        # -------------------------------------------------------------
        # Narrative/stage direction.
        #
        # In the inspected PDF there is a bold bilingual narrative
        # line containing "/". It must not be interpreted as part
        # of the preceding speaker turn.
        # -------------------------------------------------------------
        if line.bold and "/" in text:

            finish_turn()
            finish_note()

            note = [line]

            continue

        # -------------------------------------------------------------
        # Continue/finish narrative note.
        # -------------------------------------------------------------
        if note:

            if line.bold:
                note.append(line)
            else:
                finish_note()

            continue

        # -------------------------------------------------------------
        # Normal continuation of the current speaker turn.
        # -------------------------------------------------------------
        if current is not None:

            current.lines.append(line)

        # Lines appearing outside a speaker turn are simply ignored.


    # -----------------------------------------------------------------
    # Detect accidental truncation caused by end_page.
    # -----------------------------------------------------------------
    if (
        current is not None
        and following_line is not None
    ):

        text = following_line.text

        boundary = (
            _SPEAKER.match(text)
            or _HEADING.match(text)
            or _CREDITS.match(text)
            or _DIVIDER.fullmatch(text)
            or _END_MATTER.match(text)
            or (
                following_line.bold
                and "/" in text
            )
        )

        if not boundary:
            raise ValueError(
                "The selected page range ends inside a "
                f"speaker turn on page "
                f"{current.lines[-1].page}. "
                f"Its continuation starts on page "
                f"{following_line.page}; increase end_page."
            )

    # Close the final turn.
    finish_turn()
    finish_note()

    # -----------------------------------------------------------------
    # Basic validation
    # -----------------------------------------------------------------
    if not records:
        raise ValueError(
            "No complete bilingual speaker turns were found "
            "in the selected pages."
        )

    df = pd.DataFrame.from_records(
        records,
        columns=_COLUMNS,
    )

    # pair_id must be unique.
    if df["pair_id"].duplicated().any():
        raise ValueError(
            "Duplicate pair_id values were generated."
        )

    # Every detected dialogue must contain at least one extracted pair.
    counts = Counter(
        int(record["dialogue_id"])
        for record in records
    )

    empty_dialogues = seen_dialogues - counts.keys()

    if empty_dialogues:
        raise ValueError(
            "Dialogues without an extracted bilingual pair: "
            f"{sorted(empty_dialogues)}"
        )

    # This is only an informational console message.
    # No report file is created.
    if skipped_turns:
        print(
            f"Skipped incomplete/non-bilingual turns: "
            f"{skipped_turns}"
        )

    return df


# ---------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------

def scrap_pdf(
    pdf_path: str | Path,
    start_page: int = 13,
    end_page: int = 146,
) -> pd.DataFrame:
    """Extract raw Aymara-Spanish dialogue pairs from the PDF.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    start_page:
        First physical PDF page to process.
        Page numbering is 1-based and inclusive.

    end_page:
        Last physical PDF page to process.
        Page numbering is 1-based and inclusive.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:

        - source
        - pair_id
        - dialogue_id
        - dialogue_title
        - turn_id
        - speaker
        - speaker_marker
        - page_start
        - page_end
        - line_start
        - line_end
        - aym_text
        - spa_text

    Notes
    -----
    This function performs extraction only.
    """

    path = Path(pdf_path)

    # -----------------------------------------------------------------
    # Validate PDF path.
    # -----------------------------------------------------------------
    if not path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {path}"
        )

    # -----------------------------------------------------------------
    # Validate page arguments.
    # -----------------------------------------------------------------
    if (
        isinstance(start_page, bool)
        or isinstance(end_page, bool)
        or not isinstance(start_page, int)
        or not isinstance(end_page, int)
    ):
        raise TypeError(
            "start_page and end_page must be integers "
            "(1-based, inclusive)."
        )

    if start_page < 1:
        raise ValueError(
            "start_page must be >= 1."
        )

    if end_page < start_page:
        raise ValueError(
            "end_page must be >= start_page."
        )

    print(
        f"Extracting bilingual turns: "
        f"{path.name}, "
        f"PDF pages {start_page}-{end_page}"
    )

    # -----------------------------------------------------------------
    # Read PDF.
    # -----------------------------------------------------------------
    with pdfplumber.open(path) as pdf:

        number_of_pages = len(pdf.pages)

        if end_page > number_of_pages:
            raise ValueError(
                f"end_page={end_page} exceeds the PDF's "
                f"{number_of_pages} pages."
            )

        # -------------------------------------------------------------
        # Extract all physical lines from selected pages.
        # -------------------------------------------------------------
        lines = [
            line
            for page_number in range(
                start_page,
                end_page + 1,
            )
            for line in _page_lines(
                pdf.pages[page_number - 1],
                page_number,
            )
        ]

        # -------------------------------------------------------------
        # Read the first line of the following page, when available.
        #
        # This is used only to verify that end_page does not cut a
        # speaker turn in half.
        # -------------------------------------------------------------
        following_line = None

        if end_page < number_of_pages:

            next_lines = _page_lines(
                pdf.pages[end_page],
                end_page + 1,
            )

            if next_lines:
                following_line = next_lines[0]

        # -------------------------------------------------------------
        # Parse dialogue turns.
        # -------------------------------------------------------------
        df = _parse_lines(
            lines,
            following_line,
        )

    print(
        f"Extracted bilingual pairs: {len(df):,}"
    )

    return df