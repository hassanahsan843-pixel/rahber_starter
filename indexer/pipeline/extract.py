"""Stage 1: Extract. PDF, HTML, or Markdown to text, page by page."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from .models import Document, Page

# Optional imports for PDF and OCR extraction
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None


class _SimpleHTMLTableParser(HTMLParser):
    """Parse HTML tables into structured markdown pipe format."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"td", "th"}:
            self._in_cell = True
            self._current_cell = []
        elif tag == "tr":
            self._current_row = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"}:
            self._in_cell = False
            self._current_row.append(" ".join(self._current_cell).strip())
        elif tag == "tr":
            if any(self._current_row):
                self.rows.append(self._current_row)

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        lines = []
        header = self.rows[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in self.rows[1:]:
            padded = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(padded[:len(header)]) + " |")
        return "\n".join(lines)


def _extract_html(content: str, source_name: str) -> list[Page]:
    parser = _SimpleHTMLTableParser()
    parser.feed(content)
    md_table = parser.to_markdown()

    # Also strip basic HTML tags for non-table body text
    cleaned_text = re.sub(r"<style.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r"<script.*?</script>", "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r"<[^>]+>", " ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    combined = (md_table + "\n\n" + cleaned_text).strip() if md_table else cleaned_text
    return [Page(source=source_name, page=1, text=combined, ocr=False)]


def _extract_pdf(path: Path, source_name: str) -> list[Page]:
    pages: list[Page] = []
    
    # Try pdfplumber first for superior layout and table preservation
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(path)) as pdf:
                for idx, page in enumerate(pdf.pages, start=1):
                    raw_text = page.extract_text() or ""
                    # Table extraction
                    tables = page.extract_tables()
                    table_md = ""
                    if tables:
                        for tbl in tables:
                            rows = [[cell or "" for cell in r] for r in tbl if any(r)]
                            if rows and len(rows[0]) > 0:
                                table_md += "\n| " + " | ".join(rows[0]) + " |\n"
                                table_md += "| " + " | ".join(["---"] * len(rows[0])) + " |\n"
                                for r in rows[1:]:
                                    table_md += "| " + " | ".join(r) + " |\n"

                    final_text = (raw_text + "\n" + table_md).strip()
                    ocr_applied = False

                    # Property 3: OCR fallback for scanned pages with no text layer
                    if len(final_text.strip()) < 30 and pytesseract is not None:
                        try:
                            img = page.to_image(resolution=200).original
                            ocr_text = pytesseract.image_to_string(img)
                            if len(ocr_text.strip()) > len(final_text.strip()):
                                final_text = ocr_text.strip()
                                ocr_applied = True
                        except Exception:
                            pass

                    pages.append(Page(source=source_name, page=idx, text=final_text, ocr=ocr_applied))
                if pages:
                    return pages
        except Exception:
            pass

    # Fallback to pypdf
    if pypdf is not None:
        try:
            reader = pypdf.PdfReader(str(path))
            for idx, page in enumerate(reader.pages, start=1):
                txt = page.extract_text() or ""
                pages.append(Page(source=source_name, page=idx, text=txt.strip(), ocr=False))
            if pages:
                return pages
        except Exception:
            pass

    # If neither pdf library was available or extraction was empty
    return [Page(source=source_name, page=1, text="", ocr=False)]


def extract(path: str, doc_version: str, superseded: bool) -> Document:
    """Return the document's text page by page.

    Detect pages with no text layer and OCR them; mark those pages ocr=True.
    Preserves table structure and converts HTML tables to markdown pipe format.
    """
    p = Path(path)
    source_name = p.name
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        pages = _extract_pdf(p, source_name)
    elif suffix in {".html", ".htm"}:
        content = p.read_text(encoding="utf-8", errors="replace")
        pages = _extract_html(content, source_name)
    else:
        # Markdown / text
        content = p.read_text(encoding="utf-8", errors="replace")
        # Split into logical pages if form feed or page dividers present
        raw_pages = content.split("\x0c") if "\x0c" in content else [content]
        pages = [Page(source=source_name, page=i + 1, text=txt.strip(), ocr=False)
                 for i, txt in enumerate(raw_pages)]

    return Document(
        source=source_name,
        doc_version=doc_version,
        superseded=superseded,
        pages=pages,
    )
