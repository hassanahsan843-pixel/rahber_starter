"""Stage 2: Clean. Remove artefacts without destroying structure."""

from __future__ import annotations

import re
from .models import Document, Page


def clean(doc: Document) -> Document:
    """Strip headers and footers that repeat on every page; keep headings and
    section boundaries, because chunking and citations depend on them.
    
    Deliberately preserves:
    - Numbered section clauses (e.g. "1.21 Attendance Policy", "4.2 Prerequisites")
    - Markdown pipe tables (| Col | Col |)
    - Heading hashes (###) and legal outline structures
    """
    cleaned_pages: list[Page] = []

    # Common repetitive header/footer regexes
    header_footer_patterns = [
        re.compile(r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*FAST[- ]NUCES(?:\s+Academic\s+Rules)?\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*Annual\s+Report\s+\d{4}[-–]\d{2,4}\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*National\s+University\s+of\s+Computer\s+and\s+Emerging\s+Sciences\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*[-–—]\s*\d+\s*[-–—]\s*$", re.MULTILINE),
    ]

    for p in doc.pages:
        text = p.text
        # Apply header/footer patterns
        for pat in header_footer_patterns:
            text = pat.sub("", text)

        # Normalize carriage returns and odd linebreaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse 3+ newlines into double newlines (preserving paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse horizontal whitespace while preserving leading indentation if any
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        cleaned_text = "\n".join(lines).strip()

        cleaned_pages.append(Page(
            source=p.source,
            page=p.page,
            text=cleaned_text,
            ocr=p.ocr,
        ))

    return Document(
        source=doc.source,
        doc_version=doc.doc_version,
        superseded=doc.superseded,
        pages=cleaned_pages,
    )
