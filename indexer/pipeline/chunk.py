"""Stage 3: Chunk. Structure-aware, with overlap."""

from __future__ import annotations

import hashlib
import re
from .models import Chunk, Document


# Regex detecting markdown headings or numbered legal clauses (e.g., "1.21 Attendance", "Section 4", "### 5.4")
HEADING_REGEX = re.compile(
    r"^(?:#{1,6}\s+|(?:\d+\.){1,3}\d*\s+|[A-Z][A-Za-z0-9 ]+:\s*|Section\s+\d+[:\.]?\s*)(.+)$",
    re.MULTILINE,
)


def _split_into_sentences(text: str) -> list[str]:
    # Split on sentence terminals while keeping punctuation
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


def _tokenize_words(text: str) -> list[str]:
    return text.split()


def chunk(doc: Document, size: int = 400, overlap: int = 50) -> list[Chunk]:
    """Split on sections, then paragraphs, then sentences, keeping each chunk under
    `size` words/tokens, with `overlap` words/tokens shared between neighbours.
    
    Every chunk retains:
    - section heading it sits under
    - page number
    - doc_version
    - superseded flag
    - deterministic stable ID
    """
    chunks: list[Chunk] = []
    chunk_counter = 0

    current_section = "General"

    for p in doc.pages:
        page_text = p.text
        if not page_text.strip():
            continue

        # Split on structure: first detect major sections/headings
        paragraphs = _split_paragraphs(page_text)
        current_words: list[str] = []

        for para in paragraphs:
            # Check if this paragraph starts with or is a heading/section
            first_line = para.split("\n", 1)[0].strip()
            heading_match = HEADING_REGEX.match(first_line)
            if heading_match:
                # Update current active section
                current_section = first_line.lstrip("#").strip()

            para_words = _tokenize_words(para)

            # If paragraph itself is small enough, try adding to current buffer
            if len(current_words) + len(para_words) <= size:
                current_words.extend(para_words)
            else:
                # If paragraph fits within size by itself, flush current chunk and start new one
                if current_words:
                    chunk_text = " ".join(current_words)
                    chunk_counter += 1
                    stable_id = hashlib.sha256(
                        f"{doc.source}:{p.page}:{current_section}:{chunk_counter}".encode()
                    ).hexdigest()[:16]

                    chunks.append(Chunk(
                        id=f"{doc.source[:12]}-{stable_id}",
                        text=chunk_text,
                        source=doc.source,
                        section=current_section,
                        page=p.page,
                        doc_version=doc.doc_version,
                        superseded=doc.superseded,
                    ))

                    # Retain overlap words from the end of current_words
                    overlap_words = current_words[-overlap:] if overlap > 0 else []
                    current_words = list(overlap_words)

                # Now process the large paragraph: split by sentences if it exceeds size
                if len(para_words) > size:
                    sentences = _split_into_sentences(para)
                    for sent in sentences:
                        sent_words = _tokenize_words(sent)
                        if len(current_words) + len(sent_words) <= size:
                            current_words.extend(sent_words)
                        else:
                            if current_words:
                                chunk_text = " ".join(current_words)
                                chunk_counter += 1
                                stable_id = hashlib.sha256(
                                    f"{doc.source}:{p.page}:{current_section}:{chunk_counter}".encode()
                                ).hexdigest()[:16]

                                chunks.append(Chunk(
                                    id=f"{doc.source[:12]}-{stable_id}",
                                    text=chunk_text,
                                    source=doc.source,
                                    section=current_section,
                                    page=p.page,
                                    doc_version=doc.doc_version,
                                    superseded=doc.superseded,
                                ))
                                overlap_words = current_words[-overlap:] if overlap > 0 else []
                                current_words = list(overlap_words)
                            current_words.extend(sent_words)
                else:
                    current_words.extend(para_words)

        # Flush any remaining words on the page
        if current_words:
            chunk_text = " ".join(current_words)
            chunk_counter += 1
            stable_id = hashlib.sha256(
                f"{doc.source}:{p.page}:{current_section}:{chunk_counter}".encode()
            ).hexdigest()[:16]

            chunks.append(Chunk(
                id=f"{doc.source[:12]}-{stable_id}",
                text=chunk_text,
                source=doc.source,
                section=current_section,
                page=p.page,
                doc_version=doc.doc_version,
                superseded=doc.superseded,
            ))

    return chunks
