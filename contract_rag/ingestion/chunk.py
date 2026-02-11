from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple
import re

from contract_rag.utils.textnorm import sentencize, normalize_text


def _merge_until_limit(parts: List[str], limit: int) -> List[str]:
    bucket: List[str] = []
    total = 0
    for part in parts:
        length = len(part)
        if bucket and total + length > limit:
            yield " ".join(bucket)
            bucket = []
            total = 0
        bucket.append(part)
        total += length
    if bucket:
        yield " ".join(bucket)


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    ordinal: int
    section_number: str | None = None
    section_title: str | None = None


# Numbered heading matcher like "1.", "2.3", "4.2.1 Clause Title"
heading_re = re.compile(r"^(?P<num>\d+(?:\.\d+)*)(?:[\s\.)]+)(?P<title>.+)$")


def _split_by_headings(text: str) -> List[Tuple[str | None, str, str]]:
    """Return list of (section_number, section_title, section_text).

    Captures a preamble before the first heading with section_number=None.
    """
    sections: List[Tuple[str | None, str, str]] = []
    current_num: str | None = None
    current_title: str = "Preamble"
    bucket: List[str] = []

    def flush() -> None:
        if not bucket:
            return
        text_block = "\n".join(bucket).strip()
        sections.append((current_num, current_title, text_block))

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        m = heading_re.match(line)
        if m:
            flush()
            current_num = m.group("num")
            current_title = m.group("title").strip()
            bucket = []
            continue
        bucket.append(line)
    flush()
    return [(num, title, body) for num, title, body in sections if body]


def chunk_document(doc: Dict[str, str], target_size: int = 900, overlap: int = 120) -> List[Chunk]:
    sections = _split_by_headings(doc["text"])
    chunks: List[Chunk] = []

    if sections:
        for idx, (sec_num, sec_title, body) in enumerate(sections):
            cleaned = normalize_text(body)
            # Include heading words in the chunk text to aid matching (e.g., "Termination")
            heading_prefix = f"{sec_num} {sec_title}".strip() if sec_num else sec_title
            full_text = f"{heading_prefix}. {cleaned}".strip()
            chunk_id = f"{doc['id']}_{sec_num or idx}"
            source = doc["source"]
            if sec_num:
                source = f"{source} §{sec_num} {sec_title}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    source=source,
                    text=full_text,
                    ordinal=idx,
                    section_number=sec_num,
                    section_title=sec_title,
                )
            )
        return chunks

    # Fallback: length-based sentence merging when headings are absent.
    sentences = sentencize(doc["text"])
    merged = list(_merge_until_limit(sentences, target_size))

    for idx, text in enumerate(merged):
        chunk_id = f"{doc['id']}_{idx}"
        chunks.append(Chunk(id=chunk_id, source=doc["source"], text=text, ordinal=idx))
    if overlap and chunks:
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1].text[-overlap:]
            chunks[i].text = prev_tail + " " + chunks[i].text
    return chunks


def chunk_all(docs: List[Dict[str, str]], target_size: int, overlap: int) -> List[Chunk]:
    chunk_lists = [chunk_document(doc, target_size, overlap) for doc in docs]
    return list(itertools.chain.from_iterable(chunk_lists))
