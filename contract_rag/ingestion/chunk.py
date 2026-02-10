from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List

from contract_rag.utils.textnorm import sentencize


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


def chunk_document(doc: Dict[str, str], target_size: int = 900, overlap: int = 120) -> List[Chunk]:
    sentences = sentencize(doc["text"])
    merged = list(_merge_until_limit(sentences, target_size))

    chunks: List[Chunk] = []
    for idx, text in enumerate(merged):
        chunk_id = f"{doc['id']}_{idx}"
        chunks.append(Chunk(id=chunk_id, source=doc["source"], text=text, ordinal=idx))
    # Apply simple overlap by prepending tail of previous chunk
    if overlap and chunks:
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1].text[-overlap:]
            chunks[i].text = prev_tail + " " + chunks[i].text
    return chunks


def chunk_all(docs: List[Dict[str, str]], target_size: int, overlap: int) -> List[Chunk]:
    chunk_lists = [chunk_document(doc, target_size, overlap) for doc in docs]
    return list(itertools.chain.from_iterable(chunk_lists))
