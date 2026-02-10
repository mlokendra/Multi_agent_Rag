from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Dict

from contract_rag.utils.textnorm import normalize_text

logger = logging.getLogger(__name__)


SUPPORTED_EXTS = {".txt", ".pdf", ".docx"}


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    try:
        import pypdf
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("pypdf not installed; skipping %s", path.name)
        return ""
    text_parts: List[str] = []
    with path.open("rb") as fh:
        pdf = pypdf.PdfReader(fh)
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def _read_docx(path: Path) -> str:
    try:
        import docx
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("python-docx not installed; skipping %s", path.name)
        return ""
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


READERS = {
    ".txt": _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


def load_documents(data_dir: Path) -> List[Dict[str, str]]:
    """Load supported documents from a directory into a list of dicts.

    Returns [{id, source, text}]. Skips files that cannot be parsed.
    """
    docs: List[Dict[str, str]] = []
    for path in sorted(data_dir.glob("**/*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue
        text = READERS[ext](path)
        text = normalize_text(text)
        if not text:
            logger.warning("No text extracted from %s", path.name)
            continue
        docs.append({"id": path.stem, "source": path.name, "text": text})
    return docs
