import re

_PAGE_NUM_RE = re.compile(r"^\s*(page\s+\d+(\s+of\s+\d+)?|-?\s*\d{1,3}\s*-?)\s*$", re.IGNORECASE)
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    lines = [ln for ln in text.split("\n") if not _PAGE_NUM_RE.match(ln)]
    text = "\n".join(lines)

    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 3500, overlap: int = 300) -> list[str]:

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks
