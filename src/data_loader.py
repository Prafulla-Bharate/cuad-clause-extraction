import random
from pathlib import Path

from pypdf import PdfReader


def list_contract_pdfs(pdf_dir: str) -> list[Path]:
    
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"'{pdf_dir}' doesn't exist. Download the CUAD dataset first "
            "(see README) and point --pdf_dir at the full_contract_pdf folder."
        )
    return sorted(pdf_dir.rglob("*.pdf"))


def sample_contracts(
    pdf_dir: str,
    n: int = 50,
    seed: int = 42,
    start: int = 0,
    batch_size: int | None = None,
) -> list[Path]:
   
    all_pdfs = list_contract_pdfs(pdf_dir)
    if len(all_pdfs) == 0:
        raise ValueError(f"No PDFs found in {pdf_dir}")

    if len(all_pdfs) <= n:
        sampled = all_pdfs
    else:
        rng = random.Random(seed)
        sampled = rng.sample(all_pdfs, n)

    if start < 0:
        raise ValueError("start must be >= 0")
    if batch_size is None:
        batch_size = len(sampled) - start
    if batch_size < 0:
        raise ValueError("batch_size must be >= 0")

    end = min(start + batch_size, len(sampled))
    return sampled[start:end]


def extract_text_from_pdf(path: Path) -> str:
    
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(text)
    return "\n".join(pages)
