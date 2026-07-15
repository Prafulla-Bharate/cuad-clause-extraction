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
) -> list[Path]:
   
    all_pdfs = list_contract_pdfs(pdf_dir)
    if len(all_pdfs) == 0:
        raise ValueError(f"No PDFs found in {pdf_dir}")

    unique_paths = []
    seen_stems = set()
    for path in all_pdfs:
        if path.stem not in seen_stems:
            seen_stems.add(path.stem)
            unique_paths.append(path)

    if len(unique_paths) <= n:
        return unique_paths

    rng = random.Random(seed)
    return rng.sample(unique_paths, n)


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
