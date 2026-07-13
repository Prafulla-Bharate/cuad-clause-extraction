import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data_loader import extract_text_from_pdf, sample_contracts
from src.extraction import extract_all_clauses
from src.llm_client import GeminiClient
from src.preprocess import normalize_text
from src.summarization import summarize_contract


def run_pipeline(
    pdf_dir: str,
    output_path: str,
    n_contracts: int = 50,
    seed: int = 42,
    model: str = None,
    batch_start: int = 0,
    batch_size: int | None = None,
    reset_output: bool = False,
) -> tuple[list[dict], dict]:
    contract_paths = sample_contracts(
        pdf_dir,
        n=n_contracts,
        seed=seed,
        start=batch_start,
        batch_size=batch_size,
    )
    client = GeminiClient(model=model) if model else GeminiClient()

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    if reset_output and output_path_obj.exists():
        output_path_obj.unlink()

    existing_results = _load_existing_results(output_path_obj)
    results_by_id = dict(existing_results)
    processed_count = 0
    successful_count = 0
    for path in tqdm(contract_paths, desc="Processing contracts"):
        contract_id = path.stem
        existing_row = results_by_id.get(contract_id)
        if existing_row is not None and _is_completed(existing_row):
            continue

        processed_count += 1
        try:
            raw_text = extract_text_from_pdf(path)
            text = normalize_text(raw_text)

            if not text.strip():
                results_by_id[contract_id] = _empty_row(contract_id, error="empty_extraction")
                continue

            clauses = extract_all_clauses(client, text)
            summary = summarize_contract(client, text)

            results_by_id[contract_id] = {
                "contract_id": contract_id,
                "summary": summary,
                "termination_clause": clauses["termination"]["clause_text"],
                "confidentiality_clause": clauses["confidentiality"]["clause_text"],
                "liability_clause": clauses["liability"]["clause_text"],
            }
            successful_count += 1
        except Exception as e:
            results_by_id[contract_id] = _empty_row(contract_id, error=str(e))

    results = list(results_by_id.values())
    _save_results(results, output_path)
    return results, {
        "processed_count": processed_count,
        "successful_count": successful_count,
        "rows_in_output": len(results),
    }


def _empty_row(contract_id: str, error: str = "") -> dict:
    return {
        "contract_id": contract_id,
        "summary": "",
        "termination_clause": "",
        "confidentiality_clause": "",
        "liability_clause": "",
        "error": error,
    }


def _is_completed(row: dict) -> bool:
    if not row:
        return False
    if str(row.get("error", "")).strip():
        return False
    summary = str(row.get("summary", "") or "").strip()
    termination = str(row.get("termination_clause", "") or "").strip()
    confidentiality = str(row.get("confidentiality_clause", "") or "").strip()
    liability = str(row.get("liability_clause", "") or "").strip()
    return bool(summary or termination or confidentiality or liability)


def _load_existing_results(output_path: Path) -> dict:
    if not output_path.exists():
        return {}

    if output_path.suffix.lower() == ".json":
        with open(output_path) as f:
            data = json.load(f)
    else:
        try:
            data = pd.read_csv(output_path).to_dict("records")
        except Exception:
            return {}

    return {row.get("contract_id"): row for row in data if row.get("contract_id")}


def _save_results(results: list[dict], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    else:
        pd.DataFrame(results).to_csv(output_path, index=False)
