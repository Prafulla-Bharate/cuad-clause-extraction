import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data_loader import extract_text_from_pdf, sample_contracts
from src.extraction import extract_all_clauses
from src.llm_client import GeminiClient, RateLimitError
from src.preprocess import normalize_text
from src.summarization import summarize_contract


def run_pipeline(
    pdf_dir: str,
    output_path: str,
    n_contracts: int = 50,
    seed: int = 42,
    model: str = None,
    reset_output: bool = False,
) -> tuple[list[dict], dict]:
    contract_paths = sample_contracts(
        pdf_dir,
        n=n_contracts,
        seed=seed,
    )
    client = GeminiClient(model=model) if model else GeminiClient()

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    if reset_output and output_path_obj.exists():
        try:
            output_path_obj.unlink()
        except PermissionError:
            pass

    existing_results = _get_existing_results(output_path_obj, reset_output)
    results_by_id = dict(existing_results)
    processed_count = 0
    successful_count = 0
    skipped_count = 0
    stopped_early = False
    stop_reason = ""

    for path in tqdm(contract_paths, desc="Processing contracts"):
        contract_id = str(path.stem or "").strip()
        existing_row = results_by_id.get(contract_id)
        if existing_row is not None and _is_completed(existing_row):
            skipped_count += 1
            continue

        processed_count += 1
        try:
            raw_text = extract_text_from_pdf(path)
            text = normalize_text(raw_text)

            if not text.strip():
                results_by_id[contract_id] = _merge_results(
                    existing_row,
                    _empty_row(contract_id, error="empty_extraction"),
                )
                _checkpoint_results(results_by_id, output_path)
                continue

            clauses = extract_all_clauses(client, text)
            summary = summarize_contract(client, text)

            results_by_id[contract_id] = _merge_results(
                existing_row,
                {
                    "contract_id": contract_id,
                    "summary": summary,
                    "termination_clause": clauses["termination"]["clause_text"],
                    "confidentiality_clause": clauses["confidentiality"]["clause_text"],
                    "liability_clause": clauses["liability"]["clause_text"],
                },
            )
            successful_count += 1
            _checkpoint_results(results_by_id, output_path)
        except Exception as e:
            results_by_id[contract_id] = _merge_results(
                existing_row,
                _empty_row(contract_id, error=str(e)),
            )
            _checkpoint_results(results_by_id, output_path)
            if isinstance(e, RateLimitError):
                stopped_early = True
                stop_reason = "api_rate_limit"
                break

    results = list(results_by_id.values())
    _save_results(results, output_path)
    return results, {
        "processed_count": processed_count,
        "successful_count": successful_count,
        "skipped_count": skipped_count,
        "rows_in_output": len(results),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }


def _empty_row(contract_id: str, error: str = "") -> dict:
    return {
        "contract_id": contract_id,
        "summary": "",
        "termination_clause": "Clause Not Present",
        "confidentiality_clause": "Clause Not Present",
        "liability_clause": "Clause Not Present",
        "error": error,
    }


def _get_existing_results(output_path: Path, reset_output: bool) -> dict:
    if reset_output:
        return {}
    return _load_existing_results(output_path)


def _is_completed(row: dict) -> bool:
    if not row:
        return False
    error = row.get("error", "")
    if pd.isna(error):
        error = ""
    if str(error).strip():
        return False
    summary = str(row.get("summary", "") or "").strip()
    termination = str(row.get("termination_clause", "") or "").strip()
    confidentiality = str(row.get("confidentiality_clause", "") or "").strip()
    liability = str(row.get("liability_clause", "") or "").strip()
    return bool(summary or termination or confidentiality or liability)


def _merge_results(existing_row: dict | None, new_row: dict) -> dict:
    if existing_row is None:
        return new_row
    if _is_completed(existing_row) and not _is_completed(new_row):
        return existing_row
    return new_row


def _load_existing_results(output_path: Path) -> dict:
    if not output_path.exists():
        return {}

    if output_path.suffix.lower() == ".json":
        with open(output_path) as f:
            data = json.load(f)
    else:
        try:
            data = pd.read_csv(
                output_path,
                dtype=str,
                keep_default_na=False,
                on_bad_lines="skip",
            ).fillna("").to_dict("records")
        except Exception:
            try:
                import csv

                with output_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    data = [
                        {k: (v or "").strip() for k, v in row.items()}
                        for row in reader
                    ]
            except Exception:
                return {}

    results = {}
    for row in data:
        contract_id = str(row.get("contract_id", "") or "").strip()
        if contract_id:
            row["contract_id"] = contract_id
            results[contract_id] = row
    return results


def _save_results(results: list[dict], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".json":
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with open(temp_path, "w") as f:
            json.dump(results, f, indent=2)
        temp_path.replace(output_path)
    else:
        # Write the CSV atomically, retrying briefly if the file is locked (Windows handles).
        csv_df = pd.DataFrame(results)
        csv_df = csv_df.reindex(
            columns=[
                "contract_id",
                "summary",
                "termination_clause",
                "confidentiality_clause",
                "liability_clause",
            ]
        ).fillna("")
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        max_attempts = 6
        for attempt in range(1, max_attempts + 1):
            try:
                csv_df.to_csv(temp_path, index=False)
                temp_path.replace(output_path)
                break
            except PermissionError:
                if attempt == max_attempts:
                    # Give up after a few retries so the caller can see the error.
                    raise
                time.sleep(1)


def _checkpoint_results(results_by_id: dict, output_path: str):
    _save_results(list(results_by_id.values()), output_path)
