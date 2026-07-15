"""Part A: clause extraction (termination, confidentiality, liability).

Strategy for handling long contracts without blowing up token usage:
1. Cheap keyword pre-filter narrows the chunk list down to ones that plausibly
   contain a given clause type.
2. Only those chunks get sent to the LLM, one clause type at a time, with a
   short few-shot example so the model knows the expected output shape.
3. If nothing matches, we skip the LLM call entirely and mark it "Not found".
"""

from src.preprocess import chunk_text

CLAUSE_TYPES = ["termination", "confidentiality", "liability"]

# Keywords used to shortlist candidate chunks per clause type. These are broader
# to improve recall while still avoiding full-document calls in the common case.
_KEYWORDS = {
    "termination": [
        "terminat",
        "expire",
        "expiration",
        "cancel",
        "cancellation",
        "notice",
        "notice of default",
        "default",
        "breach",
        "suspend",
        "renewal",
        "renew",
        "for cause",
        "without cause",
        "immediately",
    ],
    "confidentiality": [
        "confidential",
        "confidentiality",
        "non-disclosure",
        "non disclosure",
        "proprietary",
        "trade secret",
        "secret",
        "nda",
        "disclos",
        "use or disclose",
        "privacy",
        "protected information",
        "restricted",
        "recipient",
    ],
    "liability": [
        "liability",
        "indemnif",
        "damages",
        "limitation of liability",
        "hold harmless",
        "cap",
        "waiver",
        "warranty",
        "losses",
        "claims",
        "consequential",
        "liquidated damages",
        "negligence",
        "attorney fees",
        "tort",
        "in no event",
    ],
}

_MAX_CHARS_FOR_HEURISTIC = 6000

_CLAUSE_PROMPT_INSTRUCTIONS = {
    "termination": (
        "Focus on wording that grants either party the right to terminate the agreement, "
        "including notice periods, cure periods, material breach, automatic termination, "
        "and renewal/cancellation conditions."
    ),
    "confidentiality": (
        "Focus on wording that defines confidential information, obligations to keep "
        "information secret, permitted disclosures, third-party recipients, and the "
        "survival of confidentiality obligations."
    ),
    "liability": (
        "Focus on wording that allocates financial responsibility, indemnity, damage caps, "
        "warranties, exclusions, liquidated damages, and limits on consequential loss."
    ),
}

# One worked example per clause type, used as a few-shot anchor in the prompt.
_FEW_SHOT_EXAMPLES = {
    "termination": (
        "Excerpt: \"Either party may terminate this Agreement upon thirty (30) "
        "days' written notice to the other party. In the event of a material "
        "breach that remains uncured for fifteen (15) days after notice, the "
        "non-breaching party may terminate immediately.\"\n"
        'Expected output: {"found": true, "clause_text": "Either party may '
        "terminate this Agreement upon thirty (30) days' written notice... "
        "the non-breaching party may terminate immediately.\", "
        '"summary": "Either party can terminate with 30 days notice, or '
        'immediately for an uncured material breach after 15 days."}'
    ),
    "confidentiality": (
        "Excerpt: \"Each party agrees to hold the other's Confidential "
        "Information in strict confidence and not to disclose it to any "
        "third party without prior written consent, for a period of five "
        "(5) years following termination.\"\n"
        'Expected output: {"found": true, "clause_text": "Each party agrees '
        "to hold the other's Confidential Information in strict confidence... "
        "for a period of five (5) years following termination.\", "
        '"summary": "Mutual confidentiality obligation surviving 5 years '
        'post-termination."}'
    ),
    "liability": (
        "Excerpt: \"In no event shall either party's aggregate liability "
        "exceed the total fees paid under this Agreement in the twelve (12) "
        "months preceding the claim.\"\n"
        'Expected output: {"found": true, "clause_text": "In no event shall '
        "either party's aggregate liability exceed the total fees paid... "
        "in the twelve (12) months preceding the claim.\", "
        '"summary": "Liability is capped at fees paid in the prior 12 months."}'
    ),
}

_PROMPT_TEMPLATE = """You are a contract review assistant. Find the {clause_type} clause in the contract excerpt below and return ONLY valid JSON (no markdown fences).

Important:
- clause_text must reproduce the clause wording exactly as it appears in the excerpt.
- Do not paraphrase or summarize clause_text.
- Retain original punctuation and legal phrasing.
- summary may explain the clause in plain English.

Clause type instructions:
{clause_instructions}

Example:
{few_shot}

Output schema:
{{"found": true/false, "clause_text": "<verbatim or lightly trimmed quote from the excerpt, \
or 'Clause Not Present' if no relevant clause exists>", "summary": "<one-sentence plain-English summary, or 'Clause Not Present' if no relevant clause exists>"}}

If the contract excerpt contains multiple relevant provisions, return the most representative combined clause_text and one concise summary. If there is no relevant clause, return the exact marker 'Clause Not Present'.
\"\"\"
{excerpt}
\"\"\"

JSON output:"""


def _find_relevant_chunks(chunks: list[str], clause_type: str) -> list[str]:
    keywords = _KEYWORDS[clause_type]
    scored_chunks = []
    for chunk in chunks:
        lower = chunk.lower()
        score = sum(lower.count(kw) for kw in keywords)
        if score > 0:
            scored_chunks.append((score, chunk))
    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored_chunks]


def _call_clause_extractor(client, clause_type: str, excerpt: str) -> dict:
    prompt = _PROMPT_TEMPLATE.format(
        clause_type=clause_type,
        clause_instructions=_CLAUSE_PROMPT_INSTRUCTIONS[clause_type],
        few_shot=_FEW_SHOT_EXAMPLES[clause_type],
        excerpt=excerpt,
    )
    try:
        return client.generate_json(prompt)
    except Exception:
        return {"error": "parse_failed"}


def _normalize_clause_result(result: dict) -> dict:
    clause_text = str(result.get("clause_text") or "").strip()
    summary = str(result.get("summary") or "").strip()
    found_flag = bool(result.get("found", False)) or bool(clause_text and clause_text != "Clause Not Present")
    return {
        "found": found_flag,
        "clause_text": clause_text or "Clause Not Present",
        "summary": summary or "Clause Not Present",
    }


def _try_clause_extraction(client, clause_type: str, excerpt: str) -> dict | None:
    result = _call_clause_extractor(client, clause_type, excerpt)
    if result.get("error") == "parse_failed":
        return None
    clause_text = str(result.get("clause_text") or "").strip()
    summary = str(result.get("summary") or "").strip()
    if not clause_text and not summary:
        return None
    return _normalize_clause_result(result)


def extract_clause(client, contract_text: str, clause_type: str) -> dict:
    """Extract a single clause type from a contract, returning
    {found, clause_text, summary}."""
    not_found = {"found": False, "clause_text": "Clause Not Present", "summary": "Clause Not Present"}

    text = contract_text[:_MAX_CHARS_FOR_HEURISTIC].lower()
    if not any(keyword in text for keyword in _KEYWORDS[clause_type]):
        return not_found

    chunks = chunk_text(contract_text, chunk_size=2400, overlap=400)
    candidates = _find_relevant_chunks(chunks, clause_type)

    candidate_excerpt = "\n...\n".join(candidates[:5]) if candidates else ""
    result = None
    if candidate_excerpt:
        result = _try_clause_extraction(client, clause_type, candidate_excerpt)

    if not result or not result["found"]:
        # Fallback: try the full normalized contract text if the candidate
        # chunk extraction did not return a usable clause.
        fallback_excerpt = contract_text
        result = _try_clause_extraction(client, clause_type, fallback_excerpt) or result

    return result or not_found


def extract_all_clauses(client, contract_text: str) -> dict:
    return {ct: extract_clause(client, contract_text, ct) for ct in CLAUSE_TYPES}
