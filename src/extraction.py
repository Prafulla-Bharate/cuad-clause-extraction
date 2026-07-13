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

# Keywords used to shortlist candidate chunks per clause type. Not exhaustive,
# just enough to avoid sending an entire 40-page contract to the model for a
# clause that only shows up on page 6.
_KEYWORDS = {
    "termination": ["terminat", "expir", "cancellation", "notice of default"],
    "confidentiality": ["confidential", "non-disclosure", "proprietary information", "nda"],
    "liability": ["liability", "indemnif", "damages", "limitation of liability", "hold harmless"],
}

# Short, cheap heuristic: if a contract has no obvious signal for a clause type,
# skip the LLM entirely and return an empty result. This saves a call on a lot
# of contracts that simply don't contain that clause.
_MAX_CHARS_FOR_HEURISTIC = 6000

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

_PROMPT_TEMPLATE = """You are a contract review assistant. Find the {clause_type} clause \
in the contract excerpt below and return ONLY valid JSON (no markdown fences).

Output schema:
{{"found": true/false, "clause_text": "<verbatim or lightly trimmed quote from the excerpt, \
or empty string if not found>", "summary": "<one-sentence plain-English summary, or empty \
string if not found>"}}

Example:
{few_shot}

Now do the same for this excerpt:
\"\"\"
{excerpt}
\"\"\"

JSON output:"""


def _find_relevant_chunks(chunks: list[str], clause_type: str) -> list[str]:
    keywords = _KEYWORDS[clause_type]
    hits = [c for c in chunks if any(kw in c.lower() for kw in keywords)]
    return hits


def extract_clause(client, contract_text: str, clause_type: str) -> dict:
    """Extract a single clause type from a contract, returning
    {found, clause_text, summary}."""
    text = contract_text[:_MAX_CHARS_FOR_HEURISTIC].lower()
    if not any(keyword in text for keyword in _KEYWORDS[clause_type]):
        return {"found": False, "clause_text": "", "summary": ""}

    chunks = chunk_text(contract_text)
    candidates = _find_relevant_chunks(chunks, clause_type)

    if not candidates:
        return {"found": False, "clause_text": "", "summary": ""}

    # Most contracts only need 1-2 chunks; cap at 2 to keep cost bounded.
    excerpt = "\n...\n".join(candidates[:2])

    prompt = _PROMPT_TEMPLATE.format(
        clause_type=clause_type,
        few_shot=_FEW_SHOT_EXAMPLES[clause_type],
        excerpt=excerpt,
    )

    try:
        result = client.generate_json(prompt)
    except Exception:
        # fall back gracefully if the model returns malformed JSON
        return {"found": False, "clause_text": "", "summary": "", "error": "parse_failed"}

    return {
        "found": result.get("found", False),
        "clause_text": result.get("clause_text", ""),
        "summary": result.get("summary", ""),
    }


def extract_all_clauses(client, contract_text: str) -> dict:
    return {ct: extract_clause(client, contract_text, ct) for ct in CLAUSE_TYPES}
