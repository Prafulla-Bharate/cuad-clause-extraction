"""Part B: 100-150 word contract summary."""

_SUMMARY_PROMPT = """Read the following contract and write a concise summary of 100-150 words \
covering:
- The purpose of the agreement
- Key obligations of each party
- Notable risks or penalties (e.g. liability caps, termination penalties)

Write it as plain prose (no bullet points, no headers). Stay strictly within 100-150 words.

Contract:
\"\"\"
{text}
\"\"\"

Summary:"""


MAX_CHARS_FOR_SUMMARY = 8000


def summarize_contract(client, contract_text: str) -> str:
    truncated = contract_text[:MAX_CHARS_FOR_SUMMARY]
    prompt = _SUMMARY_PROMPT.format(text=truncated)
    summary = client.generate(prompt, temperature=0.2)
    return summary.strip()
