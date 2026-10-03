import re
from typing import Any


# A lightweight content check to keep unrelated PDFs out of the legal QA flow.
# Requiring several signals avoids accepting a PDF merely because it mentions
# words such as "law" or "court" in passing.
LEGAL_TERMS = (
    "act", "section", "subsection", "article", "court", "judgment",
    "judgement", "petitioner", "respondent", "appellant", "plaintiff",
    "defendant", "hereby", "whereas", "tribunal", "ordinance", "statute",
    "regulation", "prosecution", "provision", "constitutional",
)


def is_legal_pdf(extracted: dict[str, Any]) -> tuple[bool, str]:
    """Return whether extracted PDF text has enough legal-document signals."""
    text = extracted.get("text", "")
    if not text or len(re.sub(r"\s+", "", text)) < 80:
        return False, "This PDF has too little readable text to verify. Please upload a text-readable legal PDF."

    normalized = text.lower()
    hits = sum(bool(re.search(rf"\b{re.escape(term)}\b", normalized)) for term in LEGAL_TERMS)
    # A document must contain at least two different legal signals and repeat
    # legal terminology enough to avoid incidental mentions.
    occurrences = sum(len(re.findall(rf"\b{re.escape(term)}\b", normalized)) for term in LEGAL_TERMS)
    if hits < 2 or occurrences < 4:
        return False, "This PDF does not appear to be a legal document. Please upload a legal PDF (such as an Act, judgment, or case document)."
    return True, ""
