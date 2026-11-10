"""Glue: preprocess -> extract (LLM) -> ground -> derive -> recommend."""
from __future__ import annotations

import hashlib

from .derive import compute_experience
from .extract import extract
from .preprocess import preprocess
from .recommend import recommend


def run_pipeline(cv_text: str, llm, jd: dict, as_of: str = "2026-11") -> dict:
    pre = preprocess(cv_text)
    res = extract(llm, pre.text_for_llm)
    derived = compute_experience(res.data["experience"], as_of)
    rec = recommend(jd, res.data, derived, pre, res)
    return {
        "candidate_id": hashlib.sha256(pre.text_for_llm.encode()).hexdigest()[:10],
        "extraction": res.data,
        "derived": derived,
        "recommendation": rec,
        "contact_local_only": pre.contact,  # never sent to the LLM; restrict access in real use
        "pipeline_meta": {
            "llm": getattr(llm, "name", "unknown"),
            "extraction_ok": res.ok,
            "attempts": res.attempts,
            "validation_errors": res.errors,
            "ungrounded_items_dropped": res.dropped,
            "items_before_grounding": res.items_before_grounding,
            "sensitive_lines_removed": pre.removed_sensitive_lines,
            "injection_flags": pre.injection_flags,
            "chars_sent_to_llm": pre.char_count,
        },
    }
