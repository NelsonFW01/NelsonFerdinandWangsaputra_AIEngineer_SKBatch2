"""Output schema for the extraction step + a small stdlib-only validator.

Design notes
- Every claim the LLM makes carries an `evidence` string that must be a verbatim
  snippet of the CV. The pipeline later checks that the snippet really exists
  (see extract.ground) and drops anything it cannot verify.
- Contact data (email/phone) and protected attributes (age, gender, religion,
  marital status, ...) are deliberately NOT part of the schema.
"""
from __future__ import annotations

import re
from typing import Any

YM_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

# JSON-Schema-style description. Used in the prompt and can be passed to
# providers that support structured output.
EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "full_name", "location", "summary", "education", "experience",
        "skills", "certifications", "languages", "extraction_notes",
    ],
    "properties": {
        "full_name": {"type": ["string", "null"]},
        "location": {"type": ["string", "null"], "description": "City/region only"},
        "summary": {"type": ["string", "null"]},
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["institution", "degree", "field", "end_year", "evidence"],
                "properties": {
                    "institution": {"type": ["string", "null"]},
                    "degree": {"type": ["string", "null"], "description": "e.g. S1, D3, Bachelor"},
                    "field": {"type": ["string", "null"]},
                    "end_year": {"type": ["integer", "null"]},
                    "evidence": {"type": "string", "description": "verbatim CV snippet"},
                },
            },
        },
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "company", "start", "end", "is_internship", "highlights", "evidence"],
                "properties": {
                    "title": {"type": ["string", "null"]},
                    "company": {"type": ["string", "null"]},
                    "start": {"type": ["string", "null"], "description": "YYYY-MM or null"},
                    "end": {"type": ["string", "null"], "description": "YYYY-MM, 'present', or null"},
                    "is_internship": {"type": "boolean"},
                    "highlights": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "string", "description": "verbatim CV snippet"},
                },
            },
        },
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "evidence"],
                "properties": {"name": {"type": "string"}, "evidence": {"type": "string"}},
            },
        },
        "certifications": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "evidence"],
                "properties": {"name": {"type": "string"}, "evidence": {"type": "string"}},
            },
        },
        "languages": {"type": "array", "items": {"type": "string"}},
        "extraction_notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ambiguities / things the extractor was unsure about",
        },
    },
}


def empty_extraction() -> dict[str, Any]:
    return {
        "full_name": None, "location": None, "summary": None,
        "education": [], "experience": [], "skills": [], "certifications": [],
        "languages": [], "extraction_notes": [],
    }


def _str_or_null(v: Any) -> bool:
    return v is None or isinstance(v, str)


def validate_extraction(d: Any) -> list[str]:
    """Return a list of human-readable validation errors ([] means valid)."""
    if not isinstance(d, dict):
        return ["root must be a JSON object"]
    errs: list[str] = []
    for k in EXTRACTION_JSON_SCHEMA["required"]:
        if k not in d:
            errs.append(f"missing key: {k}")
    if errs:
        return errs

    for k in ("full_name", "location", "summary"):
        if not _str_or_null(d[k]):
            errs.append(f"{k} must be string or null")

    def _list(key: str) -> list:
        if not isinstance(d[key], list):
            errs.append(f"{key} must be an array")
            return []
        return d[key]

    for i, e in enumerate(_list("education")):
        p = f"education[{i}]"
        if not isinstance(e, dict):
            errs.append(f"{p} must be object"); continue
        for k in ("institution", "degree", "field"):
            if not _str_or_null(e.get(k)):
                errs.append(f"{p}.{k} must be string or null")
        y = e.get("end_year")
        if y is not None and not (isinstance(y, int) and not isinstance(y, bool) and 1950 <= y <= 2100):
            errs.append(f"{p}.end_year must be int 1950-2100 or null")
        if not (isinstance(e.get("evidence"), str) and e["evidence"].strip()):
            errs.append(f"{p}.evidence must be non-empty string")

    for i, e in enumerate(_list("experience")):
        p = f"experience[{i}]"
        if not isinstance(e, dict):
            errs.append(f"{p} must be object"); continue
        for k in ("title", "company"):
            if not _str_or_null(e.get(k)):
                errs.append(f"{p}.{k} must be string or null")
        if e.get("title") is None and e.get("company") is None:
            errs.append(f"{p} needs title or company")
        s, en = e.get("start"), e.get("end")
        if s is not None and not (isinstance(s, str) and YM_RE.match(s)):
            errs.append(f"{p}.start must be YYYY-MM or null")
        if en is not None and not (isinstance(en, str) and (YM_RE.match(en) or en == "present")):
            errs.append(f"{p}.end must be YYYY-MM, 'present' or null")
        if not isinstance(e.get("is_internship"), bool):
            errs.append(f"{p}.is_internship must be boolean")
        h = e.get("highlights")
        if not (isinstance(h, list) and all(isinstance(x, str) for x in h)):
            errs.append(f"{p}.highlights must be array of strings")
        if not (isinstance(e.get("evidence"), str) and e["evidence"].strip()):
            errs.append(f"{p}.evidence must be non-empty string")

    for key in ("skills", "certifications"):
        for i, e in enumerate(_list(key)):
            p = f"{key}[{i}]"
            if not isinstance(e, dict):
                errs.append(f"{p} must be object"); continue
            for k in ("name", "evidence"):
                if not (isinstance(e.get(k), str) and e[k].strip()):
                    errs.append(f"{p}.{k} must be non-empty string")

    for key in ("languages", "extraction_notes"):
        v = _list(key)
        if not all(isinstance(x, str) for x in v):
            errs.append(f"{key} must be array of strings")
    return errs
