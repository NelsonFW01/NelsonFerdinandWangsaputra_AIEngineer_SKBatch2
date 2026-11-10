"""Step 1 - deterministic pre-processing BEFORE any text reaches an LLM.

1. Strip zero-width / control characters (hidden-text tricks).
2. Pull contact data out locally, then redact it from the LLM input (privacy).
3. Drop lines carrying protected / sensitive attributes (fairness).
4. Detect and remove instruction-like text (prompt-injection defence, layer 1).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Indonesian mobile numbers (+62 / 62 / 08..) - deliberately narrow so year
# ranges like "2020 - 2024" are never matched. Known limitation: other formats.
PHONE_RE = re.compile(r"(?<!\d)(?:\+62|62|0)\s?8[\d\s\-]{7,13}\d(?!\d)")

SENSITIVE_LINE_RE = re.compile(
    r"^\s*(tanggal lahir|tempat,? ?tanggal lahir|ttl|date of birth|dob|birth ?date|usia|age|"
    r"jenis kelamin|gender|sex|agama|religion|status (pernikahan|perkawinan)|marital status|"
    r"kewarganegaraan|nationality|tinggi badan|berat badan|height|weight)\s*[:\-]",
    re.IGNORECASE,
)

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions?|prompts?)",
    r"disregard (all |any )?(the )?(previous|prior|above)",
    r"abaikan (semua )?(instruksi|perintah)",
    r"\b(you are|kamu adalah|anda adalah) now\b",
    r"system prompt",
    r"rate this candidate",
    r"(nilai|beri skor|beri nilai) kandidat",
    r"recommend (this candidate|immediate)",
    r"\bjailbreak\b",
]
_INJECTION_RES = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

MAX_CHARS = 15000  # hard cap on what we send to the model


@dataclass
class PreprocessResult:
    text_for_llm: str
    contact: dict = field(default_factory=dict)  # kept locally, NOT sent to the LLM
    removed_sensitive_lines: int = 0
    injection_flags: list = field(default_factory=list)
    zero_width_removed: int = 0
    truncated: bool = False
    char_count: int = 0  # chars of meaningful CV text after cleaning


def preprocess(raw: str) -> PreprocessResult:
    zw = len(_ZERO_WIDTH_RE.findall(raw))
    text = _CONTROL_RE.sub("", _ZERO_WIDTH_RE.sub("", raw))
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    contact = {
        "emails": sorted(set(EMAIL_RE.findall(text))),
        "phones": sorted(set(m.group(0).strip() for m in PHONE_RE.finditer(text))),
    }

    out_lines: list[str] = []
    removed = 0
    flags: list[dict] = []
    for ln in text.split("\n"):
        if SENSITIVE_LINE_RE.match(ln):
            removed += 1
            continue
        hit = next((r for r in _INJECTION_RES if r.search(ln)), None)
        if hit is not None:
            flags.append({"pattern": hit.pattern, "snippet": ln.strip()[:80]})
            continue  # line is dropped from LLM input
        ln = EMAIL_RE.sub("[EMAIL]", ln)
        ln = PHONE_RE.sub("[PHONE]", ln)
        out_lines.append(ln.rstrip())

    cleaned = "\n".join(out_lines).strip()
    truncated = len(cleaned) > MAX_CHARS
    if truncated:
        cleaned = cleaned[:MAX_CHARS]
    return PreprocessResult(
        text_for_llm=cleaned,
        contact=contact,
        removed_sensitive_lines=removed,
        injection_flags=flags,
        zero_width_removed=zw,
        truncated=truncated,
        char_count=len(cleaned),
    )
