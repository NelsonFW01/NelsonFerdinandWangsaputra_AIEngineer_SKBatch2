"""Step 3 - deterministic derivations. Arithmetic is done in code, never by the LLM."""
from __future__ import annotations


def _idx(ym: str) -> int:
    return int(ym[:4]) * 12 + int(ym[5:7]) - 1


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1] + 1:  # overlapping or back-to-back
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def compute_experience(experience: list[dict], as_of_ym: str) -> dict:
    """Months of experience (overlaps counted once; both start & end month inclusive)."""
    as_of = _idx(as_of_ym)
    all_iv: list[tuple[int, int]] = []
    intern_iv: list[tuple[int, int]] = []
    missing = invalid = 0
    for e in experience:
        s, en = e.get("start"), e.get("end")
        if not s or not en:
            missing += 1
            continue
        a = _idx(s)
        b = as_of if en == "present" else min(_idx(en), as_of)
        if b < a:
            invalid += 1
            continue
        all_iv.append((a, b))
        if e.get("is_internship"):
            intern_iv.append((a, b))
    merged = _merge(all_iv)
    total = sum(b - a + 1 for a, b in merged)
    gaps = sum(1 for (_, b1), (a2, _) in zip(merged, merged[1:]) if a2 - b1 - 1 > 6)
    return {
        "total_months": total,
        "total_years": round(total / 12, 1),
        "internship_months": sum(b - a + 1 for a, b in _merge(intern_iv)),
        "entries_total": len(experience),
        "entries_missing_dates": missing,
        "entries_invalid_range": invalid,
        "gaps_over_6_months": gaps,
        "as_of": as_of_ym,
    }
