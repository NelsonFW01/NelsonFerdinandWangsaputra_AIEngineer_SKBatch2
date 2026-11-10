"""Step 4 - recommendation = transparent rubric + explicit caveats.

The score is NOT produced by the LLM. It is a deterministic, auditable rubric
over the validated extraction, so every point can be traced to evidence and the
CV cannot talk its way into a higher score. The output is a *suggested next
step for a human*, never a hire/reject decision.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ALIASES = {
    "ml": "machine learning", "machine-learning": "machine learning",
    "py": "python", "python3": "python",
    "mysql": "sql", "postgresql": "sql", "postgres": "sql", "t-sql": "sql",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "large language model": "llm", "large language models": "llm", "llms": "llm",
    "llm applications": "llm", "prompting": "prompt engineering",
    "prompt design": "prompt engineering", "fast api": "fastapi",
    "github": "git", "gitlab": "git", "natural language processing": "nlp",
    "rest apis": "rest api", "restful api": "rest api",
}


def normalize_skill(name: str) -> str:
    s = re.sub(r"[\s_]+", " ", name.strip().lower())
    return ALIASES.get(s, s)


def _surface_forms(canonical: str) -> list[str]:
    return [canonical] + [a for a, c in ALIASES.items() if c == canonical]


def load_jd(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _caveat(code: str, severity: str, message: str) -> dict:
    return {"code": code, "severity": severity, "message": message}


def recommend(jd: dict, extraction: dict, derived: dict, pre, ext_result) -> dict:
    skills = {normalize_skill(s["name"]) for s in extraction["skills"]}
    usage_text = " \n".join(
        " ".join([e.get("title") or ""] + e.get("highlights", [])) for e in extraction["experience"]
    ).lower()

    def used(canon: str) -> bool:
        return any(re.search(rf"(?<![a-z0-9]){re.escape(f)}(?![a-z0-9])", usage_text)
                   for f in _surface_forms(canon))

    def match(required: list[str]) -> dict:
        r = {"strong": [], "listed_only": [], "missing": []}
        for req in required:
            c = normalize_skill(req)
            if c in skills and used(c):
                r["strong"].append(req)
            elif c in skills or used(c):
                r["listed_only"].append(req)
            else:
                r["missing"].append(req)
        return r

    def cover(m: dict, total: int) -> float:
        return (len(m["strong"]) + 0.6 * len(m["listed_only"])) / total if total else 0.0

    must, nice = match(jd["must_have"]), match(jd["nice_to_have"])
    must_s, nice_s = cover(must, len(jd["must_have"])), cover(nice, len(jd["nice_to_have"]))
    exp_s = min(derived["total_months"] / jd["preferred_months_experience"], 1.0)

    edu_text = " ".join(f"{e.get('field') or ''} {e.get('degree') or ''}" for e in extraction["education"]).lower()
    if any(f in edu_text for f in jd["relevant_education_fields"]):
        edu_s = 1.0
    elif extraction["education"]:
        edu_s = 0.4
    else:
        edu_s = 0.0

    w = jd["weights"]
    score = round(100 * (w["must_have"] * must_s + w["nice_to_have"] * nice_s
                         + w["experience"] * exp_s + w["education"] * edu_s), 1)
    b = jd["bands"]
    band = ("strong_match" if score >= b["strong_match"] else
            "possible_match" if score >= b["possible_match"] else
            "weak_match" if score >= b["weak_match"] else "low_match")

    # ---------------- caveats ----------------
    cav = [_caveat("HUMAN_DECISION_REQUIRED", "info",
                   "Ini saran awal berbasis rubric heuristik yang belum divalidasi terhadap hasil rekrutmen "
                   "nyata. Keputusan akhir harus dibuat recruiter setelah membaca CV asli.")]
    if pre.removed_sensitive_lines:
        cav.append(_caveat("SENSITIVE_FIELDS_REMOVED", "info",
                           f"{pre.removed_sensitive_lines} baris atribut sensitif (mis. usia/gender/agama) "
                           "dibuang sebelum diproses dan tidak dipakai dalam penilaian."))
    if pre.injection_flags:
        cav.append(_caveat("INJECTION_TEXT_DETECTED", "high",
                           f"CV memuat {len(pre.injection_flags)} baris yang menyerupai instruksi ke AI "
                           "(dibuang dari input). Periksa CV secara manual."))
    if pre.zero_width_removed:
        cav.append(_caveat("HIDDEN_CHARACTERS_REMOVED", "warn",
                           f"{pre.zero_width_removed} karakter tersembunyi (zero-width) dihapus dari CV."))
    if pre.truncated:
        cav.append(_caveat("CV_TRUNCATED", "warn", "CV terlalu panjang dan dipotong; sebagian isi tidak dianalisis."))
    if pre.char_count < 300:
        cav.append(_caveat("SHORT_OR_EMPTY_CV", "high",
                           "Teks CV sangat pendek (mungkin hasil scan/gambar atau CV tidak lengkap). "
                           "Minta CV yang lebih lengkap."))
    if not ext_result.ok:
        cav.append(_caveat("EXTRACTION_FAILED", "high",
                           "Ekstraksi otomatis gagal setelah beberapa percobaan; tidak ada data terstruktur."))
    elif ext_result.attempts > 1:
        cav.append(_caveat("EXTRACTION_REPAIR_USED", "warn",
                           f"Output model baru valid pada percobaan ke-{ext_result.attempts}."))
    if ext_result.dropped:
        cav.append(_caveat("UNGROUNDED_ITEMS_DROPPED", "warn",
                           f"{len(ext_result.dropped)} item hasil ekstraksi dibuang karena bukti tidak ditemukan di CV."))
    n_items = len(extraction["skills"]) + len(extraction["experience"]) + len(extraction["education"])
    if ext_result.ok and pre.char_count >= 300 and (not extraction["skills"] or
                                                    (not extraction["experience"] and not extraction["education"])):
        cav.append(_caveat("EXTRACTION_LOW_COVERAGE", "high",
                           "CV cukup panjang tetapi hampir tidak ada data yang berhasil diekstrak; skor tidak "
                           "dapat dipercaya (format CV mungkin tidak umum). Baca CV secara manual."))
    if derived["entries_missing_dates"] or derived["entries_invalid_range"]:
        cav.append(_caveat("MISSING_DATES", "warn",
                           f"{derived['entries_missing_dates'] + derived['entries_invalid_range']} pengalaman tanpa "
                           "tanggal valid; durasi pengalaman kemungkinan terhitung terlalu rendah."))
    if any("year precision" in n.lower() or "year-only" in n.lower() for n in extraction["extraction_notes"]):
        cav.append(_caveat("YEAR_ONLY_DATES", "info", "Sebagian tanggal hanya berupa tahun; durasi adalah perkiraan."))
    if derived["gaps_over_6_months"]:
        cav.append(_caveat("EMPLOYMENT_GAP", "info",
                           f"Ada {derived['gaps_over_6_months']} jeda >6 bulan antar pengalaman. "
                           "Ini bukan indikator negatif; tanyakan saat wawancara jika relevan."))
    if must["listed_only"]:
        cav.append(_caveat("SKILLS_LISTED_WITHOUT_USAGE_EVIDENCE", "warn",
                           f"Skill berikut hanya tercantum di daftar tanpa bukti pemakaian di pengalaman/proyek: "
                           f"{', '.join(must['listed_only'])}. Verifikasi saat screening."))
    if must["missing"]:
        cav.append(_caveat("MUST_HAVE_MISSING", "warn",
                           f"Requirement utama tidak ditemukan di CV: {', '.join(must['missing'])}. "
                           "Bisa jadi memang tidak ada, atau tidak ditulis."))

    sev = [c["severity"] for c in cav]
    confidence = ("low" if "high" in sev or sev.count("warn") >= 3 else
                  "medium" if "warn" in sev else "high")
    codes = {c["code"] for c in cav}
    if codes & {"EXTRACTION_FAILED", "SHORT_OR_EMPTY_CV"}:
        next_step = "request_more_information"
    elif confidence == "low":
        next_step = "manual_review_required"
    else:
        next_step = {"strong_match": "prioritize_human_review",
                     "possible_match": "standard_human_review"}.get(band, "low_priority_human_review")

    strengths, gaps = [], []
    if must["strong"]:
        strengths.append(f"Requirement utama dengan bukti pemakaian: {', '.join(must['strong'])}.")
    if nice["strong"] or nice["listed_only"]:
        strengths.append(f"Nilai tambah ditemukan: {', '.join(nice['strong'] + nice['listed_only'])}.")
    if derived["total_months"]:
        strengths.append(f"Pengalaman terdeteksi {derived['total_months']} bulan "
                         f"(termasuk magang {derived['internship_months']} bulan).")
    if edu_s == 1.0:
        strengths.append("Latar pendidikan relevan dengan posisi.")
    if must["listed_only"]:
        gaps.append(f"Hanya tercantum tanpa bukti pemakaian: {', '.join(must['listed_only'])}.")
    if must["missing"]:
        gaps.append(f"Belum terlihat: {', '.join(must['missing'])}.")
    if derived["total_months"] < jd["preferred_months_experience"]:
        gaps.append(f"Pengalaman {derived['total_months']} bulan < target {jd['preferred_months_experience']} bulan.")

    return {
        "job_title": jd["title"],
        "fit_score": score,
        "band": band,
        "suggested_next_step": next_step,
        "confidence": confidence,
        "requires_human_review": True,
        "strengths": strengths,
        "gaps": gaps,
        "caveats": cav,
        "score_breakdown": {
            "must_have": {"weight": w["must_have"], "coverage": round(must_s, 2), **must},
            "nice_to_have": {"weight": w["nice_to_have"], "coverage": round(nice_s, 2), **nice},
            "experience": {"weight": w["experience"], "coverage": round(exp_s, 2), "months": derived["total_months"]},
            "education": {"weight": w["education"], "coverage": edu_s},
        },
        "disclaimer": "Bukan keputusan rekrutmen. Tidak ada kandidat yang ditolak otomatis oleh sistem ini.",
    }
