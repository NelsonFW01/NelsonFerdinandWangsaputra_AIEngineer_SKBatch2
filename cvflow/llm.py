"""LLM clients behind one tiny interface: generate_json(system, user) -> str.

- MockLLM    : offline, rule-based stand-in (default). Lets the whole flow run with
               no API key, no cost and no real data. It is NOT an LLM - it only
               works on the simple CV layouts in data/cvs and is intentionally
               brittle on free-form text (this is what the evaluation exposes).
- GeminiLLM  : real LLM via Google AI Studio free tier (needs GEMINI_API_KEY).
- OllamaLLM  : real LLM running locally (no key, data never leaves the machine).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request


class FatalLLMError(RuntimeError):
    """Provider refuses to serve us (e.g. quota exhausted); retrying right away is pointless."""


class LLMClient:
    name = "base"

    def generate_json(self, system: str, user: str) -> str:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------
# Real providers (stdlib only, so there is nothing to pip install)
# --------------------------------------------------------------------------
class GeminiLLM(LLMClient):
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 60):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set GEMINI_API_KEY (free key from Google AI Studio).")
        # Model names change over time: override with GEMINI_MODEL if this one is retired.
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.timeout = timeout
        # Free tier has a low requests-per-minute cap: keep a minimum gap between calls.
        self.min_interval = float(os.environ.get("GEMINI_MIN_INTERVAL", "7"))
        self._last_call = 0.0
        self.name = f"gemini:{self.model}"

    def generate_json(self, system: str, user: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "maxOutputTokens": 8192},
        }
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        payload = None
        for attempt in range(3):
            gap = self.min_interval - (time.time() - self._last_call)
            if gap > 0:
                time.sleep(gap)
            self._last_call = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code in (429, 503):
                    if attempt < 2:
                        time.sleep(20 * (attempt + 1))  # back off, then try again
                        continue
                    raise FatalLLMError(f"HTTP {e.code} from Gemini (after retries): {detail}") from None
                raise RuntimeError(f"HTTP {e.code} from Gemini: {detail}") from None
        # Join every non-"thought" text part: newer models may split the answer or add thoughts.
        try:
            cand = payload["candidates"][0]
            text = "".join(p.get("text", "") for p in cand["content"]["parts"] if not p.get("thought"))
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"Unexpected Gemini response: {str(payload)[:300]}") from None
        if not text.strip():
            raise RuntimeError(f"Empty Gemini answer (finishReason={cand.get('finishReason')})")
        if cand.get("finishReason") == "MAX_TOKENS":
            text += "\n/*TRUNCATED_BY_MAX_TOKENS*/"
        return text


class OllamaLLM(LLMClient):
    def __init__(self, model: str | None = None, host: str | None = None, timeout: int = 180):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout
        self.name = f"ollama:{self.model}"

    def generate_json(self, system: str, user: str) -> str:
        body = {
            "model": self.model, "stream": False, "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"]


# --------------------------------------------------------------------------
# Offline mock: rule-based "extractor" that imitates the LLM's JSON contract
# --------------------------------------------------------------------------
SECTION_ALIASES = {
    "summary": {"summary", "profile", "about", "about me", "ringkasan", "profil"},
    "experience": {"experience", "work experience", "pengalaman", "pengalaman kerja"},
    "education": {"education", "pendidikan"},
    "skills": {"skills", "skill", "technical skills", "keahlian", "keterampilan"},
    "certifications": {"certifications", "certification", "sertifikasi", "sertifikat"},
    "languages": {"languages", "bahasa"},
}
_HEADER_LOOKUP = {a: k for k, v in SECTION_ALIASES.items() for a in v}

MONTHS = {
    "jan": 1, "january": 1, "januari": 1, "feb": 2, "february": 2, "februari": 2,
    "mar": 3, "march": 3, "maret": 3, "apr": 4, "april": 4, "may": 5, "mei": 5,
    "jun": 6, "june": 6, "juni": 6, "jul": 7, "july": 7, "juli": 7,
    "aug": 8, "august": 8, "agu": 8, "agt": 8, "agustus": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "okt": 10, "october": 10, "oktober": 10, "nov": 11, "november": 11,
    "dec": 12, "des": 12, "december": 12, "desember": 12,
}
_DATE_TOKEN = re.compile(
    r"(?:(?P<mon>[A-Za-z]+)\.?\s+)?(?P<year>(?:19|20)\d{2})|(?P<now>present|sekarang|saat ini|now)",
    re.IGNORECASE,
)
_DEGREE_RE = re.compile(
    r"^(S1|S2|S3|D3|D4|SMK|SMA|Bachelor|Master|B\.?Sc\.?|M\.?Sc\.?|PhD)\b[\s\-:]*(?:of|in)?\s*(.*)$",
    re.IGNORECASE,
)
_KNOWN_SKILLS = [
    (r"\bpython\b", "Python"), (r"\bsql\b", "SQL"), (r"\bgit\b", "Git"),
    (r"\bmachine learning\b", "Machine Learning"), (r"\bnlp\b", "NLP"),
    (r"\bpandas\b", "Pandas"), (r"\bfastapi\b", "FastAPI"), (r"\bdocker\b", "Docker"),
    (r"\bprompt engineering\b", "Prompt Engineering"), (r"\blangchain\b", "LangChain"),
    (r"\bllm\b", "LLM"), (r"\bscikit-learn\b", "Scikit-learn"), (r"\btableau\b", "Tableau"),
    (r"\bexcel\b", "Excel"), (r"\btensorflow\b", "TensorFlow"), (r"\bpytorch\b", "PyTorch"),
    (r"\brest apis?\b", "REST API"),
]


def _between(s: str, a: str, b: str) -> str:
    i, j = s.find(a), s.rfind(b)
    return s[i + len(a): j] if i != -1 and j > i else s


def _ym(mon: str | None, year: str, end: bool) -> tuple[str, bool]:
    if mon and mon.lower() in MONTHS:
        return f"{year}-{MONTHS[mon.lower()]:02d}", False
    return f"{year}-{'12' if end else '01'}", True  # year-only -> imprecise


def _parse_range(text: str):
    toks = list(_DATE_TOKEN.finditer(text))
    if not toks:
        return None
    imprecise = False
    first = toks[0]
    if first.group("now"):
        return None
    start, imp = _ym(first.group("mon"), first.group("year"), end=False)
    imprecise |= imp
    end = None
    if len(toks) > 1:
        t = toks[1]
        if t.group("now"):
            end = "present"
        else:
            end, imp = _ym(t.group("mon"), t.group("year"), end=True)
            imprecise |= imp
    return start, end, imprecise


def _rule_based_extract(cv: str) -> dict:
    lines = [ln.rstrip() for ln in cv.splitlines()]
    nonempty = [ln for ln in lines if ln.strip()]
    notes: list[str] = []
    name = location = summary = None

    def is_header(s: str) -> str | None:
        return _HEADER_LOOKUP.get(s.strip().rstrip(":").lower())

    if nonempty:
        first = nonempty[0].strip()
        if "|" not in first and len(first) <= 60 and not is_header(first):
            name = first.title()
        if len(nonempty) > 1:
            second = nonempty[1].strip()
            if "|" in second:
                location = second.split("|")[0].strip() or None
            elif len(second) <= 40 and not is_header(second):
                location = second

    sections: dict[str, list[str]] = {k: [] for k in SECTION_ALIASES}
    cur = None
    for ln in lines:
        h = is_header(ln) if ln.strip() else None
        if h:
            cur = h
        elif cur:
            sections[cur].append(ln)
    if not any(sections.values()):
        notes.append("No recognizable section headers found; extraction may be incomplete.")

    if sections["summary"]:
        summary = next((l.strip() for l in sections["summary"] if l.strip()), None)

    # experience
    experience: list[dict] = []
    year_only = False
    for ln in sections["experience"]:
        s = ln.strip()
        if not s:
            continue
        if s[0] in "-•*":
            if experience:
                experience[-1]["highlights"].append(s.lstrip("-•* ").strip())
            continue
        parts = [p.strip() for p in s.split("|")]
        if len(parts) < 2:
            continue
        start = end = None
        if len(parts) >= 3:
            rng = _parse_range(parts[-1])
            if rng:
                start, end, imp = rng
                year_only |= imp
        experience.append({
            "title": parts[0] or None, "company": parts[1] or None,
            "start": start, "end": end,
            "is_internship": bool(re.search(r"intern|magang", s, re.I)),
            "highlights": [], "evidence": s,
        })
    if year_only:
        notes.append("Some experience dates had year precision only; months were assumed.")

    # education
    education: list[dict] = []
    for ln in sections["education"]:
        s = ln.strip()
        if not s or s[0] in "-•*":
            continue
        parts = [p.strip() for p in s.split("|")]
        degree, field_ = None, parts[0]
        m = _DEGREE_RE.match(parts[0])
        if m:
            degree, field_ = m.group(1).upper(), (m.group(2).strip() or None)
        years = re.findall(r"(?:19|20)\d{2}", parts[-1]) if len(parts) > 1 else []
        education.append({
            "institution": parts[1] if len(parts) > 1 else None,
            "degree": degree, "field": field_,
            "end_year": int(years[-1]) if years else None, "evidence": s,
        })

    # skills: explicit list first, then usage found inside experience bullets
    skills: list[dict] = []
    seen: set[str] = set()
    for ln in sections["skills"]:
        s = ln.strip().lstrip("-•* ")
        if not s:
            continue
        body = s.split(":", 1)[1] if ":" in s else s
        for tok in re.split(r"[,;|•]", body):
            tok = tok.strip()
            if tok and tok.lower() not in seen:
                seen.add(tok.lower())
                skills.append({"name": tok, "evidence": ln.strip()})
    for e in experience:
        for h in e["highlights"]:
            for pat, canon in _KNOWN_SKILLS:
                if re.search(pat, h, re.I) and canon.lower() not in seen:
                    seen.add(canon.lower())
                    skills.append({"name": canon, "evidence": h})

    certs = []
    for ln in sections["certifications"]:
        s = ln.strip().lstrip("-•* ").strip()
        if s:
            certs.append({"name": s, "evidence": ln.strip()})

    languages = []
    for ln in sections["languages"]:
        for tok in re.split(r"[,;]", ln.strip().lstrip("-•* ")):
            if tok.strip():
                languages.append(tok.strip())

    return {
        "full_name": name, "location": location, "summary": summary,
        "education": education, "experience": experience, "skills": skills,
        "certifications": certs, "languages": languages, "extraction_notes": notes,
    }


class MockLLM(LLMClient):
    name = "mock-rule-based"

    def __init__(self):
        self.calls: list[dict] = []  # every prompt "sent", so tests can inspect it

    def generate_json(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        cv = _between(user, "<cv>", "</cv>")
        return json.dumps(_rule_based_extract(cv), ensure_ascii=False)


def get_llm(provider: str) -> LLMClient:
    provider = (provider or "mock").lower()
    if provider == "mock":
        return MockLLM()
    if provider == "gemini":
        return GeminiLLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"unknown provider: {provider}")
