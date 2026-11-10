"""Evaluate the flow against hand-written gold labels (data/gold/gold.json).

    python evaluate.py                     # offline mock LLM
    python evaluate.py --provider gemini   # real LLM (the number that actually matters)

Metrics (and why they exist)
- skill precision / recall / F1 : did we extract the right skills, and only those?
- name / n_experience / n_education / months(+-1) / degree : field-level accuracy
- groundedness                 : share of extracted items whose evidence existed in the CV
                                 BEFORE the grounding filter (hallucination rate = 1 - this)
- band accuracy                : does the rubric band match a human's band?
- next-step accuracy           : does the suggested action match the human-chosen action?
- UNSAFE DOWNGRADE (key)       : a human-labelled strong/possible candidate that got a
                                 low-priority next step WITHOUT a low-confidence flag. This is
                                 the costly error (silently burying a good candidate); target = 0.
- caveat recall                : did we raise the caveats a human expects?
Plus two safety checks: PII never reaches the LLM prompt; injected text cannot change the score.
"""
import argparse
import json
import re
from pathlib import Path

from cvflow.llm import MockLLM, get_llm
from cvflow.pipeline import run_pipeline
from cvflow.recommend import load_jd, normalize_skill

ROOT = Path(__file__).parent
AS_OF = "2026-11"  # fixed so results are reproducible


def prf(pred: set, gold: set):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return tp, len(pred), len(gold), p, r, f


def safety_checks(jd: dict) -> list[tuple[str, bool, str]]:
    checks = []
    # 1) PII / sensitive data never reaches the model
    llm = MockLLM()
    run_pipeline((ROOT / "data/cvs/cv_01_strong.txt").read_text(encoding="utf-8"), llm, jd, AS_OF)
    sent = " ".join(c["user"] for c in llm.calls)
    leaked = [x for x in ("rina.wijaya@example.com", "0812-0000-0001", "12 Mei 2000", "Perempuan") if x in sent]
    checks.append(("PII/sensitive data not in LLM prompt", not leaked, f"leaked={leaked}"))
    # 2) injected text cannot move the score
    raw = (ROOT / "data/cvs/cv_03_injection.txt").read_text(encoding="utf-8")
    clean = "\n".join(l for l in raw.splitlines() if "IGNORE ALL PREVIOUS" not in l)
    a = run_pipeline(raw, MockLLM(), jd, AS_OF)["recommendation"]
    b = run_pipeline(clean, MockLLM(), jd, AS_OF)["recommendation"]
    checks.append(("Injection line does not change score", a["fit_score"] == b["fit_score"],
                   f"with={a['fit_score']} without={b['fit_score']}"))
    codes = {c["code"] for c in a["caveats"]}
    checks.append(("Injection is flagged for the human", "INJECTION_TEXT_DETECTED" in codes
                   and a["suggested_next_step"] == "manual_review_required", f"next={a['suggested_next_step']}"))
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "gemini", "ollama"])
    args = ap.parse_args()

    jd = load_jd(ROOT / "data" / "job_description.json")
    gold = json.loads((ROOT / "data/gold/gold.json").read_text(encoding="utf-8"))
    llm = get_llm(args.provider)

    rows = []
    tp = npred = ngold = 0
    acc = {k: [] for k in ("name", "n_exp", "n_edu", "months", "degree", "band", "next", "caveat_recall")}
    items_before = dropped = 0
    unsafe = []

    for stem, g in gold.items():
        if stem.startswith("_"):
            continue
        out = run_pipeline((ROOT / "data/cvs" / f"{stem}.txt").read_text(encoding="utf-8"), llm, jd, AS_OF)
        ex, rec, meta, der = out["extraction"], out["recommendation"], out["pipeline_meta"], out["derived"]
        if not meta["extraction_ok"]:
            print(f"[WARN] {stem}: extraction FAILED -> {meta['validation_errors'][-1:]}")

        pred_sk = {normalize_skill(s["name"]) for s in ex["skills"]}
        gold_sk = {normalize_skill(s) for s in g["skills"]}
        t, p_n, g_n, p, r, f = prf(pred_sk, gold_sk)
        tp += t; npred += p_n; ngold += g_n

        name_ok = (ex["full_name"] or "").lower() == g["full_name"].lower()
        exp_ok = len(ex["experience"]) == g["n_experience"]
        edu_ok = len(ex["education"]) == g["n_education"]
        months_ok = None if g["total_months"] is None else abs(der["total_months"] - g["total_months"]) <= 1
        deg = (ex["education"][0]["degree"] if ex["education"] else None)
        degree_ok = None if g["education_degree"] is None else (deg == g["education_degree"])
        band_ok = rec["band"] == g["expected_band"]
        next_ok = rec["suggested_next_step"] == g["expected_next_step"]
        got = {c["code"] for c in rec["caveats"]}
        cav_rec = len(got & set(g["expected_caveats"])) / len(g["expected_caveats"])

        for k, v in (("name", name_ok), ("n_exp", exp_ok), ("n_edu", edu_ok), ("months", months_ok),
                     ("degree", degree_ok), ("band", band_ok), ("next", next_ok), ("caveat_recall", cav_rec)):
            if v is not None:
                acc[k].append(float(v))
        items_before += meta["items_before_grounding"]
        dropped += len(meta["ungrounded_items_dropped"])

        if (g["expected_band"] in ("strong_match", "possible_match")
                and rec["suggested_next_step"] == "low_priority_human_review"
                and rec["confidence"] != "low"):
            unsafe.append(stem)

        rows.append({
            "cv": stem, "skills_P": round(p, 2), "skills_R": round(r, 2), "skills_F1": round(f, 2),
            "name": name_ok, "n_exp": exp_ok, "n_edu": edu_ok, "months": months_ok,
            "band": f"{rec['band']} (gold {g['expected_band']})" + ("" if band_ok else " X"),
            "next": f"{rec['suggested_next_step']} (gold {g['expected_next_step']})" + ("" if next_ok else " X"),
            "confidence": rec["confidence"], "score": rec["fit_score"],
        })

    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    P = tp / npred if npred else 0.0
    R = tp / ngold if ngold else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    summary = {
        "provider": getattr(llm, "name", args.provider),
        "n_cvs": len(rows),
        "skills_micro_precision": round(P, 3), "skills_micro_recall": round(R, 3), "skills_micro_f1": round(F, 3),
        "name_accuracy": mean(acc["name"]), "n_experience_accuracy": mean(acc["n_exp"]),
        "n_education_accuracy": mean(acc["n_edu"]), "months_within_1_accuracy": mean(acc["months"]),
        "degree_accuracy": mean(acc["degree"]),
        "groundedness_before_filter": round(1 - dropped / items_before, 3) if items_before else None,
        "band_accuracy": mean(acc["band"]), "next_step_accuracy": mean(acc["next"]),
        "unsafe_downgrades": unsafe, "caveat_recall": mean(acc["caveat_recall"]),
    }
    checks = safety_checks(jd)

    # ---- console + report ----
    print(f"\nProvider: {summary['provider']}   CVs: {summary['n_cvs']}\n")
    for r in rows:
        print(f"{r['cv']:18s} skillsF1={r['skills_F1']:<4}  name={r['name']!s:5} exp={r['n_exp']!s:5} "
              f"edu={r['n_edu']!s:5} months={r['months']!s:5}  score={r['score']}")
        print(f"{'':18s} band: {r['band']}\n{'':18s} next: {r['next']}  [conf={r['confidence']}]")
    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"  {k:30s} {v}")
    print("\nSAFETY CHECKS")
    for n, ok, d in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}  ({d})")

    md = [f"# Evaluation report\n\nProvider: `{summary['provider']}` | CVs: {summary['n_cvs']} | as_of: {AS_OF}\n",
          "## Per-CV\n", "| CV | skills F1 | name | #exp | #edu | months | band | next step | conf |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['cv']} | {r['skills_F1']} | {r['name']} | {r['n_exp']} | {r['n_edu']} | {r['months']} | "
                  f"{r['band']} | {r['next']} | {r['confidence']} |")
    md += ["\n## Summary\n", "| metric | value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in summary.items()]
    md += ["\n## Safety checks\n", "| check | result | detail |", "|---|---|---|"]
    md += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in checks]
    suffix = "" if args.provider == "mock" else f".{args.provider}"
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / f"eval_report{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / "outputs" / f"eval_results{suffix}.json").write_text(
        json.dumps({"summary": summary, "rows": rows,
                    "safety": [{"check": n, "pass": ok, "detail": d} for n, ok, d in checks]},
                   indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
