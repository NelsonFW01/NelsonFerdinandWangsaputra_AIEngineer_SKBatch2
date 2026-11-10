"""Run with:  python -m unittest discover -s tests -v   (stdlib only, no network)"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvflow.derive import compute_experience  # noqa: E402
from cvflow.llm import MockLLM, LLMClient  # noqa: E402
from cvflow.pipeline import run_pipeline  # noqa: E402
from cvflow.preprocess import preprocess  # noqa: E402
from cvflow.recommend import load_jd  # noqa: E402
from cvflow.schema import validate_extraction, empty_extraction  # noqa: E402

JD = load_jd(ROOT / "data" / "job_description.json")
AS_OF = "2026-11"


def cv(name: str) -> str:
    return (ROOT / "data" / "cvs" / f"{name}.txt").read_text(encoding="utf-8")


class HallucinatingLLM(MockLLM):
    """Wraps the mock but injects a skill/job that is NOT in the CV."""
    def generate_json(self, system, user):
        d = json.loads(super().generate_json(system, user))
        d["skills"].append({"name": "Kubernetes", "evidence": "Led Kubernetes migration at Google"})
        d["experience"].append({"title": "CTO", "company": "Fake Corp", "start": "2015-01", "end": "2024-12",
                                "is_internship": False, "highlights": [], "evidence": "CTO | Fake Corp | 2015 - 2024"})
        return json.dumps(d)


class FlakyLLM(MockLLM):
    """First answer is broken (markdown chatter), second is valid."""
    def __init__(self):
        super().__init__()
        self.n = 0

    def generate_json(self, system, user):
        self.n += 1
        good = super().generate_json(system, user)
        return "Sure! here you go: not json" if self.n == 1 else good


class DeadLLM(LLMClient):
    name = "dead"

    def generate_json(self, system, user):
        raise TimeoutError("simulated outage")


class SchemaTests(unittest.TestCase):
    def test_empty_is_valid(self):
        self.assertEqual(validate_extraction(empty_extraction()), [])

    def test_catches_bad_types(self):
        d = empty_extraction()
        d["skills"] = [{"name": "Python"}]                        # missing evidence
        d["experience"] = [{"title": "x", "company": None, "start": "2024/01", "end": "now",
                            "is_internship": "no", "highlights": [], "evidence": "x"}]
        errs = validate_extraction(d)
        self.assertTrue(any("skills[0].evidence" in e for e in errs))
        self.assertTrue(any("start must be YYYY-MM" in e for e in errs))
        self.assertTrue(any("is_internship" in e for e in errs))


class PreprocessTests(unittest.TestCase):
    def test_pii_sensitive_injection(self):
        r = preprocess(cv("cv_01_strong"))
        self.assertNotIn("@", r.text_for_llm)
        self.assertNotIn("0812", r.text_for_llm)
        self.assertNotIn("Tanggal Lahir", r.text_for_llm)
        self.assertEqual(r.removed_sensitive_lines, 2)
        self.assertEqual(r.contact["emails"], ["rina.wijaya@example.com"])
        r3 = preprocess(cv("cv_03_injection"))
        self.assertEqual(len(r3.injection_flags), 1)
        self.assertNotIn("IGNORE ALL PREVIOUS", r3.text_for_llm)

    def test_year_ranges_not_treated_as_phone(self):
        r = preprocess("S1 Informatika | Univ | 2020 - 2024")
        self.assertIn("2020 - 2024", r.text_for_llm)

    def test_zero_width_removed(self):
        r = preprocess("Python​, SQL")
        self.assertEqual(r.zero_width_removed, 1)


class DeriveTests(unittest.TestCase):
    def test_overlap_counted_once_and_present(self):
        exp = [
            {"start": "2024-01", "end": "2024-12", "is_internship": False},
            {"start": "2024-06", "end": "2025-03", "is_internship": True},
            {"start": "2026-10", "end": "present", "is_internship": False},
        ]
        d = compute_experience(exp, "2026-11")
        self.assertEqual(d["total_months"], 15 + 2)  # Jan24-Mar25 = 15, Oct-Nov26 = 2
        self.assertEqual(d["gaps_over_6_months"], 1)

    def test_missing_dates_counted(self):
        d = compute_experience([{"start": None, "end": None, "is_internship": False}], AS_OF)
        self.assertEqual((d["total_months"], d["entries_missing_dates"]), (0, 1))


class PipelineTests(unittest.TestCase):
    def test_happy_path(self):
        out = run_pipeline(cv("cv_01_strong"), MockLLM(), JD, AS_OF)
        rec = out["recommendation"]
        self.assertEqual(out["derived"]["total_months"], 30)
        self.assertEqual(rec["band"], "strong_match")
        self.assertEqual(rec["suggested_next_step"], "prioritize_human_review")

    def test_always_human_in_the_loop(self):
        for n in ("cv_01_strong", "cv_02_fresh_grad", "cv_03_injection", "cv_04_sparse", "cv_05_freeform"):
            rec = run_pipeline(cv(n), MockLLM(), JD, AS_OF)["recommendation"]
            self.assertTrue(rec["requires_human_review"])
            self.assertIn("HUMAN_DECISION_REQUIRED", {c["code"] for c in rec["caveats"]})
            self.assertNotIn("reject", rec["suggested_next_step"])

    def test_hallucinations_are_dropped_and_reported(self):
        out = run_pipeline(cv("cv_01_strong"), HallucinatingLLM(), JD, AS_OF)
        names = {s["name"] for s in out["extraction"]["skills"]}
        self.assertNotIn("Kubernetes", names)
        self.assertNotIn("CTO", {e["title"] for e in out["extraction"]["experience"]})
        self.assertEqual(len(out["pipeline_meta"]["ungrounded_items_dropped"]), 2)
        self.assertIn("UNGROUNDED_ITEMS_DROPPED", {c["code"] for c in out["recommendation"]["caveats"]})
        self.assertEqual(out["derived"]["total_months"], 30)  # fake CTO job did not inflate experience

    def test_repair_retry(self):
        out = run_pipeline(cv("cv_01_strong"), FlakyLLM(), JD, AS_OF)
        self.assertTrue(out["pipeline_meta"]["extraction_ok"])
        self.assertEqual(out["pipeline_meta"]["attempts"], 2)
        self.assertIn("EXTRACTION_REPAIR_USED", {c["code"] for c in out["recommendation"]["caveats"]})

    def test_total_failure_fails_safe(self):
        out = run_pipeline(cv("cv_01_strong"), DeadLLM(), JD, AS_OF)
        rec = out["recommendation"]
        self.assertFalse(out["pipeline_meta"]["extraction_ok"])
        self.assertEqual(rec["suggested_next_step"], "request_more_information")
        self.assertEqual(rec["confidence"], "low")

    def test_pii_never_sent_to_llm(self):
        llm = MockLLM()
        run_pipeline(cv("cv_01_strong"), llm, JD, AS_OF)
        sent = " ".join(c["user"] + c["system"] for c in llm.calls)
        for secret in ("rina.wijaya@example.com", "0812-0000-0001", "12 Mei 2000", "Perempuan"):
            self.assertNotIn(secret, sent)

    def test_injection_cannot_change_score(self):
        raw = cv("cv_03_injection")
        clean = "\n".join(l for l in raw.splitlines() if "IGNORE ALL PREVIOUS" not in l)
        a = run_pipeline(raw, MockLLM(), JD, AS_OF)["recommendation"]
        b = run_pipeline(clean, MockLLM(), JD, AS_OF)["recommendation"]
        self.assertEqual(a["fit_score"], b["fit_score"])
        self.assertLess(a["fit_score"], 75)
        self.assertEqual(a["suggested_next_step"], "manual_review_required")

    def test_low_coverage_blocks_silent_downgrade(self):
        rec = run_pipeline(cv("cv_05_freeform"), MockLLM(), JD, AS_OF)["recommendation"]
        self.assertIn("EXTRACTION_LOW_COVERAGE", {c["code"] for c in rec["caveats"]})
        self.assertEqual(rec["suggested_next_step"], "manual_review_required")


if __name__ == "__main__":
    unittest.main()
