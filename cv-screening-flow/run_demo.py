"""Run the flow on one CV (or all sample CVs) and write JSON to outputs/.

    python run_demo.py                                  # all sample CVs, offline mock LLM
    python run_demo.py --cv data/cvs/cv_01_strong.txt   # one CV
    python run_demo.py --provider gemini                # real LLM (needs GEMINI_API_KEY)
    python run_demo.py --provider ollama                # real local LLM
"""
import argparse
import json
from pathlib import Path

from cvflow.llm import get_llm
from cvflow.pipeline import run_pipeline
from cvflow.recommend import load_jd

ROOT = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", help="path to a CV .txt file (default: all in data/cvs)")
    ap.add_argument("--provider", default="mock", choices=["mock", "gemini", "ollama"])
    ap.add_argument("--jd", default=str(ROOT / "data" / "job_description.json"))
    ap.add_argument("--as-of", default="2026-11", help="reference month for 'present' (YYYY-MM)")
    ap.add_argument("--quiet", action="store_true", help="do not print full JSON")
    args = ap.parse_args()

    jd = load_jd(args.jd)
    llm = get_llm(args.provider)
    paths = [Path(args.cv)] if args.cv else sorted((ROOT / "data" / "cvs").glob("*.txt"))
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    for p in paths:
        result = run_pipeline(p.read_text(encoding="utf-8"), llm, jd, as_of=args.as_of)
        suffix = "" if args.provider == "mock" else f".{args.provider}"
        (out_dir / f"{p.stem}{suffix}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        r = result["recommendation"]
        print(f"{p.name:24s} score={r['fit_score']:5.1f}  band={r['band']:14s} "
              f"next={r['suggested_next_step']:26s} confidence={r['confidence']}")
        if not args.quiet and args.cv:
            print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
