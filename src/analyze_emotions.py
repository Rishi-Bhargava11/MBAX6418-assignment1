"""Step 5 — compare the LLM's primary emotion with the NRC word-list emotion.

Takes a scored-results JSON (rows carry llm_emotion and nrc_emotion) and
reports how often the two agree, their distributions, and where they diverge.
This is the two-independent-takes comparison the assignment asks for.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import config


def _coverage(rows):
    llm = sum(1 for r in rows if r.get("llm_emotion"))
    nrc = sum(1 for r in rows if r.get("nrc_emotion"))
    return {"llm": llm, "nrc": nrc, "n": len(rows)}


def _agreement_matrix(rows):
    """{llm: {nrc: count}} over rows where both emotions exist."""
    mat = defaultdict(Counter)
    for r in rows:
        llm, nrc = r.get("llm_emotion"), r.get("nrc_emotion")
        if llm and nrc:
            mat[llm][nrc] += 1
    return {k: dict(v) for k, v in mat.items()}


def analyze(rows: list[dict]) -> dict:
    cov = _coverage(rows)
    both = [r for r in rows if r.get("llm_emotion") and r.get("nrc_emotion")]
    agree = sum(1 for r in both if r["llm_emotion"] == r["nrc_emotion"])
    return {
        "coverage": cov,
        "n_with_both": len(both),
        "agreement": len(both),
        "n_agree": agree,
        "agree_rate": round(agree / len(both), 4) if both else 0.0,
        "llm_dist": dict(Counter(r.get("llm_emotion") for r in rows
                                 if r.get("llm_emotion"))),
        "nrc_dist": dict(Counter(r.get("nrc_emotion") for r in rows
                                 if r.get("nrc_emotion"))),
        "matrix": _agreement_matrix(rows),
        "divergent": [{"idx": i, "llm": r["llm_emotion"],
                       "nrc": r["nrc_emotion"], "title": r["title"],
                       "text": r["text"][:160], "rating": r["rating"]}
                      for i, r in enumerate(rows)
                      if r.get("llm_emotion") and r.get("nrc_emotion")
                      and r["llm_emotion"] != r["nrc_emotion"]][:8],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analyze_emotions")
    parser.add_argument("--input", type=str,
                        default=str(config.RESULTS_DIR / "balanced_50_per_class.json"))
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload["rows"]
    res = analyze(rows)

    print(f"=== EMOTION COMPARISON (LLM vs NRC) — {args.input} ===")
    cov = res["coverage"]
    print(f"Reviews            : {cov['n']}")
    print(f"LLM emotion given  : {cov['llm']}")
    print(f"NRC emotion given  : {cov['nrc']}")
    print(f"Both given         : {res['n_with_both']}")
    print(f"Agreement          : {res['n_agree']}/{res['n_with_both']} "
          f"({res['agree_rate']:.1%})")
    print("\nLLM emotion distribution:")
    for e, c in sorted(res["llm_dist"].items(), key=lambda x: -x[1]):
        print(f"  {e:<14} {c}")
    print("\nNRC emotion distribution:")
    for e, c in sorted(res["nrc_dist"].items(), key=lambda x: -x[1]):
        print(f"  {e:<14} {c}")

    if args.out:
        out = Path(args.out) or config.RESULTS_DIR / "emotion_analysis.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
