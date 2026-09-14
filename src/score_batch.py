"""Scoring runners for the Step 1 binary classifier (Steps 2, 5, 6).

Produces, for every review it scores:
  rating            star rating (ground truth source; never shown to the model)
  truth             class derived from the rating ("correct answer")
  predicted         the model's sentiment label
  llm_emotion       the model's primary emotion (Step 5)
  nrc_emotion       primary emotion from the NRC word list (Step 5)
  title, text, raw  the review and the model's verbatim output

Two run modes:
  first   -- read the first N reviews in file order (Step 2 binary)
  balanced-- draw a roughly-equal number from each class from the whole file
             using a fixed random seed, so the same set comes up every time
             (Step 6, three classes)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from . import config
from .classify import classify, get_client
from .data import iter_reviews
from .nrc import dominant as nrc_dominant, score_text as nrc_score
from .prompt import BINARY, THREE

POSITIVE = "POSITIVE"
NEUTRAL = "NEUTRAL"
NEGATIVE = "NEGATIVE"


def truth_binary(rating: float) -> str:
    return POSITIVE if rating >= 4 else NEGATIVE


def truth_three(rating: float) -> str:
    if rating >= 4:
        return POSITIVE
    if rating >= 3:
        return NEUTRAL
    return NEGATIVE


def _call_with_retry(client, title: str, text: str, retries: int = 3,
                     classes: list[str] | None = None) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            return classify(client, title, text, classes=classes)
        except Exception as err:
            last_err = err
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"classify failed after {retries} tries: {last_err}")


def run(mode: str, n: int = 100, per_class: int = 50) -> dict:
    client = get_client()
    truth_fn = truth_binary if mode == "first" else truth_three
    classes = BINARY if mode == "first" else THREE

    if mode == "balanced":
        # Read the whole file once, in file order.
        all_rows = list(iter_reviews())
        rng = random.Random(config.RANDOM_SEED)
        buckets = {POSITIVE: [], NEUTRAL: [], NEGATIVE: []}
        for idx, review in enumerate(all_rows):
            buckets[truth_three(float(review["rating"]))].append(idx)
        picks: list[int] = []
        actual = {}
        for cls in (POSITIVE, NEUTRAL, NEGATIVE):
            pool = buckets[cls]
            chosen = rng.sample(pool, min(per_class, len(pool))) if pool else []
            picks.extend(chosen)
            actual[cls] = len(chosen)
        picks.sort()  # keep file order
        rows = [all_rows[i] for i in picks]
    else:
        rows = list(iter_reviews(n))

    out_rows = []
    for review in rows:
        rating = float(review["rating"])
        truth = truth_fn(rating)
        title = review.get("title", "")
        text = review.get("text", "")
        result = _call_with_retry(client, title, text, classes=classes)
        nrc_counts = nrc_score(f"{title} {text}")
        nrc_emo = nrc_dominant(nrc_counts)
        out_rows.append({
            "rating": rating,
            "truth": truth,
            "predicted": result["sentiment"],
            "llm_emotion": result["primary_emotion"],
            "nrc_emotion": nrc_emo,
            "nrc_counts": nrc_counts,
            "title": title,
            "text": text,
            "raw": result["raw"],
        })
    return {"mode": mode, "rows": out_rows,
            "n_classes": 2 if mode == "first" else 3, "classes": classes}


def summarize(rows: list[dict], classes: list[str]) -> dict:
    total = len(rows)
    correct = sum(1 for r in rows if r["predicted"] == r["truth"])
    per_class = {}
    for cls in classes:
        sub = [r for r in rows if r["truth"] == cls]
        ok = sum(1 for r in sub if r["predicted"] == cls)
        per_class[cls] = {
            "n": len(sub),
            "correct": ok,
            "accuracy": round(ok / len(sub), 4) if sub else 0.0,
        }
    wrong = [i for i, r in enumerate(rows) if r["predicted"] != r["truth"]]
    return {
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "per_class": per_class,
        "wrong_indices": wrong,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.score_batch")
    parser.add_argument("--mode", choices=["first", "balanced"], default="first")
    parser.add_argument("--n", type=int, default=100, help="rows (mode=first)")
    parser.add_argument("--per-class", type=int, default=50, help="per class (mode=balanced)")
    parser.add_argument("--out", type=str)
    parser.add_argument("--classes", nargs="*",
                        default=[POSITIVE, NEUTRAL, NEGATIVE])
    args = parser.parse_args(argv)

    run_result = run(args.mode, args.n, getattr(args, "per_class"))
    rows = run_result["rows"]
    classes = run_result["classes"]
    summary = summarize(rows, classes)

    out = Path(args.out or (config.RESULTS_DIR /
                            (f"balanced_{args.per_class}_per_class.json" if args.mode == "balanced"
                             else f"first_{args.n}.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"mode": args.mode, "n": len(rows),
                        "classes": classes,
                        "seed": config.RANDOM_SEED,
                        "model": config.OPENAI_MODEL},
               "summary": summary, "rows": rows}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} scored rows -> {out}")

    s = summary
    print("\n========== SCORE SUMMARY ==========")
    print(f"Mode             : {args.mode}")
    print(f"Rows scored      : {s['n']}")
    print(f"Overall accuracy : {s['accuracy']:.2%}  ({s['correct']}/{s['n']})")
    for cls in classes:
        c = s["per_class"].get(cls, {"n": 0, "correct": 0, "accuracy": 0.0})
        print(f"  {cls:<8} : {c['correct']}/{c['n']} correct ({c['accuracy']:.1%})")
    print(f"Wrong indices    : {s['wrong_indices']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
