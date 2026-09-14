"""CLI entry point for binary sentiment classification (Assignment Step 1).

Usage:
    python -m src.cli spotcheck
        Classify a few hand-written example reviews to sanity-check the prompt.

    python -m src.cli review --title "..." --text "..."
        Classify a single review given on the command line.

Everything here runs against the MBAX 6418 OpenAI-compatible endpoint and
never sees or uses the star rating.
"""
from __future__ import annotations

import argparse
import json
import sys

from .classify import classify, get_client
from .prompt import build_user_prompt

# Hand-written examples used to spot-check the prompt (Step 1). Each is a
# (title, text) pair; expected is OUR expectation of the true sentiment.
SPOTCHECKS = [
    ("Absolutely loved it", "Great gift card, my friend was thrilled. Arrived instantly.", "POSITIVE"),
    ("Terrible experience", "The card never activated and support was useless. Avoid.", "NEGATIVE"),
    ("cashed in", "worked exactly as described, no complaints", "POSITIVE"),
    ("Waste of money", "do not buy", "NEGATIVE"),
    ("Decent", "It was okay, arrived on time but nothing special.", None),  # ambiguous/tence case
]


def _print_classification(title: str, text: str, client, show: str) -> None:
    result = classify(client, title, text)
    print("=" * 70)
    print(f"TITLE : {title}")
    print(f"TEXT  : {text[:200]}")
    print(f"LABEL : {result['sentiment']}")
    print(f"EMO   : {result['primary_emotion']}")
    if show == "raw":
        print(f"RAW   : {result['raw']}")
    print()


def cmd_spotcheck(args) -> int:
    client = get_client()
    ok = 0
    for title, text, expected in SPOTCHECKS:
        result = classify(client, title, text)
        label = result["sentiment"]
        print(f"{str(label):>8}  | expected={str(expected):>8}  | {title}")
        if expected is None:
            continue
        if label == expected:
            ok += 1
    total = sum(1 for _, _, e in SPOTCHECKS if e is not None)
    print(f"\nSpot-check: {ok}/{total} matched expectation.")
    return 0 if ok == total else 1


def cmd_review(args) -> int:
    client = get_client()
    _print_classification(args.title, args.text, client, args.show)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sp = sub.add_parser("spotcheck", help="run hand-written example reviews")
    p_sp.set_defaults(func=cmd_spotcheck)

    p_rv = sub.add_parser("review", help="classify one review")
    p_rv.add_argument("--title", required=True)
    p_rv.add_argument("--text", required=True)
    p_rv.add_argument("--show", choices=["label", "raw"], default="label")
    p_rv.set_defaults(func=cmd_review)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
