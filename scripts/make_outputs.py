"""Reproduce the derived deliverables from a scored-results JSON.

Usage:
    python scripts/make_outputs.py [--input results/balanced_50_per_class.json]

Regenerates from one saved scored run:
  results/dashboard.html            final dashboard
  results/emotion_analysis.json     LLM vs NRC emotion comparison (Step 5)

Scoring itself (which calls the model endpoint) is a separate step:
    python -m src.score_batch --mode balanced --per-class 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src` importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.analyze_emotions import analyze  # noqa: E402
from src.dashboard import build_html  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=str(config.RESULTS_DIR / "balanced_50_per_class.json"))
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    print(f"Loaded {len(rows)} rows from {in_path}")

    # Emotion comparison -> results/emotion_analysis.json
    emo = analyze(rows)
    emo_out = config.RESULTS_DIR / "emotion_analysis.json"
    emo_out.write_text(json.dumps(emo, indent=2), encoding="utf-8")
    print(f"Emotion analysis -> {emo_out}")

    # Dashboard -> results/dashboard.html
    html = build_html(payload)
    dash_out = config.RESULTS_DIR / "dashboard.html"
    dash_out.write_text(html, encoding="utf-8")
    print(f"Dashboard -> {dash_out} ({len(html):,} bytes)")

    print(f"\nAgreement (LLM vs NRC emotion): "
          f"{emo['n_agree']}/{emo['n_with_both']} ({emo['agree_rate']:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
