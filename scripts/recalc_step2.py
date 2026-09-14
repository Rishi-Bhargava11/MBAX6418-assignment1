"""One-off: recompute Step 2 summary + show wrong rows from the saved JSON."""
import json
import sys
from collections import Counter

sys.path.insert(0, ".")
from src.score_batch import summarize

data = json.load(open("results/step2_batch100.json", encoding="utf-8"))
s = summarize(data["rows"])
print("=== FIXED SUMMARY ===")
print(f"Rows           : {s['n']}")
print(f"Correct        : {s['correct']}")
print(f"Overall acc    : {s['accuracy']:.4f}")
for cls in ("positive", "negative"):
    c = s[cls]
    print(f"{cls:>8} : {c['correct']}/{c['n']} correct ({c['accuracy']:.4f})")
print(f"Wrong indices  : {s['wrong_indices']}")
print("Truth balance:", dict(Counter(r['truth'] for r in data['rows'])))
print("Predicted bal :", dict(Counter(r['predicted'] for r in data['rows'])))
print("\n=== WRONG ROWS ===")
for i in s['wrong_indices']:
    r = data['rows'][i]
    print(f"[{i}] rating={r['rating']} truth={r['truth']} pred={r['predicted']}")
    print(f"    title: {r['title']!r}")
    print(f"    text : {r['text'][:180]!r}")
