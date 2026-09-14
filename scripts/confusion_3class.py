"""One-off: 3-class confusion + where neutral collapses, from balanced results."""
import json
from collections import Counter

data = json.load(open("results/balanced_50_per_class.json", encoding="utf-8"))
rows = data["rows"]
classes = data["meta"]["classes"]

# where does each truth class get predicted?
print("=== Where each truth class is predicted (confusion) ===")
for t in classes:
    sub = [r for r in rows if r["truth"] == t]
    c = Counter(r["predicted"] for r in sub)
    print(f"{t:<8} n={len(sub)} -> " + ", ".join(f"{k}: {c.get(k,0)}" for k in classes))

# neutrals specifically
print("\n=== NEUTRAL misclassification breakdown ===")
neu = [r for r in rows if r["truth"] == "NEUTRAL"]
c = Counter(r["predicted"] for r in neu)
print(f"neutral {len(neu)} -> {dict(c)}")

# examples of neutral reviews and what the model said
print("\n=== Sample NEUTRAL reviews and predictions ===")
for r in neu[:6]:
    print(f"[{r['predicted']}] {r['rating']}★ | {r['title']!r} | {r['text'][:90]!r}")
