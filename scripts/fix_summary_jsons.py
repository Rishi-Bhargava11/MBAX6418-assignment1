"""Recompute + persist the corrected summary inside a scored-results JSON so
dashboard figures read from stored data match the true per-class numbers."""
import json
import sys

sys.path.insert(0, ".")
from src.score_batch import summarize

path = "results/step2_batch100.json"
data = json.load(open(path, encoding="utf-8"))
data["summary"] = summarize(data["rows"])  # recomputed from rows (fixed formula)
json.dump(data, open(path, "w", encoding="utf-8"), indent=2)
print("Updated summary:", data["summary"])
