"""Reading the Amazon 2023 "Gift Cards" review data.

The file is gzipped JSON Lines: one review object per line. This module offers
reusable helpers to stream reviews without loading the whole (large) file into
memory at once.
"""
from __future__ import annotations

import gzip
import json
from typing import Iterator

from .config import DATA_FILE


def iter_reviews(num: int | None = None) -> Iterator[dict]:
    """Yield review dicts from the gzipped JSONL file, in file order.

    num: only yield the first `num` reviews if given, else stream everything.
    """
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if num is not None and i >= num:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def label_from_rating(rating: float) -> str:
    """Binary ground truth from the star rating (Step 2 rule). Never shown to
    the model — used only for scoring afterwards."""
    return "POSITIVE" if rating >= 4 else "NEGATIVE"
