"""NRC Emotion Lexicon (word-level) loader.

Reads the public NRC-Emotion-Lexicon-Wordlevel-v0.92.txt (TSV: word, emotion,
association 0/1) and exposes a lookup of word -> set(emotions). The lexicon
links words to eight emotions (anger, anticipation, disgust, fear, joy,
sadness, surprise, trust) plus two sentiments (positive, negative); only the
eight emotions are used for the emotion detectors.
"""
from __future__ import annotations

from collections import defaultdict

from .config import NRC_FILE

EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust",
]

_word_emotions: dict[str, set[str]] | None = None


def load() -> dict[str, set[str]]:
    """Return {lemma: set(emotion)} for every word with an association=1."""
    global _word_emotions
    if _word_emotions is not None:
        return _word_emotions
    table: dict[str, set[str]] = defaultdict(set)
    with open(NRC_FILE, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, emotion, assoc = parts
            if emotion not in EMOTIONS:
                continue
            if assoc.strip() == "1":
                table[word.strip().lower()].add(emotion)
    _word_emotions = dict(table)
    return _word_emotions


def score_text(text: str) -> dict[str, int]:
    """Emotion counts for a piece of text using the NRC lexicon.

    Tokenizes crudely, lowercases, strips punctuation, and tallies any token
    found in the lexicon. Returns {emotion: count}.
    """
    import re

    table = load()
    counts = {e: 0 for e in EMOTIONS}
    tokens = re.findall(r"[A-Za-z']+", (text or "").lower())
    for tok in tokens:
        for emo in table.get(tok, ()):
            counts[emo] += 1
    return counts


def dominant(counts: dict[str, int]) -> str | None:
    """The emotion with the highest count; None if there's a tie or no hits."""
    hits = {e: c for e, c in counts.items() if c > 0}
    if not hits:
        return None
    top = max(hits.values())
    winners = [e for e, c in hits.items() if c == top]
    return winners[0] if len(winners) == 1 else None
