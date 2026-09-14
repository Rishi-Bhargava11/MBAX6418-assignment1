"""Reusable classification prompt(s) (Assignment Steps 1 & 6).

Contrary to Step 1's binary classifier, Step 6 redefines sentiment to three
classes. This module makes the prompt class-set aware so the SAME machinery
serves both: pass the allowed sentiment labels and the prompt adapts.

The prompt takes a review's title + text and asks the LLM for the sentiment
and a primary emotion. It never relies on the star rating, and it requests a
single machine-readable JSON object so answers read back programmatically.
"""
from __future__ import annotations

from . import config

EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust",
]

POSITIVE = "POSITIVE"
NEUTRAL = "NEUTRAL"
NEGATIVE = "NEGATIVE"
BINARY = [POSITIVE, NEGATIVE]
THREE = [POSITIVE, NEUTRAL, NEGATIVE]

_SYSTEM_TEMPLATE = """\
You are a highly reliable sentiment and emotion classifier for Amazon customer
reviews.

You will be given a review split into a short title and a body (text). Decide:
1. The overall sentiment — one of: {labels}.
2. The PRIMARY emotion the review expresses, chosen from exactly these eight:
   anger, anticipation, disgust, fear, joy, sadness, surprise, trust.

Strict rules:
- Sentiment must be exactly one of: {labels}. Base it ONLY on the title and
  body text. Never rely on, guess, or infer a star rating.
- The primary emotion must be exactly one of the eight listed above (lowercase).
  Pick the single strongest emotion the review conveys.
- If the title and body conflict, weigh the body text as more informative.
- Judge terse, short, or sarcastic reviews by their overall tone.
- Reply with exactly one JSON object in this shape (no other text, keys, or
  explanation):

  {{"sentiment": "{first}", "primary_emotion": "joy"}}

  where "{first}" is replaced by whatever sentiment applies.
"""

USER_PROMPT_TEMPLATE = """\
Review title:
{title}

Review body:
{text}

What is the sentiment and primary emotion? Respond with the JSON object only.
"""


def build_system_prompt(classes: list[str]) -> str:
    labels = ", ".join(classes)
    return _SYSTEM_TEMPLATE.format(labels=labels, first=classes[0])


def build_user_prompt(title: str, text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        title=(title or "").strip(),
        text=(text or "").strip(),
    )


def build_messages(title: str, text: str,
                   classes: list[str] | None = None) -> list[dict]:
    """Message list for a chat-completions call. Defaults to the three-class
    split, matching the final assignment definition (Step 6)."""
    if classes is None:
        classes = THREE
    return [
        {"role": "system", "content": build_system_prompt(classes)},
        {"role": "user", "content": build_user_prompt(title, text)},
    ]
