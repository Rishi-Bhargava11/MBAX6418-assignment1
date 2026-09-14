"""Binary sentiment classifier backed by the MBAX 6418 OpenAI-compatible
endpoint (Assignment Step 1).

classify() takes a title + text, sends it to the LLM through the prompt in
prompt.py, and returns a strict POSITIVE / NEGATIVE label plus the verbatim
model output, so callers never have to guess what label the model picked.
"""
from __future__ import annotations

import json
import re

from openai import OpenAI

from . import config, prompt
from .prompt import build_messages

POSITIVE = "POSITIVE"
NEUTRAL = "NEUTRAL"
NEGATIVE = "NEGATIVE"
VALID = {POSITIVE, NEUTRAL, NEGATIVE}


def get_client() -> OpenAI:
    return OpenAI(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY)


def parse_response(raw: str, classes: list[str] | None = None) -> dict:
    """Extract {'sentiment': ..., 'primary_emotion': ...} from raw model output.

    Tries the embedded JSON object first, then falls back to a plain word scan
    for the sentiment. Values are normalized; a sentiment outside `classes`
    (or unknown/missing emotion) becomes None.
    """
    if classes is None:
        classes = prompt.THREE
    allowed = set(classes)
    text = (raw or "").strip()
    emotion = None
    # 1) JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    data = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            data = {}
    label = str(data.get("sentiment", "")).strip().upper()
    if label not in allowed:
        label = _scan_sentiment(text, allowed)
    emo = str(data.get("primary_emotion", "")).strip().lower()
    if emo in prompt.EMOTIONS:
        emotion = emo
    return {"sentiment": label, "primary_emotion": emotion}


def _scan_sentiment(text: str, allowed: set[str]) -> str | None:
    upper = text.upper()
    if upper.strip() in allowed:
        return upper.strip()
    # find any allowed label mentioned, most specific first
    found = [l for l in allowed if l in upper]
    if len(found) == 1:
        return found[0]
    return None


def classify(
    client: OpenAI,
    title: str,
    text: str,
    temperature: float = 0.0,
    classes: list[str] | None = None,
) -> dict:
    """Classify one review. Returns {'sentiment', 'primary_emotion', 'raw'}."""
    if classes is None:
        classes = prompt.THREE
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=build_messages(title, text, classes),
        temperature=temperature,
    )
    raw = resp.choices[0].message.content or ""
    parsed = parse_response(raw, classes)
    return {"sentiment": parsed["sentiment"],
            "primary_emotion": parsed["primary_emotion"],
            "raw": raw}
