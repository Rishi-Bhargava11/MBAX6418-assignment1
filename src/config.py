"""Project configuration.

Reads endpoint/model settings from environment variables, falling back to a
local `.env` file (gitignored) and finally to the course defaults.
Everything in this project targets the MBAX 6418 OpenAI-compatible endpoint.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
DATA_FILE = DATA_DIR / "Gift_Cards.jsonl.gz"
NRC_FILE = DATA_DIR / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

# --- tiny .env loader (no third-party dependency) ---------------------------
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(BASE_DIR / ".env")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://dobolyi.com:9000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "DeepSeek-V4-Flash-0731")

# Fixed random seed so any sampling is reproducible run-to-run.
RANDOM_SEED = 6418
