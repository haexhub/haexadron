import os
from pathlib import Path

# Load .env file if present
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

BITGN_HOST = os.getenv("BITGN_HOST", "https://api.bitgn.com")
BITGN_API_KEY = os.getenv("BITGN_API_KEY", "")
BENCH_ID = os.getenv("BENCH_ID", "bitgn/pac1-dev")

# Model configuration
MODEL_ID = os.getenv("MODEL_ID", "openai/gpt-4.1")
CLASSIFIER_MODEL_ID = os.getenv("CLASSIFIER_MODEL_ID", "openai/gpt-4.1")
INSPECTOR_MODEL_ID = os.getenv("INSPECTOR_MODEL_ID", "google/gemma-4-31b-it")
COMPLETENESS_MODEL_ID = os.getenv("COMPLETENESS_MODEL_ID", "qwen/qwen3.6-plus")
CORRECTNESS_MODEL_ID = os.getenv("CORRECTNESS_MODEL_ID", "qwen/qwen3.6-plus")
INBOX_ANALYZER_MODEL_ID = os.getenv("INBOX_ANALYZER_MODEL_ID", "anthropic/claude-sonnet-4.6")
FALLBACK_MODEL_ID = os.getenv("FALLBACK_MODEL_ID", "qwen/qwen3.6-plus")

MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "30"))
