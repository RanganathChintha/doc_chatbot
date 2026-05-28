"""Single source of truth for LangSmith configuration and tracing.

All other modules should import `langsmith_traceable` (or the convenience
re-exports below) from here — nothing LangSmith-related belongs in
config.py or anywhere else.
"""

import logging
import os
from functools import partial

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Env-driven configuration (one place)
# ─────────────────────────────────────────────
LANGSMITH_API_KEY: str | None = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT_NAME: str = os.getenv("LANGSMITH_PROJECT_NAME", "doc_chatbot")

# Mirror to LANGCHAIN_API_KEY so LangChain's auto-tracing picks it up too.
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT_NAME)
else:
    os.environ["LANGSMITH_TRACING_V2"] = "false"

# ─────────────────────────────────────────────
# Client + decorator
# ─────────────────────────────────────────────
try:
    from langsmith import Client
    from langsmith.run_helpers import traceable as _traceable
except ImportError:
    Client = None
    _traceable = None

langsmith_client = None
if Client is not None and LANGSMITH_API_KEY:
    try:
        langsmith_client = Client(api_key=LANGSMITH_API_KEY)
    except Exception as exc:
        logger.warning("LangSmith client init failed: %s", exc)

if _traceable is None:
    # LangSmith not installed → no-op decorator.
    def _noop(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    langsmith_traceable = _noop
elif langsmith_client is None:
    # Library installed but no key → keep traceable but unbound.
    langsmith_traceable = _traceable
else:
    langsmith_traceable = partial(
        _traceable,
        client=langsmith_client,
        project_name=LANGSMITH_PROJECT_NAME,
    )

# Convenience alias so other modules can simply do:
#     from langsmith_tracing import traceable
traceable = langsmith_traceable

__all__ = [
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT_NAME",
    "langsmith_client",
    "langsmith_traceable",
    "traceable",
]
