import os
from functools import partial
from dotenv import load_dotenv

load_dotenv()

try:
    from langsmith import Client
    from langsmith.run_helpers import traceable
except ImportError:
    # Fall back to a no-op decorator if LangSmith is not installed.
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    Client = None
    traceable = traceable
    langsmith_client = None
    LANGSMITH_PROJECT_NAME = None
else:
    # Ensure tracing is enabled and the API key is available.
    os.environ.setdefault("LANGSMITH_TRACING_V2", "true")
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    try:
        langsmith_client = Client(api_key=api_key) if api_key else Client()
    except Exception:
        langsmith_client = None
    LANGSMITH_PROJECT_NAME = os.getenv("LANGSMITH_PROJECT_NAME", "doc_chatbot")

# Export a ready-to-use decorator that binds the LangSmith client/project.
if "traceable" in globals() and not callable(traceable):
    # No-op trace helper already defined.
    langsmith_traceable = traceable
else:
    langsmith_traceable = partial(
        traceable,
        client=langsmith_client,
        project_name=LANGSMITH_PROJECT_NAME,
    )
