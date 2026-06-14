"""
backend/api/__init__.py — API 层入口
"""

from .chat import router as chat_router
from .search import router as search_router
from .documents import router as documents_router
from .health import router as health_router
from .auth import router as auth_router
from .stream import router as stream_router
from .traces import router as traces_router  # P2.2 Trace Viewer
from .eval import router as eval_router  # P2.3 Evaluation dashboard

__all__ = [
    "chat_router",
    "search_router",
    "documents_router",
    "health_router",
    "auth_router",
    "stream_router",
    "traces_router",
    "eval_router",
]
