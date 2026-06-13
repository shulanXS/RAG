"""
backend/api/__init__.py — API 层入口
"""

from .chat import router as chat_router
from .search import router as search_router
from .documents import router as documents_router
from .health import router as health_router

__all__ = ["chat_router", "search_router", "documents_router", "health_router"]
