"""FastAPI 路由注册 — chat / search / documents / health / auth / stream / eval"""
from .chat import router as chat_router
from .search import router as search_router
from .documents import router as documents_router
from .health import router as health_router
from .auth import router as auth_router
from .stream import router as stream_router
from .eval import router as eval_router
