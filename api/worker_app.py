"""工作进程用的精简程序。

只直接处理生图 / 对话这几个耗资源的接口（号池、日志、鉴权都经管家进程），
其余所有请求（后台页面、后台接口、模型列表、图片下载、任务查询……）原样转给管家进程。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from curl_cffi.requests import AsyncSession
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.routing import APIRoute

from api import ai
from api.errors import install_exception_handlers
from services.config import config
from services.coordinator import coordinator_client
from services.process_role import coordinator_url
from services.thread_pool import apply_request_thread_pool_size

# 工作进程自己处理的接口（POST）；其余一律转给管家
WORKER_POST_PATHS = frozenset({
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/messages",
    "/v1/search",
})

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
}
_PROXY_TIMEOUT_SECS = 600


def _worker_router() -> APIRouter:
    router = APIRouter()
    for route in ai.create_router().routes:
        if isinstance(route, APIRoute) and route.path in WORKER_POST_PATHS and "POST" in route.methods:
            router.routes.append(route)
    return router


def _forward_request_headers(request: Request) -> dict[str, str]:
    # Host 原样带过去：管家按它拼图片链接等地址
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP or key.lower() == "host"
    }
    headers["accept-encoding"] = "identity"
    headers.setdefault("x-forwarded-proto", request.url.scheme)
    client = request.client.host if request.client else ""
    if client:
        previous = request.headers.get("x-forwarded-for")
        headers["x-forwarded-for"] = f"{previous}, {client}" if previous else client
    return headers


def _response_headers(upstream_headers) -> list[tuple[str, str]]:
    items = upstream_headers.multi_items() if hasattr(upstream_headers, "multi_items") else upstream_headers.items()
    has_encoding = any(key.lower() == "content-encoding" for key, _ in items)
    headers: list[tuple[str, str]] = []
    for key, value in items:
        lower = key.lower()
        if lower in _HOP_BY_HOP and not (lower == "content-length" and not has_encoding):
            continue
        if lower == "content-encoding":
            continue
        headers.append((key, value))
    return headers


def create_worker_app() -> FastAPI:
    session_holder: dict[str, AsyncSession] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        pool_size = apply_request_thread_pool_size(config.request_thread_pool_size)
        coordinator_client.connect(timeout_secs=120)
        print(f"[worker] connected to coordinator, request thread pool size = {pool_size}", flush=True)
        session_holder["session"] = AsyncSession(timeout=_PROXY_TIMEOUT_SECS, allow_redirects=False)
        try:
            yield
        finally:
            session = session_holder.pop("session", None)
            if session is not None:
                await session.close()

    # 文档页由管家提供（经下面的转发），工作进程不另开
    app = FastAPI(
        title="chatgpt2api-worker",
        version=config.app_version,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    install_exception_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(_worker_router())

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        include_in_schema=False,
    )
    async def proxy_to_coordinator(full_path: str, request: Request):
        session = session_holder["session"]
        url = f"{coordinator_url()}{request.url.path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        body = await request.body()
        upstream = await session.request(
            request.method,
            url,
            headers=_forward_request_headers(request),
            data=body or None,
            stream=True,
        )
        headers = _response_headers(upstream.headers)

        if request.method == "HEAD":
            await upstream.aclose()
            return Response(status_code=upstream.status_code, headers=dict(headers))

        async def body_iter() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_content():
                    if chunk:
                        yield chunk
            finally:
                await upstream.aclose()

        response = StreamingResponse(body_iter(), status_code=upstream.status_code)
        response.raw_headers = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in headers]
        return response

    return app


app = create_worker_app()
