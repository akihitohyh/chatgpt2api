from __future__ import annotations

from anyio import to_thread


def apply_request_thread_pool_size(size: int) -> int:
    """调整同步接口共用的线程池大小（anyio 默认 40）。必须在事件循环里调用。"""
    limiter = to_thread.current_default_thread_limiter()
    limiter.total_tokens = max(1, int(size))
    return int(limiter.total_tokens)
