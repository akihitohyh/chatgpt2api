"""多进程运行时的进程角色。

- single（默认）：和以前一样，一个进程干所有事。
- coordinator：管家进程。号池、日志、鉴权、图片索引、后台任务都只在这里，
  并通过本机 socket 把这些服务提供给工作进程；同时照常提供后台页面等全部接口。
- worker：工作进程。只处理生图 / 对话这类耗资源的接口，状态全部问管家要，
  其余请求原样转给管家。

本模块只能依赖标准库，config 等模块在导入时会用到它。
"""
from __future__ import annotations

import os

ROLE_ENV = "CHATGPT2API_PROCESS_ROLE"
COORDINATOR_SOCKET_ENV = "CHATGPT2API_COORDINATOR_SOCKET"
COORDINATOR_AUTHKEY_ENV = "CHATGPT2API_COORDINATOR_AUTHKEY"
COORDINATOR_URL_ENV = "CHATGPT2API_COORDINATOR_URL"

DEFAULT_COORDINATOR_SOCKET = "/tmp/chatgpt2api-coordinator.sock"
DEFAULT_COORDINATOR_URL = "http://127.0.0.1:8081"

ROLE_SINGLE = "single"
ROLE_COORDINATOR = "coordinator"
ROLE_WORKER = "worker"


def process_role() -> str:
    value = os.getenv(ROLE_ENV, "").strip().lower()
    if value in {ROLE_COORDINATOR, ROLE_WORKER}:
        return value
    return ROLE_SINGLE


def is_worker() -> bool:
    return process_role() == ROLE_WORKER


def is_coordinator() -> bool:
    return process_role() == ROLE_COORDINATOR


def coordinator_socket_path() -> str:
    return os.getenv(COORDINATOR_SOCKET_ENV, "").strip() or DEFAULT_COORDINATOR_SOCKET


def coordinator_authkey() -> bytes:
    value = os.getenv(COORDINATOR_AUTHKEY_ENV, "").strip()
    if not value:
        raise RuntimeError(f"多进程模式缺少 {COORDINATOR_AUTHKEY_ENV}，请用 serve.py 启动")
    return value.encode("utf-8")


def coordinator_url() -> str:
    return (os.getenv(COORDINATOR_URL_ENV, "").strip() or DEFAULT_COORDINATOR_URL).rstrip("/")


class WorkerUnavailable:
    """工作进程里不该被用到的服务的占位对象：一旦被调用就明确报错。"""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, item: str):
        raise RuntimeError(f"{self._name}.{item} 只能在管家进程里使用，工作进程应把这类请求转给管家")
