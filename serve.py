"""容器启动入口。

CHATGPT2API_WORKERS 没设或为 0：和以前完全一样，单进程跑 main:app。
CHATGPT2API_WORKERS = N（>0）：多进程模式——
  1. 先起一个管家进程（main:app，只听 127.0.0.1:8081），号池、日志、鉴权、图片索引、
     后台任务都在它这里，并在本机 socket 上把这些服务提供给工作进程；
  2. 管家就绪后，再起 N 个工作进程（api.worker_app:app，听对外端口），
     生图 / 对话由工作进程处理，其余请求转给管家。
任何一边退出，另一边也会被停掉，整个容器退出交给 Docker 重启。
"""
from __future__ import annotations

import os
import secrets
import signal
import socket
import subprocess
import sys
import time

from services.process_role import (
    COORDINATOR_AUTHKEY_ENV,
    COORDINATOR_SOCKET_ENV,
    COORDINATOR_URL_ENV,
    DEFAULT_COORDINATOR_SOCKET,
    ROLE_COORDINATOR,
    ROLE_ENV,
    ROLE_WORKER,
)

STARTUP_TIMEOUT_SECS = 180
STOP_TIMEOUT_SECS = 30


def _int_env(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, "")).strip() or default)
    except ValueError:
        return default


def _uvicorn(app: str, host: str, port: int, *extra: str) -> list[str]:
    return [sys.executable, "-m", "uvicorn", app, "--host", host, "--port", str(port), "--access-log", *extra]


def _wait_ready(process: subprocess.Popen, socket_path: str, port: int) -> None:
    deadline = time.time() + STARTUP_TIMEOUT_SECS
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"[serve] 管家进程启动失败，退出码 {process.returncode}")
        if os.path.exists(socket_path):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
            except OSError:
                pass
        time.sleep(0.5)
    raise SystemExit("[serve] 等管家进程就绪超时")


def _stop(process: subprocess.Popen | None, name: str) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=STOP_TIMEOUT_SECS)
    except subprocess.TimeoutExpired:
        print(f"[serve] {name} 没按时退出，强制结束", flush=True)
        process.kill()
        process.wait()


def build_commands(workers: int, host: str, port: int, coordinator_port: int) -> tuple[list[str], list[str]]:
    coordinator = _uvicorn("main:app", "127.0.0.1", coordinator_port)
    worker = _uvicorn("api.worker_app:app", host, port, "--workers", str(workers))
    return coordinator, worker


def main() -> int:
    workers = _int_env("CHATGPT2API_WORKERS", 0)
    host = os.getenv("CHATGPT2API_HOST", "0.0.0.0").strip() or "0.0.0.0"
    port = _int_env("CHATGPT2API_PORT", 80)

    if workers <= 0:
        command = _uvicorn("main:app", host, port)
        os.execv(command[0], command)

    coordinator_port = _int_env("CHATGPT2API_COORDINATOR_PORT", 8081)
    socket_path = os.getenv(COORDINATOR_SOCKET_ENV, "").strip() or DEFAULT_COORDINATOR_SOCKET
    shared_env = {
        **os.environ,
        COORDINATOR_SOCKET_ENV: socket_path,
        COORDINATOR_AUTHKEY_ENV: secrets.token_hex(32),
        COORDINATOR_URL_ENV: f"http://127.0.0.1:{coordinator_port}",
    }
    coordinator_cmd, worker_cmd = build_commands(workers, host, port, coordinator_port)

    coordinator: subprocess.Popen | None = None
    worker: subprocess.Popen | None = None
    stopping = {"flag": False}

    def handle_signal(signum, _frame):
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        print(f"[serve] 多进程模式：1 个管家 + {workers} 个工作进程", flush=True)
        coordinator = subprocess.Popen(coordinator_cmd, env={**shared_env, ROLE_ENV: ROLE_COORDINATOR})
        _wait_ready(coordinator, socket_path, coordinator_port)
        worker = subprocess.Popen(worker_cmd, env={**shared_env, ROLE_ENV: ROLE_WORKER})
        while not stopping["flag"]:
            if coordinator.poll() is not None:
                print(f"[serve] 管家进程退出（{coordinator.returncode}），停掉工作进程", flush=True)
                return coordinator.returncode or 1
            if worker.poll() is not None:
                print(f"[serve] 工作进程组退出（{worker.returncode}），停掉管家", flush=True)
                return worker.returncode or 1
            time.sleep(1)
        return 0
    finally:
        # 先停工作进程（不再接新请求、把手上的做完），再停管家（它退出时会把账号和索引写盘）
        _stop(worker, "工作进程")
        _stop(coordinator, "管家进程")


if __name__ == "__main__":
    sys.exit(main())
