"""端到端：真起 serve.py（1 个管家 + 2 个工作进程），验证分工、转发和退出。

需要几秒钟；设环境变量 CHATGPT2API_SKIP_E2E=1 可跳过。
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTH_KEY = "e2e-admin-key"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _request(url: str, *, method: str = "GET", body: dict | None = None, token: str | None = None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"content-type": "application/json"} if body is not None else {}
    if token:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace"), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace"), dict(exc.headers)


@unittest.skipIf(os.getenv("CHATGPT2API_SKIP_E2E"), "CHATGPT2API_SKIP_E2E set")
class MultiprocessEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        app_dir = Path(cls.tmp.name) / "app"
        app_dir.mkdir()
        for name in ("api", "services", "utils"):
            shutil.copytree(ROOT / name, app_dir / name, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("main.py", "serve.py", "VERSION"):
            shutil.copy2(ROOT / name, app_dir / name)
        (app_dir / "config.json").write_text(json.dumps({"auth-key": AUTH_KEY}), "utf-8")
        cls.app_dir = app_dir
        cls.port = _free_port()
        cls.socket_path = str(Path(cls.tmp.name) / "coord.sock")
        env = {
            **os.environ,
            "CHATGPT2API_WORKERS": "2",
            "CHATGPT2API_HOST": "127.0.0.1",
            "CHATGPT2API_PORT": str(cls.port),
            "CHATGPT2API_COORDINATOR_PORT": str(_free_port()),
            "CHATGPT2API_COORDINATOR_SOCKET": cls.socket_path,
            "STORAGE_BACKEND": "json",
        }
        env.pop("CHATGPT2API_AUTH_KEY", None)
        env.pop("CHATGPT2API_PROCESS_ROLE", None)
        cls.log_path = Path(cls.tmp.name) / "serve.log"
        cls.log_file = cls.log_path.open("w")
        cls.proc = subprocess.Popen(
            [sys.executable, "serve.py"], cwd=app_dir, env=env, stdout=cls.log_file, stderr=subprocess.STDOUT,
        )
        cls.base = f"http://127.0.0.1:{cls.port}"
        deadline = time.time() + 90
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                break
            try:
                status, _, _ = _request(f"{cls.base}/version")
                if status == 200:
                    return
            except OSError:
                pass
            time.sleep(0.5)
        cls.tearDownClass()
        raise AssertionError("serve.py 没能启动：\n" + cls.log_path.read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "proc", None) is not None and cls.proc.poll() is None:
            cls.proc.send_signal(signal.SIGTERM)
            try:
                cls.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
        if getattr(cls, "log_file", None) is not None:
            cls.log_file.close()
        cls.tmp.cleanup()

    def test_non_generation_requests_are_served_by_coordinator(self) -> None:
        status, body, _ = _request(f"{self.base}/version")
        self.assertEqual(status, 200, body)
        status, body, _ = _request(f"{self.base}/api/settings", token=AUTH_KEY)
        self.assertEqual(status, 200, body)
        self.assertIn("request_thread_pool_size", body)

    def test_generation_runs_in_worker_and_uses_coordinator_pool(self) -> None:
        status, body, _ = _request(
            f"{self.base}/v1/images/generations",
            method="POST",
            body={"model": "gpt-image-2", "prompt": "a red apple"},
            token=AUTH_KEY,
        )
        # 号池是空的：工作进程问管家要号，管家回"没有可用额度"，错误原样传回客户端
        self.assertGreaterEqual(status, 400, body)
        self.assertIn("quota", body.lower())
        # 调用日志由管家进程写进它自己的 logs.jsonl
        deadline = time.time() + 5
        logs = self.app_dir / "data" / "logs.jsonl"
        while time.time() < deadline and not (logs.exists() and "/v1/images/generations" in logs.read_text("utf-8")):
            time.sleep(0.2)
        self.assertIn("/v1/images/generations", logs.read_text("utf-8"))

    def test_unauthorized_generation_is_rejected_by_worker(self) -> None:
        status, body, _ = _request(
            f"{self.base}/v1/images/generations",
            method="POST",
            body={"model": "gpt-image-2", "prompt": "x"},
            token="wrong-key",
        )
        self.assertEqual(status, 401, body)

    def test_started_one_coordinator_and_two_workers(self) -> None:
        log = self.log_path.read_text()
        self.assertIn("1 个管家 + 2 个工作进程", log)
        self.assertGreaterEqual(log.count("[worker] connected to coordinator"), 2, log)
        self.assertTrue(os.path.exists(self.socket_path))


if __name__ == "__main__":
    unittest.main()
