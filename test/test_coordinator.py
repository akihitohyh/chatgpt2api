from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "test-auth")

from services import coordinator as coordinator_module
from services.account_service import AccountService
from services.coordinator import (
    AccountsFacade,
    CoordinatorServer,
    RemoteAccountService,
    RemoteAuthService,
    RemoteLogService,
    coordinator_client,
    register_remote_image,
)
from services.process_role import COORDINATOR_AUTHKEY_ENV, COORDINATOR_SOCKET_ENV
from services.storage.json_storage import JSONStorageBackend


def _jwt(payload: dict[str, Any]) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{body}.sig"


class FakeLogs:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, dict | None]] = []

    def add(self, type: str, summary: str = "", detail: dict | None = None) -> None:
        self.items.append((type, summary, detail))


class FakeAuth:
    def authenticate(self, raw_key: str) -> dict | None:
        return {"id": "k1", "role": "user"} if raw_key == "good" else None


class FakeImages:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}

    def add_index_item(self, rel: str, item: dict) -> None:
        self.items[rel] = item


def _account_service(tmp_dir: str, tokens: list[str]) -> AccountService:
    path = Path(tmp_dir) / "accounts.json"
    path.write_text(json.dumps([
        {"access_token": token, "type": "free", "status": "正常", "quota": 5} for token in tokens
    ]), encoding="utf-8")
    service = AccountService(JSONStorageBackend(path))
    # 选号时会去 ChatGPT 校验号，这里直接返回本地记录
    service.fetch_remote_info = lambda token, event="", defer_invalid_removal=True: service.get_account(token)
    return service


class CoordinatorRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.socket_path = str(Path(self.tmp.name) / "coord.sock")
        self.env = patch.dict(os.environ, {
            COORDINATOR_SOCKET_ENV: self.socket_path,
            COORDINATOR_AUTHKEY_ENV: "secret-for-test",
        })
        self.env.start()
        coordinator_client.reset()
        self.accounts = _account_service(self.tmp.name, ["tok-a", "tok-b"])
        self.logs = FakeLogs()
        self.images = FakeImages()
        self.server = CoordinatorServer(
            accounts=self.accounts, logs=self.logs, auth=FakeAuth(), images=self.images,
        )
        self.server.start()
        self.remote = RemoteAccountService()

    def tearDown(self) -> None:
        self.server.stop()
        coordinator_client.reset()
        self.env.stop()
        self.tmp.cleanup()

    def test_image_slot_is_taken_in_coordinator_and_released_by_result(self) -> None:
        token = self.remote.get_available_access_token()
        self.assertIn(token, {"tok-a", "tok-b"})
        self.assertEqual(self.accounts._image_inflight.get(token), 1)
        self.assertEqual(self.server.accounts.owner_slot_counts(), {os.getpid(): 1})

        result = self.remote.mark_image_result(token, True)

        self.assertEqual(result["quota"], 4)
        self.assertEqual(self.accounts.get_account(token)["quota"], 4)
        self.assertNotIn(token, self.accounts._image_inflight)
        self.assertEqual(self.server.accounts.owner_slot_counts(), {})

    def test_release_image_slot_round_trip(self) -> None:
        token = self.remote.get_available_access_token()
        self.remote.release_image_slot(token)
        self.assertNotIn(token, self.accounts._image_inflight)
        self.assertEqual(self.server.accounts.owner_slot_counts(), {})

    def test_errors_from_coordinator_are_raised_in_worker(self) -> None:
        self.accounts.update_account("tok-a", {"quota": 0, "status": "限流"}, quiet=True)
        self.accounts.update_account("tok-b", {"quota": 0, "status": "限流"}, quiet=True)
        with self.assertRaisesRegex(RuntimeError, "no available image quota"):
            self.remote.get_available_access_token()

    def test_other_account_calls(self) -> None:
        self.assertEqual(self.remote.get_account("tok-a")["access_token"], "tok-a")
        self.assertIsNone(self.remote.get_account(""))
        self.remote.cool_down_image_account("tok-a", 60, reason="test")
        self.assertGreater(self.remote.image_cooldown_remaining("tok-a"), 50)
        self.assertEqual(len(self.remote.list_accounts()), 2)
        self.assertFalse(self.remote.flush_pending_save())

    def test_static_helpers_run_locally_and_stateful_methods_are_refused(self) -> None:
        self.assertEqual(self.remote._decode_jwt_payload(_jwt({"sub": "x"}))["sub"], "x")
        self.assertEqual(self.remote._normalize_source_type(None), "web")
        with self.assertRaises(AttributeError):
            self.remote.delete_accounts(["tok-a"])

    def test_logs_auth_and_images_go_to_coordinator(self) -> None:
        RemoteLogService().add("call", "生图", {"a": 1})
        RemoteLogService().add("call", "其它", b=2)
        self.assertEqual(self.logs.items, [("call", "生图", {"a": 1}), ("call", "其它", {"b": 2})])
        self.assertEqual(RemoteAuthService().authenticate("good"), {"id": "k1", "role": "user"})
        self.assertIsNone(RemoteAuthService().authenticate("bad"))
        register_remote_image("2026/11/09/x.png", {"rel": "2026/11/09/x.png"})
        self.assertEqual(self.images.items, {"2026/11/09/x.png": {"rel": "2026/11/09/x.png"}})

    def test_wrong_authkey_cannot_connect(self) -> None:
        coordinator_client.reset()
        with patch.dict(os.environ, {COORDINATOR_AUTHKEY_ENV: "wrong"}):
            with self.assertRaises(Exception):
                coordinator_client.connect(timeout_secs=0)
        coordinator_client.reset()


class DeadWorkerSlotTests(unittest.TestCase):
    def test_slots_of_exited_worker_are_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            accounts = _account_service(tmp_dir, ["tok-a"])
            facade = AccountsFacade(accounts)
            dead_pid = 2 ** 22 + 12345
            facade.acquire_image_token(dead_pid)
            facade.acquire_image_token(os.getpid())
            self.assertEqual(accounts._image_inflight.get("tok-a"), 2)

            with patch.object(coordinator_module, "_pid_alive", lambda pid: pid == os.getpid()):
                released = facade.release_dead_owners()

            self.assertEqual(released, 1)
            self.assertEqual(accounts._image_inflight.get("tok-a"), 1)
            self.assertEqual(facade.owner_slot_counts(), {os.getpid(): 1})


if __name__ == "__main__":
    unittest.main()
