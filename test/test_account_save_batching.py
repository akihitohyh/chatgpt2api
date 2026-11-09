from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "test-auth")

import anyio
from anyio import to_thread

from api.app import apply_request_thread_pool_size
from services.account_service import AccountService
from services.config import ConfigStore
from services.storage.base import StorageBackend
from services.storage.json_storage import JSONStorageBackend


class CountingStorage(StorageBackend):
    def __init__(self, accounts: list[dict[str, Any]] | None = None) -> None:
        self.accounts = list(accounts or [])
        self.saves: list[list[dict[str, Any]]] = []
        self.lock = threading.Lock()

    def load_accounts(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.accounts]

    def save_accounts(self, accounts: list[dict[str, Any]]) -> None:
        with self.lock:
            self.saves.append([dict(item) for item in accounts])

    def load_auth_keys(self) -> list[dict[str, Any]]:
        return []

    def save_auth_keys(self, auth_keys: list[dict[str, Any]]) -> None:
        pass

    def health_check(self) -> dict[str, Any]:
        return {"status": "healthy"}

    def get_backend_info(self) -> dict[str, Any]:
        return {"type": "counting"}


def _account(token: str) -> dict[str, Any]:
    return {"access_token": token, "type": "free", "status": "正常", "quota": 5}


class JSONStorageAtomicWriteTests(unittest.TestCase):
    def test_save_writes_complete_file_and_leaves_no_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "accounts.json"
            storage = JSONStorageBackend(path)
            storage.save_accounts([{"access_token": "a"}, {"access_token": "b"}])

            self.assertEqual([item["access_token"] for item in json.loads(path.read_text("utf-8"))], ["a", "b"])
            self.assertEqual(sorted(p.name for p in Path(tmp_dir).iterdir()), ["accounts.json"])

    def test_failed_save_keeps_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "accounts.json"
            storage = JSONStorageBackend(path)
            storage.save_accounts([{"access_token": "old"}])

            with self.assertRaises(TypeError):
                storage.save_accounts([{"access_token": object()}])

            self.assertEqual(json.loads(path.read_text("utf-8")), [{"access_token": "old"}])
            self.assertEqual(sorted(p.name for p in Path(tmp_dir).iterdir()), ["accounts.json"])

    def test_corrupt_file_is_backed_up_before_returning_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "accounts.json"
            path.write_text('[{"access_token": "half', encoding="utf-8")
            storage = JSONStorageBackend(path)

            self.assertEqual(storage.load_accounts(), [])
            backups = [p for p in Path(tmp_dir).iterdir() if p.name.startswith("accounts.json.corrupt-")]
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text("utf-8"), '[{"access_token": "half')


class AccountSaveBatchingTests(unittest.TestCase):
    def test_zero_interval_saves_immediately_like_before(self) -> None:
        storage = CountingStorage([_account("a")])
        service = AccountService(storage)

        service.update_account("a", {"quota": 4}, quiet=True)

        self.assertEqual(len(storage.saves), 1)
        self.assertEqual(storage.saves[0][0]["quota"], 4)

    def test_many_updates_are_merged_into_few_saves(self) -> None:
        storage = CountingStorage([_account("a"), _account("b")])
        service = AccountService(storage, save_interval_secs=0.2)

        for quota in range(50):
            service.update_account("a", {"quota": quota}, quiet=True)
        self.assertEqual(storage.saves, [])

        deadline = time.time() + 3
        while not storage.saves and time.time() < deadline:
            time.sleep(0.02)
        time.sleep(0.3)

        self.assertGreaterEqual(len(storage.saves), 1)
        self.assertLessEqual(len(storage.saves), 2)
        final = {item["access_token"]: item for item in storage.saves[-1]}
        self.assertEqual(final["a"]["quota"], 49)
        self.assertEqual(final["b"]["quota"], 5)

    def test_flush_writes_pending_changes_once(self) -> None:
        storage = CountingStorage([_account("a")])
        service = AccountService(storage, save_interval_secs=30)

        service.update_account("a", {"quota": 1}, quiet=True)
        self.assertTrue(service.flush_pending_save())
        self.assertFalse(service.flush_pending_save())

        self.assertEqual(len(storage.saves), 1)
        self.assertEqual(storage.saves[0][0]["quota"], 1)

    def test_failed_save_is_retried(self) -> None:
        class FlakyStorage(CountingStorage):
            def __init__(self) -> None:
                super().__init__([_account("a")])
                self.fail_next = True

            def save_accounts(self, accounts: list[dict[str, Any]]) -> None:
                if self.fail_next:
                    self.fail_next = False
                    raise OSError("disk full")
                super().save_accounts(accounts)

        storage = FlakyStorage()
        service = AccountService(storage, save_interval_secs=30)
        service.update_account("a", {"quota": 2}, quiet=True)

        self.assertFalse(service.flush_pending_save())
        self.assertTrue(service.flush_pending_save())
        self.assertEqual(storage.saves[-1][0]["quota"], 2)


class ThreadPoolAndConfigTests(unittest.TestCase):
    def test_apply_request_thread_pool_size_changes_default_limiter(self) -> None:
        async def run() -> tuple[int, int]:
            before = to_thread.current_default_thread_limiter().total_tokens
            applied = apply_request_thread_pool_size(80)
            return int(before), applied

        before, applied = anyio.run(run)
        self.assertEqual(before, 40)
        self.assertEqual(applied, 80)

    def test_config_defaults_and_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps({"auth-key": "x"}), encoding="utf-8")
            store = ConfigStore(path)
            self.assertEqual(store.request_thread_pool_size, 40)
            self.assertEqual(store.account_save_interval_secs, 1.0)

            store.data.update({"request_thread_pool_size": 5000, "account_save_interval_secs": -3})
            self.assertEqual(store.request_thread_pool_size, 1000)
            self.assertEqual(store.account_save_interval_secs, 0.0)

            store.data.update({"request_thread_pool_size": "abc", "account_save_interval_secs": "x"})
            self.assertEqual(store.request_thread_pool_size, 40)
            self.assertEqual(store.account_save_interval_secs, 1.0)

            self.assertIn("request_thread_pool_size", store.get())
            self.assertIn("account_save_interval_secs", store.get())


if __name__ == "__main__":
    unittest.main()
