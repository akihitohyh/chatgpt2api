from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "test-auth")

from services import image_storage_service as storage_module
from services.config import ConfigStore
from services.image_storage_service import ImageStorageService


def _item(rel: str) -> dict[str, object]:
    return {"rel": rel, "path": rel, "name": Path(rel).name, "storage": "local", "local": True, "webdav": False}


class ImageIndexCacheTests(unittest.TestCase):
    def test_new_images_are_merged_into_one_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index = Path(tmp_dir) / "image_index.json"
            service = ImageStorageService(index, save_interval_secs=0.2)

            writes = []
            original = service._write_snapshot

            def counting(gen, items):
                writes.append(len(items))
                original(gen, items)

            service._write_snapshot = counting
            for i in range(30):
                service.add_index_item(f"2026/11/09/{i}_x.png", _item(f"2026/11/09/{i}_x.png"))

            # 还没写盘时，内存里已经能查到
            self.assertIn("2026/11/09/29_x.png", service._load_clean_index())
            deadline = time.time() + 3
            while not writes and time.time() < deadline:
                time.sleep(0.02)
            time.sleep(0.3)

            self.assertEqual(writes, [30])
            on_disk = json.loads(index.read_text("utf-8"))["items"]
            self.assertEqual(len(on_disk), 30)

    def test_flush_and_immediate_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index = Path(tmp_dir) / "image_index.json"
            service = ImageStorageService(index, save_interval_secs=30)
            service.add_index_item("2026/11/09/a_x.png", _item("2026/11/09/a_x.png"))
            self.assertFalse(index.exists())
            self.assertTrue(service.flush_index())
            self.assertFalse(service.flush_index())
            self.assertIn("2026/11/09/a_x.png", json.loads(index.read_text("utf-8"))["items"])

            immediate = ImageStorageService(Path(tmp_dir) / "other.json", save_interval_secs=0)
            immediate.add_index_item("2026/11/09/b_x.png", _item("2026/11/09/b_x.png"))
            self.assertIn("2026/11/09/b_x.png", json.loads((Path(tmp_dir) / "other.json").read_text("utf-8"))["items"])

    def test_existing_index_is_loaded_once_and_kept_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index = Path(tmp_dir) / "image_index.json"
            index.write_text(json.dumps({"items": {"2026/11/08/old_x.png": _item("2026/11/08/old_x.png")}}), "utf-8")
            service = ImageStorageService(index, save_interval_secs=30)
            with patch.object(service, "_load_index", wraps=service._load_index) as load:
                service._load_clean_index()
                service._load_clean_index()
                service.add_index_item("2026/11/09/new_x.png", _item("2026/11/09/new_x.png"))
                self.assertEqual(load.call_count, 1)
            service.flush_index()
            items = json.loads(index.read_text("utf-8"))["items"]
            self.assertEqual(set(items), {"2026/11/08/old_x.png", "2026/11/09/new_x.png"})

    def test_older_snapshot_never_overwrites_newer_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index = Path(tmp_dir) / "image_index.json"
            service = ImageStorageService(index, save_interval_secs=30)
            service._write_snapshot(2, {"new": {}})
            service._write_snapshot(1, {"old": {}})
            self.assertEqual(json.loads(index.read_text("utf-8"))["items"], {"new": {}})
            self.assertEqual(sorted(p.name for p in Path(tmp_dir).iterdir()), ["image_index.json"])

    def test_directory_scan_on_save_is_throttled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = ImageStorageService(Path(tmp_dir) / "image_index.json", save_interval_secs=30)
            with patch.object(storage_module, "_last_save_cleanup_at", 0.0), \
                    patch.object(storage_module.config, "cleanup_old_images") as cleanup, \
                    patch.object(storage_module, "_local_image_path", lambda rel: Path(tmp_dir) / rel):
                for _ in range(3):
                    service.save(b"\x89PNG\r\n\x1a\nfake", "http://x")
            self.assertEqual(cleanup.call_count, 1)


class WorkerConfigReloadTests(unittest.TestCase):
    def test_worker_config_follows_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps({"auth-key": "x", "image_poll_timeout_secs": 300}), "utf-8")
            store = ConfigStore(path)
            store._watch_file = True
            store._RELOAD_CHECK_INTERVAL_SECS = 0
            self.assertEqual(store.image_poll_timeout_secs, 300)

            time.sleep(0.01)
            path.write_text(json.dumps({"auth-key": "x", "image_poll_timeout_secs": 120}), "utf-8")
            os.utime(path, ns=(time.time_ns(), time.time_ns() + 1_000_000))
            self.assertEqual(store.image_poll_timeout_secs, 120)

    def test_image_index_save_interval_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps({"auth-key": "x"}), "utf-8")
            store = ConfigStore(path)
            self.assertEqual(store.image_index_save_interval_secs, 1.0)
            store.data["image_index_save_interval_secs"] = 999
            self.assertEqual(store.image_index_save_interval_secs, 60.0)
            store.data["image_index_save_interval_secs"] = "x"
            self.assertEqual(store.image_index_save_interval_secs, 1.0)
            self.assertIn("image_index_save_interval_secs", store.get())

    def test_single_process_config_does_not_watch_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps({"auth-key": "x", "image_poll_timeout_secs": 300}), "utf-8")
            store = ConfigStore(path)
            self.assertFalse(store._watch_file)


if __name__ == "__main__":
    unittest.main()
