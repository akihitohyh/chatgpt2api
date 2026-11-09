"""管家进程和工作进程之间的通信。

管家进程在本机 unix socket 上开一个 multiprocessing manager，把号池、日志、鉴权、
图片索引这几个有状态的服务包成"门面"对象提供出去；工作进程连上来以后，拿到的是
同名方法的代理，调用时参数和返回值经 socket 传递，状态只在管家进程里有一份。

生图占用的号位子按工作进程的 pid 记账：某个工作进程意外退出时，它占着的位子
会被管家自动释放，不会一直卡着那些号。
"""
from __future__ import annotations

import inspect
import os
import threading
import time
from collections import Counter
from multiprocessing import current_process
from multiprocessing.managers import BaseManager
from typing import Any

from services.process_role import coordinator_authkey, coordinator_socket_path

_OWNER_CHECK_INTERVAL_SECS = 15.0


class AccountsFacade:
    """管家进程里包住真实 AccountService 的门面，负责按工作进程记账生图位子。"""

    def __init__(self, accounts: Any) -> None:
        self._accounts = accounts
        self._lock = threading.Lock()
        self._owner_slots: dict[int, Counter[str]] = {}

    # ---- 生图位子（带记账） ----
    def acquire_image_token(
            self,
            owner: int,
            plan_type: str | None = None,
            source_type: str | None = None,
            plan_types: Any = None,
    ) -> str:
        token = self._accounts.get_available_access_token(
            plan_type=plan_type,
            source_type=source_type,
            plan_types=plan_types,
        )
        with self._lock:
            self._owner_slots.setdefault(int(owner), Counter())[token] += 1
        return token

    def _forget_slot(self, owner: int, token: str) -> None:
        with self._lock:
            slots = self._owner_slots.get(int(owner))
            if not slots:
                return
            if slots.get(token, 0) > 0:
                slots[token] -= 1
                if slots[token] <= 0:
                    del slots[token]
            if not slots:
                self._owner_slots.pop(int(owner), None)

    def mark_image_result(self, owner: int, token: str, success: bool) -> dict | None:
        try:
            return self._accounts.mark_image_result(token, success)
        finally:
            self._forget_slot(owner, token)

    def release_image_slot(self, owner: int, token: str) -> None:
        try:
            self._accounts.release_image_slot(token)
        finally:
            self._forget_slot(owner, token)

    def owner_slot_counts(self) -> dict[int, int]:
        with self._lock:
            return {owner: sum(slots.values()) for owner, slots in self._owner_slots.items()}

    def release_owner(self, owner: int) -> int:
        """把某个工作进程占着的位子全部还回去，返回释放的个数。"""
        with self._lock:
            slots = self._owner_slots.pop(int(owner), None) or Counter()
        released = 0
        for token, count in slots.items():
            for _ in range(count):
                self._accounts.release_image_slot(token)
                released += 1
        return released

    def release_dead_owners(self) -> int:
        with self._lock:
            owners = list(self._owner_slots)
        released = 0
        for owner in owners:
            if not _pid_alive(owner):
                count = self.release_owner(owner)
                if count:
                    print(f"[coordinator] 工作进程 {owner} 已退出，释放它占着的 {count} 个生图位子")
                released += count
        return released

    # ---- 其余直接转发 ----
    def get_account(self, token: str) -> dict | None:
        return self._accounts.get_account(token)

    def get_text_access_token(self, excluded_tokens: Any = None, model: str = "auto") -> str:
        return self._accounts.get_text_access_token(excluded_tokens=excluded_tokens, model=model)

    def mark_text_used(self, token: str) -> None:
        self._accounts.mark_text_used(token)

    def refresh_access_token(self, token: str, force: bool = False, event: str = "refresh_access_token") -> str:
        return self._accounts.refresh_access_token(token, force=force, event=event)

    def remove_invalid_token(self, token: str, event: str, quiet: bool = False) -> bool:
        return self._accounts.remove_invalid_token(token, event, quiet=quiet)

    def cool_down_image_account(self, token: str, seconds: float, reason: str = "") -> None:
        self._accounts.cool_down_image_account(token, seconds, reason=reason)

    def image_cooldown_remaining(self, token: str) -> float:
        return self._accounts.image_cooldown_remaining(token)

    def list_accounts(self) -> list[dict]:
        return self._accounts.list_accounts()


class LogsFacade:
    def __init__(self, logs: Any) -> None:
        self._logs = logs

    def add(self, type: str, summary: str = "", detail: dict | None = None) -> None:
        self._logs.add(type, summary, detail)


class AuthFacade:
    def __init__(self, auth: Any) -> None:
        self._auth = auth

    def authenticate(self, raw_key: str) -> dict | None:
        return self._auth.authenticate(raw_key)


class ImagesFacade:
    def __init__(self, images: Any) -> None:
        self._images = images

    def register(self, rel: str, item: dict) -> None:
        self._images.add_index_item(rel, item)


_ACCOUNT_METHODS = [name for name in AccountsFacade.__dict__ if not name.startswith("_")]

_SERVER_OBJECTS: dict[str, Any] = {}


def _server_object(name: str):
    def factory():
        obj = _SERVER_OBJECTS.get(name)
        if obj is None:
            raise RuntimeError(f"coordinator object {name} is not ready")
        return obj

    factory.__name__ = f"coordinator_{name}"
    return factory


class CoordinatorManager(BaseManager):
    pass


CoordinatorManager.register("accounts", callable=_server_object("accounts"), exposed=_ACCOUNT_METHODS)
CoordinatorManager.register("logs", callable=_server_object("logs"), exposed=["add"])
CoordinatorManager.register("auth", callable=_server_object("auth"), exposed=["authenticate"])
CoordinatorManager.register("images", callable=_server_object("images"), exposed=["register"])


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class CoordinatorServer:
    def __init__(self, *, accounts: Any, logs: Any, auth: Any, images: Any,
                 socket_path: str | None = None, authkey: bytes | None = None) -> None:
        self.socket_path = socket_path or coordinator_socket_path()
        self.authkey = authkey or coordinator_authkey()
        self.accounts = AccountsFacade(accounts)
        _SERVER_OBJECTS.update({
            "accounts": self.accounts,
            "logs": LogsFacade(logs),
            "auth": AuthFacade(auth),
            "images": ImagesFacade(images),
        })
        self._server = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        manager = CoordinatorManager(address=self.socket_path, authkey=self.authkey)
        self._server = manager.get_server()
        os.chmod(self.socket_path, 0o600)
        # 不用自带的 serve_forever：它的接收线程在 socket 关掉后会空转吃 CPU，没法正常停
        current_process()._manager_server = self._server
        self._server.stop_event = threading.Event()  # serve_forever 里才会建，各连接线程要用
        serve = threading.Thread(target=self._accept_loop, name="coordinator-rpc", daemon=True)
        janitor = threading.Thread(target=self._janitor_loop, name="coordinator-janitor", daemon=True)
        self._threads = [serve, janitor]
        for thread in self._threads:
            thread.start()
        print(f"[coordinator] listening on {self.socket_path}", flush=True)

    def _accept_loop(self) -> None:
        listener = self._server.listener
        while not self._stop.is_set():
            try:
                connection = listener.accept()
            except OSError:
                if self._stop.is_set():
                    return
                continue
            threading.Thread(
                target=self._server.handle_request,
                args=(connection,),
                name="coordinator-rpc-conn",
                daemon=True,
            ).start()

    def _janitor_loop(self) -> None:
        while not self._stop.wait(_OWNER_CHECK_INTERVAL_SECS):
            try:
                self.accounts.release_dead_owners()
            except Exception as exc:
                print(f"[coordinator] janitor fail {exc}")

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.stop_event.set()
            try:
                # 监听器关闭时会自己删掉 socket 文件（并撤销退出时的清理钩子）
                self._server.listener.close()
            except OSError:
                pass


# ---------------- 工作进程这一侧 ----------------

class _CoordinatorClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._manager: CoordinatorManager | None = None
        self._proxies: dict[str, Any] = {}

    def connect(self, timeout_secs: float = 60.0) -> None:
        with self._lock:
            if self._manager is not None:
                return
            deadline = time.time() + max(0.0, timeout_secs)
            last_error: Exception | None = None
            while True:
                manager = CoordinatorManager(address=coordinator_socket_path(), authkey=coordinator_authkey())
                try:
                    manager.connect()
                    self._manager = manager
                    return
                except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                    last_error = exc
                if time.time() >= deadline:
                    raise RuntimeError(f"连不上管家进程（{coordinator_socket_path()}）：{last_error}")
                time.sleep(0.2)

    def proxy(self, name: str) -> Any:
        proxy = self._proxies.get(name)
        if proxy is not None:
            return proxy
        self.connect()
        with self._lock:
            proxy = self._proxies.get(name)
            if proxy is None:
                proxy = getattr(self._manager, name)()
                self._proxies[name] = proxy
            return proxy

    def reset(self) -> None:
        with self._lock:
            self._manager = None
            self._proxies = {}


coordinator_client = _CoordinatorClient()


class RemoteAccountService:
    """工作进程里的 account_service：方法名和 AccountService 一样，实际交给管家处理。"""

    def _remote(self):
        return coordinator_client.proxy("accounts")

    def get_available_access_token(self, plan_type=None, source_type=None, plan_types=None) -> str:
        return self._remote().acquire_image_token(os.getpid(), plan_type, source_type, plan_types)

    def mark_image_result(self, access_token: str, success: bool) -> dict | None:
        if not access_token:
            return None
        return self._remote().mark_image_result(os.getpid(), access_token, success)

    def release_image_slot(self, access_token: str) -> None:
        if access_token:
            self._remote().release_image_slot(os.getpid(), access_token)

    def get_account(self, access_token: str) -> dict | None:
        if not access_token:
            return None
        return self._remote().get_account(access_token)

    def get_text_access_token(self, excluded_tokens=None, model: str = "auto") -> str:
        return self._remote().get_text_access_token(excluded_tokens, model)

    def mark_text_used(self, access_token: str) -> None:
        if access_token:
            self._remote().mark_text_used(access_token)

    def refresh_access_token(self, access_token: str, *, force: bool = False, event: str = "refresh_access_token") -> str:
        if not access_token:
            return ""
        return self._remote().refresh_access_token(access_token, force, event)

    def remove_invalid_token(self, access_token: str, event: str, quiet: bool = False) -> bool:
        return self._remote().remove_invalid_token(access_token, event, quiet)

    def cool_down_image_account(self, access_token: str, seconds: float, reason: str = "") -> None:
        if access_token and seconds > 0:
            self._remote().cool_down_image_account(access_token, seconds, reason)

    def image_cooldown_remaining(self, access_token: str) -> float:
        return self._remote().image_cooldown_remaining(access_token)

    def list_accounts(self) -> list[dict]:
        return self._remote().list_accounts()

    def flush_pending_save(self) -> bool:
        return False

    def __getattr__(self, name: str):
        # 只放行 AccountService 上不碰状态的静态 / 类方法（例如 _decode_jwt_payload）
        from services.account_service import AccountService

        try:
            static = inspect.getattr_static(AccountService, name)
        except AttributeError:
            raise AttributeError(name) from None
        if isinstance(static, (staticmethod, classmethod)):
            return getattr(AccountService, name)
        raise AttributeError(f"account_service.{name} 在工作进程里不可用（只能在管家进程里调用）")


class RemoteLogService:
    def add(self, type: str, summary: str = "", detail: dict | None = None, **data: Any) -> None:
        coordinator_client.proxy("logs").add(type, summary, detail or data)

    def __getattr__(self, name: str):
        raise AttributeError(f"log_service.{name} 在工作进程里不可用（只能在管家进程里调用）")


class RemoteAuthService:
    def authenticate(self, raw_key: str) -> dict | None:
        return coordinator_client.proxy("auth").authenticate(raw_key)

    def __getattr__(self, name: str):
        raise AttributeError(f"auth_service.{name} 在工作进程里不可用（只能在管家进程里调用）")


def register_remote_image(rel: str, item: dict) -> None:
    coordinator_client.proxy("images").register(rel, item)
