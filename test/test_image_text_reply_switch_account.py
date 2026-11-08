from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "test-auth")

from services.account_service import AccountService
from services.protocol import conversation
from services.protocol.conversation import (
    IMAGE_NO_TOOL_COOLDOWN_SECS,
    IMAGE_TEXT_REPLY_NO_IMAGE_TOOL,
    IMAGE_TEXT_REPLY_RATE_LIMITED,
    ConversationRequest,
    ImageGenerationError,
    ImageOutput,
    _generate_single_image,
    classify_image_text_reply,
    image_text_reply_cooldown_secs,
)
from services.storage.json_storage import JSONStorageBackend

# 下面这些文字都是 2026-11-08 线上真实出现过的上游回复
NO_TOOL_REPLIES = [
    "目前无法在此对话中直接生成图片，但可以帮你整理成可直接使用的绘图提示词：\n\n竖构图，1024×1536",
    "目前无法在此聊天中直接生成图片，但可以帮你整理成可直接用于图片生成的提示词：",
    "目前无法直接生成图片，但可以帮你完善这段提示词，确保画面更符合真实派出所接待大厅的纪实摄影效果。",
    "抱歉，我目前无法在此对话中直接生成图片，但可以帮你整理提示词。",
    "目前无法直接生成视频或图片，但可以帮你整理成适合 AI 视频生成的完整提示词，保留竖屏 1024×1536 的构图要求。",
    "这张图的关键是表现“煮熟的挂面”。\n\n**优化后的生图提示词：**\n\n竖构图，单个主体完整入镜。",
]
RATE_LIMIT_REPLIES = [
    ("抱歉，当前图片生成服务触发了频率限制，暂时无法生成这张图片。需要等待约 1 小时后才能再次尝试。", 3600),
    ("抱歉，当前图片生成触发了系统频率限制，暂时无法生成这张图片。需要等待约一小时后才能再次尝试。", 3600),
    ("抱歉，当前图片生成服务触发了速率限制，暂时无法生成这张图片，需要等待约 1 小时后才能再次尝试。", 3600),
    ("你已达到 Free 套餐的图像生成请求上限。上限将在 10小时 后重置，届时可创建更多图像。\n\n目前未能调用图像生成工具。", 36000),
]
POLICY_REPLIES = [
    "非常抱歉，生成的图片可能违反了我们的内容政策。如果你认为此判断有误，请重试或修改提示语。",
    "非常抱歉，该提示可能违反了关于暴力内容的防护限制。如果你认为此判断有误，请重试或修改提示语。",
    "非常抱歉，生成的图片可能违反了关于青少年与儿童形象适当描绘的防护限制。如果你认为此判断有误，请重试或修改提示语。",
]


class ClassifyImageTextReplyTests(unittest.TestCase):
    def test_no_image_tool_replies(self) -> None:
        for text in NO_TOOL_REPLIES:
            with self.subTest(text=text[:20]):
                self.assertEqual(classify_image_text_reply(text), IMAGE_TEXT_REPLY_NO_IMAGE_TOOL)
                self.assertEqual(image_text_reply_cooldown_secs(IMAGE_TEXT_REPLY_NO_IMAGE_TOOL, text), IMAGE_NO_TOOL_COOLDOWN_SECS)

    def test_rate_limit_replies_and_cooldown(self) -> None:
        for text, seconds in RATE_LIMIT_REPLIES:
            with self.subTest(text=text[:20]):
                self.assertEqual(classify_image_text_reply(text), IMAGE_TEXT_REPLY_RATE_LIMITED)
                self.assertEqual(image_text_reply_cooldown_secs(IMAGE_TEXT_REPLY_RATE_LIMITED, text), seconds)

    def test_policy_and_unknown_replies_are_not_switched(self) -> None:
        for text in [*POLICY_REPLIES, "", "Here is the image you asked for."]:
            with self.subTest(text=text[:20]):
                self.assertEqual(classify_image_text_reply(text), "")

    def test_rate_limit_cooldown_is_capped(self) -> None:
        self.assertEqual(image_text_reply_cooldown_secs(IMAGE_TEXT_REPLY_RATE_LIMITED, "频率限制，请 100 小时后再试"), 24 * 3600)
        self.assertEqual(image_text_reply_cooldown_secs(IMAGE_TEXT_REPLY_RATE_LIMITED, "rate limit exceeded"), 3600)


class FakeBackend:
    def __init__(self, access_token: str = "") -> None:
        self.access_token = access_token
        self.progress_callback = None

    def close(self) -> None:
        pass


class FakeAccountService:
    def __init__(self) -> None:
        self.issued: list[str] = []
        self.cooldowns: list[tuple[str, int, str]] = []
        self.results: list[tuple[str, bool]] = []

    def get_available_access_token(self, **_: object) -> str:
        token = f"token-{len(self.issued) + 1}"
        self.issued.append(token)
        return token

    def get_account(self, token: str) -> dict:
        return {"email": f"{token}@example.com"}

    def mark_image_result(self, token: str, success: bool) -> None:
        self.results.append((token, success))

    def cool_down_image_account(self, token: str, seconds: float, reason: str = "") -> None:
        self.cooldowns.append((token, int(seconds), reason))


class SwitchAccountLoopTests(unittest.TestCase):
    def _run(self, replies: list[str | None], *, message_as_error: bool = True):
        """replies 里每一项是一个号的回复：字符串 = 回文字，None = 出图成功。"""
        accounts = FakeAccountService()
        calls: list[int] = []

        def fake_stream(_backend, request, index, total):
            reply = replies[min(len(calls), len(replies) - 1)]
            calls.append(1)
            if reply is None:
                yield ImageOutput(kind="result", model=request.model, index=index, total=total, data=[{"b64_json": "x"}])
            else:
                yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=reply, conversation_id="conv")

        request = ConversationRequest(model="gpt-image-2", prompt="a cat", message_as_error=message_as_error)
        with patch.object(conversation, "account_service", accounts), \
                patch.object(conversation, "OpenAIBackendAPI", FakeBackend), \
                patch.object(conversation, "stream_image_outputs", fake_stream), \
                patch.object(conversation, "_remove_image_conversation_later", lambda *a, **k: None):
            try:
                result = _generate_single_image(request, 0, 1)
            except ImageGenerationError as exc:
                return accounts, len(calls), exc
        return accounts, len(calls), result

    def test_switches_account_until_image_is_generated(self) -> None:
        accounts, calls, result = self._run([RATE_LIMIT_REPLIES[0][0], NO_TOOL_REPLIES[0], None])
        self.assertEqual(calls, 3)
        self.assertEqual(accounts.issued, ["token-1", "token-2", "token-3"])
        self.assertEqual([item.kind for item in result], ["result"])
        self.assertEqual(
            accounts.cooldowns,
            [
                ("token-1", 3600, IMAGE_TEXT_REPLY_RATE_LIMITED),
                ("token-2", IMAGE_NO_TOOL_COOLDOWN_SECS, IMAGE_TEXT_REPLY_NO_IMAGE_TOOL),
            ],
        )
        self.assertEqual(accounts.results, [("token-1", False), ("token-2", False), ("token-3", True)])

    def test_gives_up_after_max_switches_with_original_error(self) -> None:
        accounts, calls, exc = self._run([NO_TOOL_REPLIES[1]])
        self.assertEqual(calls, 4)
        self.assertIsInstance(exc, ImageGenerationError)
        self.assertEqual(exc.status_code, 400)
        self.assertEqual(exc.code, "content_policy_violation")
        self.assertEqual(str(exc), NO_TOOL_REPLIES[1])
        self.assertEqual(len(accounts.cooldowns), 4)

    def test_gives_up_returns_message_when_not_error_mode(self) -> None:
        _, calls, result = self._run([RATE_LIMIT_REPLIES[1][0]], message_as_error=False)
        self.assertEqual(calls, 4)
        self.assertEqual([(item.kind, item.text) for item in result], [("message", RATE_LIMIT_REPLIES[1][0])])

    def test_policy_refusal_is_not_retried(self) -> None:
        accounts, calls, exc = self._run([POLICY_REPLIES[0], None])
        self.assertEqual(calls, 1)
        self.assertIsInstance(exc, ImageGenerationError)
        self.assertEqual(str(exc), POLICY_REPLIES[0])
        self.assertEqual(accounts.cooldowns, [])


class AccountImageCooldownTests(unittest.TestCase):
    def test_cooled_down_account_is_skipped_until_expired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items(
                [
                    {"access_token": "token-a", "type": "free", "status": "正常", "quota": 5},
                    {"access_token": "token-b", "type": "free", "status": "正常", "quota": 5},
                ]
            )
            now = 1_000_000.0
            with patch("services.account_service.time.time", return_value=now):
                service.cool_down_image_account("token-a", 600, reason="rate_limited")
                self.assertEqual(service._list_ready_candidate_tokens(), ["token-b"])
                self.assertAlmostEqual(service.image_cooldown_remaining("token-a"), 600)
            with patch("services.account_service.time.time", return_value=now + 601):
                self.assertEqual(sorted(service._list_ready_candidate_tokens()), ["token-a", "token-b"])
            # 冷却只在内存里，不写进账号文件
            self.assertNotIn("cooldown", (Path(tmp_dir) / "accounts.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
