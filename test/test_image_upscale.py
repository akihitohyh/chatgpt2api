from __future__ import annotations

import base64
import io
import unittest
from unittest import mock

from PIL import Image

from services.protocol import conversation
from utils.image_upscale import requested_box, target_size, upscale_image_bytes


def image_bytes(width: int, height: int, image_format: str = "PNG", mode: str = "RGB") -> bytes:
    color = (40, 120, 200, 128) if mode == "RGBA" else (40, 120, 200)
    buffer = io.BytesIO()
    Image.new(mode, (width, height), color).save(buffer, format=image_format)
    return buffer.getvalue()


def open_image(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


class RequestedBoxTests(unittest.TestCase):
    def test_tiers_and_explicit_sizes(self):
        self.assertEqual(requested_box("2K"), (2048, 2048))
        self.assertEqual(requested_box("4k"), (3840, 3840))
        self.assertEqual(requested_box("2560x1440"), (2560, 1440))
        self.assertEqual(requested_box("2160X3840"), (2160, 3840))

    def test_ignores_small_or_invalid_sizes(self):
        for size in (None, "", "auto", "1024x1024", "1920x1088", "0x4096", "abc", "1:1"):
            self.assertIsNone(requested_box(size), size)

    def test_caps_oversized_requests(self):
        self.assertEqual(requested_box("8192x4096"), (4096, 2048))


class TargetSizeTests(unittest.TestCase):
    def test_close_aspect_reaches_exact_size(self):
        self.assertEqual(target_size(1920, 1088, (3840, 2160)), (3840, 2160))

    def test_different_aspect_is_not_stretched(self):
        self.assertEqual(target_size(1024, 1024, (3840, 2160)), (2160, 2160))
        self.assertEqual(target_size(1024, 1536, (3840, 3840)), (2560, 3840))

    def test_never_shrinks(self):
        self.assertIsNone(target_size(2048, 2048, (2048, 2048)))
        self.assertIsNone(target_size(4096, 4096, (2048, 2048)))


class UpscaleImageBytesTests(unittest.TestCase):
    def test_keeps_static_image_format(self):
        for image_format in ("PNG", "JPEG", "WEBP"):
            result = open_image(upscale_image_bytes(image_bytes(1024, 1024, image_format), "2048x2048"))
            self.assertEqual(result.format, image_format)
            self.assertEqual(result.size, (2048, 2048))

    def test_keeps_alpha_channel(self):
        result = open_image(upscale_image_bytes(image_bytes(512, 512, mode="RGBA"), "2K"))
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.size, (2048, 2048))
        self.assertLessEqual(abs(result.getpixel((1024, 1024))[3] - 128), 1)

    def test_gif_is_untouched(self):
        data = image_bytes(256, 256, "GIF")
        self.assertIs(upscale_image_bytes(data, "4K"), data)

    def test_animated_webp_is_untouched(self):
        frames = [Image.new("RGB", (64, 64), color) for color in ("red", "blue")]
        buffer = io.BytesIO()
        frames[0].save(buffer, format="WEBP", save_all=True, append_images=frames[1:])
        data = buffer.getvalue()
        self.assertIs(upscale_image_bytes(data, "4K"), data)

    def test_small_size_is_untouched(self):
        data = image_bytes(512, 512)
        self.assertIs(upscale_image_bytes(data, "1024x1024"), data)

    def test_broken_image_returns_original(self):
        data = b"\x89PNG\r\n\x1a\nbroken"
        self.assertIs(upscale_image_bytes(data, "4K"), data)


class FormatImageResultTests(unittest.TestCase):
    def test_returns_and_stores_upscaled_image(self):
        source = base64.b64encode(image_bytes(1024, 1024)).decode("ascii")
        with mock.patch.object(conversation, "save_image_bytes", return_value="/saved.png") as save:
            result = conversation.format_image_result(
                [{"b64_json": source}], "prompt", "b64_json", size="2048x2048",
            )
        item = result["data"][0]
        self.assertEqual(open_image(base64.b64decode(item["b64_json"])).size, (2048, 2048))
        self.assertEqual(open_image(save.call_args.args[0]).size, (2048, 2048))

    def test_without_size_keeps_original_base64(self):
        source = base64.b64encode(image_bytes(64, 64)).decode("ascii")
        with mock.patch.object(conversation, "save_image_bytes", return_value="/saved.png"):
            result = conversation.format_image_result([{"b64_json": source}], "prompt", "b64_json")
        self.assertEqual(result["data"][0]["b64_json"], source)


if __name__ == "__main__":
    unittest.main()
