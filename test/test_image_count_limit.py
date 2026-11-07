from __future__ import annotations

import os
import unittest

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "chatgpt2api")

from fastapi import HTTPException
from pydantic import ValidationError

from api.ai import ImageGenerationRequest
from api.image_inputs import _parse_count
from utils.helper import MAX_IMAGE_COUNT, parse_image_count


class ImageCountLimitTests(unittest.TestCase):
    def test_limit_is_ten(self):
        self.assertEqual(MAX_IMAGE_COUNT, 10)

    def test_generation_request_accepts_up_to_limit(self):
        self.assertEqual(ImageGenerationRequest(prompt="cat", n=10).n, 10)
        for value in (0, 11):
            with self.assertRaises(ValidationError):
                ImageGenerationRequest(prompt="cat", n=value)

    def test_edit_and_chat_counts_accept_up_to_limit(self):
        for parse in (_parse_count, parse_image_count):
            self.assertEqual(parse(10), 10)
            self.assertEqual(parse(None), 1)
            with self.assertRaises(HTTPException) as caught:
                parse(11)
            self.assertEqual(caught.exception.status_code, 400)
            self.assertEqual(caught.exception.detail, {"error": "n must be between 1 and 10"})


if __name__ == "__main__":
    unittest.main()
