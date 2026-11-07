from __future__ import annotations

import re

from utils.log import logger

MIN_UPSCALE_EDGE = 2048
MAX_UPSCALE_EDGE = 4096
# 比例差在这个范围内时直接缩放到请求的精确尺寸，人眼看不出变形。
ASPECT_TOLERANCE = 0.02
SIZE_TIERS = {"2k": 2048, "4k": 3840}

SIZE_PATTERN = re.compile(r"(\d{1,5})\s*[x×*]\s*(\d{1,5})")
SAVE_OPTIONS = {
    "png": (".png", {"compression": 6}),
    "jpeg": (".jpg", {"Q": 95}),
    "webp": (".webp", {"Q": 95}),
}


def requested_box(size: object) -> tuple[int, int] | None:
    """解析 2K/4K 档位的请求尺寸，其他尺寸返回 None。"""
    text = str(size or "").strip().lower()
    if text in SIZE_TIERS:
        edge = SIZE_TIERS[text]
        return edge, edge
    match = SIZE_PATTERN.fullmatch(text)
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0 or max(width, height) < MIN_UPSCALE_EDGE:
        return None
    long_edge = max(width, height)
    if long_edge > MAX_UPSCALE_EDGE:
        scale = MAX_UPSCALE_EDGE / long_edge
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
    return width, height


def target_size(width: int, height: int, box: tuple[int, int]) -> tuple[int, int] | None:
    """计算放大后的尺寸，比例差太大时保持原比例放进请求尺寸内，不需要放大时返回 None。"""
    box_width, box_height = box
    aspect_delta = abs((width / height) / (box_width / box_height) - 1)
    if aspect_delta <= ASPECT_TOLERANCE:
        target = box_width, box_height
    else:
        scale = min(box_width / width, box_height / height)
        target = round(width * scale), round(height * scale)
    if target[0] < width or target[1] < height or target == (width, height):
        return None
    return target


def image_format(data: bytes) -> str | None:
    """只识别可放大的静态图片格式，GIF 和其他内容返回 None。"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def upscale_image_bytes(data: bytes, size: object) -> bytes:
    """按请求的 2K/4K 尺寸放大图片；不需要放大或处理失败时返回原图。"""
    box = requested_box(size)
    file_format = image_format(data)
    if box is None or file_format is None:
        return data
    try:
        import pyvips

        image = pyvips.Image.new_from_buffer(data, "", access="sequential")
        if image.get_typeof("n-pages") and image.get("n-pages") > 1:
            return data
        target = target_size(image.width, image.height, box)
        if target is None:
            return data
        hscale, vscale = target[0] / image.width, target[1] / image.height
        if image.hasalpha():
            resized = image.premultiply().resize(hscale, vscale=vscale, kernel="lanczos3")
            resized = resized.unpremultiply().cast(image.format)
        else:
            resized = image.resize(hscale, vscale=vscale, kernel="lanczos3")
        suffix, options = SAVE_OPTIONS[file_format]
        return resized.write_to_buffer(suffix, **options)
    except Exception as exc:
        logger.warning({"event": "image_upscale_failed", "size": str(size), "error": str(exc)})
        return data
