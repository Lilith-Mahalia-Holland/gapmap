from __future__ import annotations


import pymupdf
from typing import Tuple


def scale_bbox(
    src: pymupdf.Page | pymupdf.Pixmap,
    dst: pymupdf.Page | pymupdf.Pixmap,
    bbox: Tuple[float, float, float, float]
) -> pymupdf.Rect:

    src_width = src.rect.width if isinstance(src, pymupdf.Page) else src.width
    src_height = src.rect.height if isinstance(src, pymupdf.Page) else src.height

    dst_width = dst.rect.width if isinstance(dst, pymupdf.Page) else dst.width
    dst_height = dst.rect.height if isinstance(dst, pymupdf.Page) else dst.height

    scaled_width = dst_width / src_width
    scaled_height = dst_height / src_height

    x0, y0, x1, y1 = bbox

    return pymupdf.Rect(
        x0 * scaled_width,
        y0 * scaled_height,
        x1 * scaled_width,
        y1 * scaled_height,
    )