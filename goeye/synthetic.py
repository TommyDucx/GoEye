"""Render synthetic Go boards for testing the recogniser headlessly.

The boards deliberately include the same visual features a screenshot has:
wood-coloured background, thin brown grid lines, star points, and stones with
a soft highlight.  A "last move" marker dot is optionally painted on one stone
to make sure the median-sampling step ignores it.
"""

from __future__ import annotations

import numpy as np

from .vision import BLACK, EMPTY, WHITE

WOOD = (196, 158, 102)  # BGR
LINE = (40, 30, 22)  # BGR
BLACK_STONE = (12, 12, 12)
WHITE_STONE = (228, 226, 219)


def render_board(
    stones: np.ndarray,
    size: int = 19,
    canvas: int = 620,
    margin_fraction: float = 0.06,
    last_move: tuple[int, int] | None = None,
    jitter: float = 0.0,
) -> np.ndarray:
    """Return a BGR image of a board with the given stones.

    ``stones`` is (size, size) with EMPTY/BLACK/WHITE.  ``jitter`` shifts the
    whole grid by a few pixels so the fitter is tested against imperfect input.
    """
    img = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    img[:] = WOOD

    span = canvas * (1 - 2 * margin_fraction)
    x0 = margin_fraction * canvas + jitter
    y0 = margin_fraction * canvas + jitter
    step = span / (size - 1)

    def px(col: int) -> int:
        return int(round(x0 + col * step))

    def py(row: int) -> int:
        return int(round(y0 + row * step))

    # Grid lines
    line_w = max(1, int(round(canvas / 620.0)))
    for i in range(size):
        cv2_line(img, (px(0), py(i)), (px(size - 1), py(i)), LINE, line_w)
        cv2_line(img, (px(i), py(0)), (px(i), py(size - 1)), LINE, line_w)

    # Star points
    stars = [3, 9, 15] if size == 19 else [2, size // 2, size - 3]
    for r in stars:
        for c in stars:
            cv2_dot(img, (px(c), py(r)), max(2, int(step * 0.1)), LINE)

    # Stones
    for row in range(size):
        for col in range(size):
            v = int(stones[row, col])
            if v == EMPTY:
                continue
            center = (px(col), py(row))
            color = BLACK_STONE if v == BLACK else WHITE_STONE
            cv2_dot(img, center, int(step * 0.45), color)
            # Subtle highlight on the upper-left for a 3-D feel.
            hl = tuple(int(c * 0.6) for c in color)
            cv2_dot(
                img,
                (center[0] - int(step * 0.13), center[1] - int(step * 0.13)),
                max(1, int(step * 0.14)),
                hl,
            )

    if last_move is not None:
        r, c = last_move
        cv2_dot(img, (px(c), py(r)), max(2, int(step * 0.08)), (40, 90, 220))

    return img


def cv2_line(img: np.ndarray, p1, p2, color, w: int) -> None:
    import cv2

    cv2.line(img, p1, p2, color, w, cv2.LINE_AA)


def cv2_dot(img: np.ndarray, center, radius: int, color) -> None:
    import cv2

    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)
