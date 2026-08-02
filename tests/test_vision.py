"""Headless accuracy check for the recogniser.

Run:  python tests/test_vision.py
Needs numpy + opencv-python-headless (no display / mss / PyQt required).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goeye.synthetic import render_board  # noqa: E402
from goeye.vision import (  # noqa: E402
    BLACK,
    EMPTY,
    WHITE,
    fit_grid,
    infer_board_size,
    read_board,
)


def random_board(size: int = 19, fill: float = 0.35, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    board = np.full((size, size), EMPTY, dtype=np.uint8)
    mask = rng.random((size, size)) < fill
    picks = rng.random((size, size))
    board[mask & (picks < 0.5)] = BLACK
    board[mask & (picks >= 0.5)] = WHITE
    return board


def evaluate(seed: int, jitter: float, last_move) -> dict:
    truth = random_board(seed=seed)
    img = render_board(truth, last_move=last_move, jitter=jitter)

    grid = fit_grid(img, 19)
    reading = read_board(img, grid)
    pred = reading.stones

    total = truth.size
    correct = int((pred == truth).sum())
    # Per-class accuracy
    per_class = {}
    for name, val in (("black", BLACK), ("white", WHITE), ("empty", EMPTY)):
        t = truth == val
        if t.sum() == 0:
            per_class[name] = 1.0
        else:
            per_class[name] = float((pred[t] == val).sum() / t.sum())
    return {
        "seed": seed,
        "conf": grid.confidence,
        "acc": correct / total,
        "per_class": per_class,
        "ambiguity": reading.ambiguity,
    }


def test_auto_size(size: int, seed: int = 7) -> bool:
    """Render a board of the given size and check auto-detection picks it."""
    truth = random_board(size=size, fill=0.30, seed=seed)
    img = render_board(truth, size=size)
    detected, conf = infer_board_size(img)
    grid = fit_grid(img, 0)
    reading = read_board(img, grid)
    ok = detected == size and grid.size == size and reading.size == size
    print(
        f"auto-size {size}x{size}: detected={detected} conf={conf:.2f} "
        f"grid_size={grid.size} reading_size={reading.size} {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main() -> int:
    # Include a board with a last-move marker to exercise the median filter.
    cases = [
        (0, 0.0, None),
        (1, 1.5, (9, 9)),
        (2, -2.0, (3, 15)),
        (3, 3.0, (15, 3)),
        (4, 0.0, None),
    ]
    print(f"{'seed':>4} {'conf':>6} {'acc':>6}  {'black':>6} {'white':>6} {'empty':>6} {'ambig':>6}")
    worst = 1.0
    for seed, jitter, last in cases:
        r = evaluate(seed, jitter, last)
        worst = min(worst, r["acc"])
        print(
            f"{r['seed']:>4} {r['conf']:>6.2f} {r['acc']:>6.3f}  "
            f"{r['per_class']['black']:>6.3f} {r['per_class']['white']:>6.3f} "
            f"{r['per_class']['empty']:>6.3f} {r['ambiguity']:>6.3f}"
        )
    ok = worst >= 0.995
    print("\nRESULT:", "PASS" if ok else f"FAIL (worst acc {worst:.3f})")
    if not ok:
        return 1

    print("\nAuto board-size detection:")
    size_ok = True
    for s in (9, 13, 19):
        size_ok &= test_auto_size(s)
    print("RESULT:", "PASS" if size_ok else "FAIL")
    return 0 if size_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
