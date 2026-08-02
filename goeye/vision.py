"""Screen capture and Go board recognition.

The pipeline is deliberately split into three independent stages so that each
one can be tested (and debugged) on its own:

    1. ``ScreenGrabber``  -- pull raw pixels for a screen rectangle.
    2. ``fit_grid``       -- locate the 19x19 line grid inside those pixels.
    3. ``read_board``     -- classify every intersection as black/white/empty.

Stage 2 is the fiddly one.  Rather than trusting the user to frame the board
perfectly, we let them drag a rough box and then snap the grid to the real
lines using a brute-force fit over (offset, spacing).  That is far more robust
than Hough transforms on boards where stones cover most of the lines.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import cv2
import numpy as np

EMPTY, BLACK, WHITE = 0, 1, 2
STONE_CHARS = {EMPTY: ".", BLACK: "X", WHITE: "O"}

# GTP column letters: 'I' is skipped by convention.
GTP_COLS = "ABCDEFGHJKLMNOPQRST"


def rc_to_gtp(row: int, col: int, size: int = 19) -> str:
    """(row 0 = top) -> GTP coordinate such as ``Q16``."""
    return f"{GTP_COLS[col]}{size - row}"


def gtp_to_rc(coord: str, size: int = 19) -> Optional[tuple[int, int]]:
    """Inverse of :func:`rc_to_gtp`.  Returns ``None`` for ``pass``."""
    coord = coord.strip().upper()
    if coord in ("PASS", "RESIGN", ""):
        return None
    col = GTP_COLS.index(coord[0])
    row = size - int(coord[1:])
    return row, col


# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------


class ScreenGrabber:
    """Thin wrapper over ``mss`` that is safe to build lazily per thread.

    ``mss`` objects are not thread-safe on macOS, so we create the handle on
    first use inside whichever thread ends up owning the grabber.
    """

    def __init__(self) -> None:
        self._sct = None

    def grab(self, region: dict) -> np.ndarray:
        """Capture ``region`` (keys: left/top/width/height) as a BGR array.

        On Retina displays the returned array is larger than the requested
        logical rectangle.  That is fine: every downstream coordinate is stored
        as a *fraction* of the region, so scale changes are absorbed.
        """
        import mss  # imported lazily; keeps module importable without a display

        if self._sct is None:
            self._sct = mss.mss()
        raw = self._sct.grab(region)
        frame = np.asarray(raw)  # BGRA, physical pixels
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None


# ---------------------------------------------------------------------------
# Grid fitting
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GridSpec:
    """Where the playing grid sits inside the captured crop.

    Coordinates are stored as fractions in ``[0, 1]`` of the crop's width and
    height so that the same spec survives a change of display scale.
    """

    fx0: float = 0.03
    fy0: float = 0.03
    fx1: float = 0.97
    fy1: float = 0.97
    size: int = 19
    confidence: float = 0.0
    visible_lines_h: int = 0  # automatically detected visible rows in the crop
    visible_lines_v: int = 0  # automatically detected visible columns in the crop

    def pixels(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the x and y pixel coordinates of every grid line."""
        xs = np.linspace(self.fx0 * width, self.fx1 * width, self.size)
        ys = np.linspace(self.fy0 * height, self.fy1 * height, self.size)
        return xs, ys

    def cell_px(self, width: int, height: int) -> float:
        span_x = (self.fx1 - self.fx0) * width
        span_y = (self.fy1 - self.fy0) * height
        return min(span_x, span_y) / max(self.size - 1, 1)


def _line_projections(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (row_signal, col_signal) emphasising the grid lines.

    We threshold the (inverted) image so dark grid lines become bright, then
    *sum* the bright pixels along each row/column.  A real grid line is a long
    run, so its row gets a large sum; a stone is only a ~13px bump, so it
    contributes far less and cannot fake a line.  We deliberately do **not**
    morphologically open the image: an opening wide enough to strip stones
    also erases lines that have gaps (every line sits under stones), which is
    exactly what breaks detection on a crowded board.
    """
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 8
    )
    # A short *closing* seals the small gaps that white stones punch in a line
    # without lengthening anything, so a line reads as one continuous run.
    close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close)

    row_signal = binary.sum(axis=1).astype(np.float64) / max(width * 255, 1)
    col_signal = binary.sum(axis=0).astype(np.float64) / max(height * 255, 1)
    return row_signal, col_signal


def _fit_evenly_spaced(signal: np.ndarray, size: int) -> tuple[float, float, float]:
    """Brute-force the best ``size`` evenly spaced peaks in a 1-D signal.

    Returns ``(score, first_position, spacing)``.  Scoring on the *mean* of all
    ``size`` sampled values means a handful of lines hidden under stones cannot
    derail the fit, while a wrong spacing immediately tanks the score.
    """
    length = len(signal)
    sig = signal.astype(np.float64)
    if sig.max() > sig.min():
        sig = (sig - sig.min()) / (sig.max() - sig.min())
    else:
        return 0.0, 0.0, max(length - 1, 1) / max(size - 1, 1)

    # Sub-pixel sampling via linear interpolation.
    offsets = np.arange(size)
    best = (-1.0, 0.0, 1.0)

    min_spacing = max(3.0, length / (size + 10))
    max_spacing = max(min_spacing + 0.25, (length - 1) / (size - 1))

    for spacing in np.arange(min_spacing, max_spacing + 1e-9, 0.25):
        span = spacing * (size - 1)
        max_start = length - 1 - span
        if max_start < 0:
            continue
        starts = np.arange(0.0, max_start + 1e-9, 0.5)
        if starts.size == 0:
            continue
        positions = starts[:, None] + spacing * offsets[None, :]
        low = np.floor(positions).astype(np.int64)
        frac = positions - low
        low = np.clip(low, 0, length - 1)
        high = np.clip(low + 1, 0, length - 1)
        values = sig[low] * (1.0 - frac) + sig[high] * frac
        # Score by how well BOTH board edges land on a line, plus the overall
        # mean.  Using the mean alone is ambiguous: a grid shifted by exactly
        # one cell spacing scores almost identically because 18 of 19 sampled
        # points still sit on real lines.  Anchoring to the two edges breaks
        # that tie, since any shifted grid leaves a valley at one board edge.
        edge = np.minimum(values[:, 0], values[:, -1])
        scores = edge * 0.6 + values.mean(axis=1) * 0.4
        best_idx = int(np.argmax(scores))
        if scores[best_idx] > best[0]:
            best = (float(scores[best_idx]), float(starts[best_idx]), float(spacing))
    return best


def _count_lines(signal: np.ndarray) -> int:
    """Estimate how many grid lines are actually visible in a 1-D projection.

    This is used to seed the automatic board-size detection.  We smooth the
    signal, threshold it, and count peaks that are separated by at least a
    few pixels (so a single wide line does not count twice).
    """
    sig = signal.astype(np.float64)
    if sig.max() > sig.min():
        sig = (sig - sig.min()) / (sig.max() - sig.min())
    else:
        return 0

    # Gentle smoothing to remove single-pixel noise without merging lines.
    window = max(3, int(len(sig) / 80)) | 1  # odd
    if window > 3:
        sig = cv2.GaussianBlur(sig.reshape(1, -1), (window, 1), 0).ravel()

    threshold = 0.35
    above = sig > threshold
    if not above.any():
        return 0

    # Find contiguous runs above threshold; each run is one line candidate.
    runs = []
    i = 0
    while i < len(above):
        if above[i]:
            start = i
            while i < len(above) and above[i]:
                i += 1
            peak = int(np.argmax(sig[start:i]) + start)
            runs.append(peak)
        i += 1

    if not runs:
        return 0

    # Merge peaks that are too close (less than ~3 px) to avoid double counts.
    merged = [runs[0]]
    for p in runs[1:]:
        if p - merged[-1] > 3:
            merged.append(p)
    return len(merged)


def _find_line_peaks(signal: np.ndarray) -> list[int]:
    """Return the pixel positions of the grid-line peaks in a 1-D projection.

    Threshold the (normalised) signal, then treat each contiguous run above
    threshold as a single line and take its centre.  Adjacent runs that are
    only a few pixels apart (a single line read as two by antialiasing / a
    closing artifact) are merged below.
    """
    sig = signal.astype(np.float64)
    if sig.max() <= sig.min():
        return []
    sig = (sig - sig.min()) / (sig.max() - sig.min())

    above = sig > 0.35
    if not above.any():
        return []

    runs: list[int] = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            start = i
            while i < n and above[i]:
                i += 1
            runs.append(int(np.argmax(sig[start:i]) + start))
        else:
            i += 1

    if len(runs) < 2:
        return runs

    # Merge runs that are closer than ~half the median gap: those are almost
    # certainly the same physical line.
    gaps = np.diff(runs)
    median_gap = float(np.median(gaps))
    if median_gap <= 0:
        return runs
    merged = [float(runs[0])]
    for p in runs[1:]:
        if p - merged[-1] < 0.5 * median_gap:
            merged[-1] = (merged[-1] + p) / 2.0
        else:
            merged.append(float(p))
    return [int(round(m)) for m in merged]


def _line_count(signal: np.ndarray) -> tuple[int, float]:
    """Estimate the number of grid lines in a 1-D projection.

    Returns ``(count, regularity)``.  Instead of naively counting peaks (which
    is fooled by the occasional spurious line a thresholding step leaves
    between two real ones), we fit a set of ``n`` evenly spaced lines to the
    detected peak positions for every plausible ``n`` and keep the one whose
    fit is tightest.  The fit is *symmetric* -- we penalise both peaks that
    fall far from a line AND lines that have no nearby peak -- which rejects
    the degenerate "half-cell" solution that a one-sided residual would accept.
    """
    peaks = _find_line_peaks(signal)
    if len(peaks) < 3:
        reg = 1.0 if len(peaks) in (1, 2) else 0.0
        return len(peaks), reg
    peaks_arr = np.asarray(peaks, dtype=np.float64)
    span = peaks_arr[-1] - peaks_arr[0]
    if span <= 0:
        return len(peaks), 0.0

    best_n, best_score, best_gap = 2, float("inf"), 0.0
    # Span only makes sense for n in [2, ~40]; a cell smaller than 3px is not a
    # real board line.
    max_n = min(40, int(span / 3) + 2)
    for n in range(2, max_n + 1):
        gap = span / (n - 1)
        if gap < 3.0:
            continue
        fitted = peaks_arr[0] + gap * np.arange(n)
        # Each peak -> distance to its nearest fitted line.
        d_peak = np.min(np.abs(peaks_arr[:, None] - fitted[None, :]), axis=1)
        # Each fitted line -> distance to its nearest peak.
        d_fit = np.min(np.abs(fitted[:, None] - peaks_arr[None, :]), axis=1)
        score = float(d_peak.mean() + d_fit.mean())
        if score < best_score:
            best_score = score
            best_n = n
            best_gap = gap

    regularity = float(np.clip(1.0 - best_score / max(best_gap, 1e-6), 0.0, 1.0))
    return best_n, regularity


_STANDARD_SIZES = (9, 13, 19)


def _snap_to_standard(n: int) -> int:
    """Snap a detected line count to the nearest canonical board size.

    Real boards are almost always 9/13/19; a detected count near one of those
    (e.g. 18, 20 from a cropped edge) is rounded to it.  Truly degenerate
    counts (< 5 or > 40) are left untouched.
    """
    if n < 5 or n > 40:
        return n
    return min(_STANDARD_SIZES, key=lambda s: abs(s - n))


def infer_board_size(
    image: np.ndarray,
    candidates: Optional[Sequence[int]] = None,
    min_confidence: float = 0.12,
    default: int = 19,
) -> tuple[int, float]:
    """Detect the board size (rows == columns) from the grid lines in ``image``.

    Returns ``(size, confidence)``.  The size is derived from the *real* line
    spacing measured in the row and column projections, not by comparing fits:
    a finer grid always contains a coarser one as a subset, so any "fit the
    fewest lines" heuristic is fooled into preferring a small board.  We
    instead find every line, measure the median gap, and divide the span by it
    to recover the true line count, then snap to the nearest canonical size.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row_signal, col_signal = _line_projections(gray)

    count_h, reg_h = _line_count(row_signal)
    count_v, reg_v = _line_count(col_signal)

    snapped_h = _snap_to_standard(count_h)
    snapped_v = _snap_to_standard(count_v)

    # Prefer the dimension that produced a canonical size; if both did, they
    # should agree, and we trust the result.  If they disagree (noise), keep
    # the one with the more regular (more reliable) spacing.
    if snapped_h in _STANDARD_SIZES or snapped_v in _STANDARD_SIZES:
        if snapped_h in _STANDARD_SIZES and snapped_v in _STANDARD_SIZES:
            size = snapped_h if snapped_h == snapped_v else max(snapped_h, snapped_v)
            confidence = 0.85
        else:
            if snapped_h in _STANDARD_SIZES:
                size, confidence = snapped_h, 0.7 * (0.5 + 0.5 * reg_h)
            else:
                size, confidence = snapped_v, 0.7 * (0.5 + 0.5 * reg_v)
    else:
        # Neither dimension is near a standard size: trust the larger count
        # (the more lines we actually see, the more likely it is the true
        # board rather than a cropped view) and report low confidence.
        size = max(count_h, count_v)
        confidence = 0.5 * (reg_h + reg_v)

    if confidence < min_confidence:
        return default, confidence
    return size, confidence


def fit_grid(image: np.ndarray, size: int = 0) -> GridSpec:
    """Locate the grid inside a crop that roughly contains a board.

    If ``size`` is 0 the board size is auto-detected from the visible grid
    lines (common sizes 9/13/19, plus the actual visible line count for
    cropped views); otherwise the requested size is enforced.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    row_signal, col_signal = _line_projections(gray)

    visible_h = _count_lines(row_signal)
    visible_v = _count_lines(col_signal)

    if size <= 0:
        detected, detect_conf = infer_board_size(
            image, min_confidence=0.12, default=19
        )
        size = detected
    else:
        detect_conf = 0.0

    row_score, y0, y_spacing = _fit_evenly_spaced(row_signal, size)
    col_score, x0, x_spacing = _fit_evenly_spaced(col_signal, size)

    y1 = y0 + y_spacing * (size - 1)
    x1 = x0 + x_spacing * (size - 1)

    spec = GridSpec(
        fx0=x0 / max(width - 1, 1),
        fy0=y0 / max(height - 1, 1),
        fx1=x1 / max(width - 1, 1),
        fy1=y1 / max(height - 1, 1),
        size=size,
        confidence=float(min(row_score, col_score)),
        visible_lines_h=visible_h,
        visible_lines_v=visible_v,
    )

    # A believable board is roughly square.  If the fit produced a wildly
    # non-square grid it almost certainly latched onto unrelated UI lines, so
    # fall back to insetting the user's box instead of returning nonsense.
    span_x = (spec.fx1 - spec.fx0) * width
    span_y = (spec.fy1 - spec.fy0) * height
    if span_x <= 0 or span_y <= 0:
        return GridSpec(
            size=size,
            confidence=0.0,
            visible_lines_h=visible_h,
            visible_lines_v=visible_v,
        )
    aspect = span_x / span_y
    if not 0.75 <= aspect <= 1.33 or spec.confidence < 0.12:
        return GridSpec(
            size=size,
            confidence=max(spec.confidence * 0.5, 0.0),
            visible_lines_h=visible_h,
            visible_lines_v=visible_v,
        )
    return spec


# ---------------------------------------------------------------------------
# Stone classification
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ColorRefs:
    """Reference HSV (value, saturation) points used to classify a sample."""

    black: Optional[tuple[float, float]] = None
    white: Optional[tuple[float, float]] = None
    empty: Optional[tuple[float, float]] = None

    def complete(self) -> bool:
        return None not in (self.black, self.white, self.empty)


@dataclasses.dataclass
class BoardReading:
    stones: np.ndarray  # (size, size) uint8 of EMPTY / BLACK / WHITE
    background: tuple[float, float]
    ambiguity: float  # 0 = crisp, 1 = every point was a coin flip
    size: int = 19

    def __post_init__(self) -> None:
        if self.size <= 0:
            self.size = int(self.stones.shape[0])

    def counts(self) -> tuple[int, int]:
        return int((self.stones == BLACK).sum()), int((self.stones == WHITE).sum())

    def to_text(self) -> str:
        return "\n".join(
            " ".join(STONE_CHARS[int(v)] for v in row) for row in self.stones
        )


def _disk_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    span = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(span, span, indexing="ij")
    keep = (dx * dx + dy * dy) <= radius * radius
    return dy[keep], dx[keep]


def _sample_medians(
    channels: Sequence[np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    radius: int,
) -> list[np.ndarray]:
    """Median of each channel over a disk around every (x, y) pair.

    The *median* (rather than the mean) makes the reading immune to the small
    "last move" dot that most Go clients paint on top of the latest stone.
    """
    height, width = channels[0].shape
    dy, dx = _disk_offsets(max(radius, 1))

    cx = np.clip(np.round(xs).astype(np.int64), 0, width - 1)
    cy = np.clip(np.round(ys).astype(np.int64), 0, height - 1)

    sample_y = np.clip(cy[:, None] + dy[None, :], 0, height - 1)
    sample_x = np.clip(cx[:, None] + dx[None, :], 0, width - 1)

    out = []
    for channel in channels:
        patch = channel[sample_y, sample_x]
        out.append(np.median(patch, axis=1))
    return out


def read_board(
    image: np.ndarray,
    grid: GridSpec,
    refs: Optional[ColorRefs] = None,
) -> BoardReading:
    """Classify every intersection of ``grid`` inside ``image``."""
    size = grid.size
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)

    xs, ys = grid.pixels(width, height)
    cell = grid.cell_px(width, height)
    stone_radius = max(1, int(round(cell * 0.30)))

    grid_x, grid_y = np.meshgrid(xs, ys)  # (size, size)
    flat_x, flat_y = grid_x.ravel(), grid_y.ravel()
    v_med, s_med = _sample_medians([value, saturation], flat_x, flat_y, stone_radius)

    # Background reference: the centre of each cell is never covered by a
    # stone (a stone's radius is < 0.5 cell, the centre is 0.707 cells away).
    mid_x = (xs[:-1] + xs[1:]) / 2.0
    mid_y = (ys[:-1] + ys[1:]) / 2.0
    bg_gx, bg_gy = np.meshgrid(mid_x, mid_y)
    bg_v, bg_s = _sample_medians(
        [value, saturation],
        bg_gx.ravel(),
        bg_gy.ravel(),
        max(1, int(round(cell * 0.18))),
    )
    background = (float(np.median(bg_v)), float(np.median(bg_s)))

    if refs is not None and refs.complete():
        stones, ambiguity = _classify_by_refs(v_med, s_med, refs)
    else:
        stones, ambiguity = _auto_classify(v_med, s_med, background)

    return BoardReading(
        stones=stones.reshape(size, size).astype(np.uint8),
        background=background,
        ambiguity=ambiguity,
        size=size,
    )


def _auto_classify(
    v_med: np.ndarray, s_med: np.ndarray, background: tuple[float, float]
) -> tuple[np.ndarray, float]:
    """Rule-based first; only reach for kMeans when the rules are unsure."""
    adaptive_stones, adaptive_amb = _classify_adaptive(v_med, s_med, background)
    if adaptive_amb < 0.06:
        return adaptive_stones, adaptive_amb
    km_stones, km_amb, km_ok = _classify_kmeans(v_med, s_med)
    if km_ok:
        return km_stones, km_amb
    return adaptive_stones, adaptive_amb


def _classify_adaptive(
    v_med: np.ndarray, s_med: np.ndarray, background: tuple[float, float]
) -> tuple[np.ndarray, float]:
    """Threshold rules derived from the measured board colour.

    Works out of the box for the wood-coloured boards used by essentially every
    Go client.  Very pale or very dark themes should use manual calibration.
    """
    bg_v, bg_s = background
    out = np.full(v_med.shape, EMPTY, dtype=np.int32)

    black_cut = min(bg_v * 0.60, bg_v - 45.0)
    white_v_cut = max(bg_v * 0.92, bg_v - 12.0)
    white_s_cut = max(28.0, bg_s * 0.45)

    is_black = v_med < black_cut
    is_white = (~is_black) & (v_med > white_v_cut) & (s_med < white_s_cut)
    out[is_black] = BLACK
    out[is_white] = WHITE

    # How close was each point to flipping class?  Used purely as a warning.
    black_margin = np.abs(v_med - black_cut) / max(bg_v, 1.0)
    white_margin = np.abs(v_med - white_v_cut) / max(bg_v, 1.0)
    margin = np.minimum(black_margin, white_margin)
    ambiguity = float(np.mean(margin < 0.03))
    return out, ambiguity


def _classify_kmeans(
    v_med: np.ndarray, s_med: np.ndarray
) -> tuple[np.ndarray, float, bool]:
    """Unsupervised 3-cluster split on (value, saturation) — borrowed from the
    Stanford CS231A Go-board reconstruction approach.

    Because it learns the three colour modes (black / empty / white) directly
    from the *current* board, it is far more robust than fixed thresholds on
    unusual boards (dark glass, tinted wood, bright themes).  It is only used
    as a fallback when the rule-based classifier is unsure, and only when the
    clustering actually produced three well-separated modes.
    """
    samples = np.stack([v_med, s_med], axis=1).astype(np.float32)
    if samples.shape[0] == 0:
        return np.full(v_med.shape, EMPTY, dtype=np.int32), 0.0, False

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    _, labels, centers = cv2.kmeans(
        samples, 3, None, criteria, 8, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    order = np.argsort(centers[:, 0])  # ascending value: black < empty < white
    v_centers = centers[order, 0]
    gaps = np.diff(v_centers)
    total = max(v_centers[-1] - v_centers[0], 1.0)
    separation_ok = bool(np.min(gaps) / total > 0.12)

    class_of = {int(order[0]): BLACK, int(order[2]): WHITE, int(order[1]): EMPTY}
    out = np.array([class_of[int(l)] for l in labels], dtype=np.int32)

    # Ambiguity: distance to the 2nd-nearest centre relative to the 1st.
    d = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2)
    ds = np.sort(d, axis=1)
    margin = (ds[:, 1] - ds[:, 0]) / np.maximum(ds[:, 0], 1e-6)
    ambiguity = float(np.mean(margin < 0.10))
    return out, ambiguity, separation_ok


def _classify_by_refs(
    v_med: np.ndarray, s_med: np.ndarray, refs: ColorRefs
) -> tuple[np.ndarray, float]:
    """Nearest-reference classification in a weighted (value, saturation) space."""
    labels = [BLACK, WHITE, EMPTY]
    points = np.array([refs.black, refs.white, refs.empty], dtype=np.float64)

    # Brightness separates stones far more reliably than saturation does.
    weights = np.array([1.0, 0.45])
    samples = np.stack([v_med, s_med], axis=1)
    diff = (samples[:, None, :] - points[None, :, :]) * weights[None, None, :]
    distances = np.linalg.norm(diff, axis=2)

    order = np.argsort(distances, axis=1)
    best = order[:, 0]
    nearest = np.take_along_axis(distances, order[:, :2], axis=1)
    gap = nearest[:, 1] - nearest[:, 0]
    ambiguity = float(np.mean(gap < 12.0))

    out = np.array([labels[i] for i in best], dtype=np.int32)
    return out, ambiguity


# ---------------------------------------------------------------------------
# Turn inference
# ---------------------------------------------------------------------------


def infer_side_to_move(
    stones: np.ndarray, previous: Optional[np.ndarray] = None
) -> str:
    """Guess whose turn it is.

    Diffing against the previous frame is by far the most reliable signal in
    live play: whoever just placed a stone is *not* the one to move.  The stone
    count is only a fallback for the very first frame.
    """
    if previous is not None and previous.shape == stones.shape:
        added_black = int(((stones == BLACK) & (previous != BLACK)).sum())
        added_white = int(((stones == WHITE) & (previous != WHITE)).sum())
        if added_black and not added_white:
            return "W"
        if added_white and not added_black:
            return "B"

    black = int((stones == BLACK).sum())
    white = int((stones == WHITE).sum())
    return "W" if black > white else "B"


def board_signature(stones: np.ndarray) -> bytes:
    return stones.tobytes()


def board_to_sgf(
    stones: np.ndarray,
    size: int = 19,
    komi: float = 7.5,
    to_move: str = "B",
) -> str:
    """Render a recognised position as an SGF file body.

    Stones are emitted as AB/AW placements; the original move order is unknown
    from a single screenshot, so this is a *position* record (good for feeding
    KataGo/Sabaki/other tools), not a replayable game.
    """
    blacks: list[str] = []
    whites: list[str] = []
    for row in range(size):
        for col in range(size):
            v = int(stones[row, col])
            coord = chr(97 + col) + chr(97 + row)  # SGF: a=a=top-left
            if v == BLACK:
                blacks.append(coord)
            elif v == WHITE:
                whites.append(coord)
    body = f"(;GM[1]FF[4]CA[UTF-8]SZ[{size}]KM[{komi}]PL[{to_move}]"
    if blacks:
        body += "AB" + "".join(f"[{c}]" for c in blacks)
    if whites:
        body += "AW" + "".join(f"[{c}]" for c in whites)
    body += ")"
    return body
