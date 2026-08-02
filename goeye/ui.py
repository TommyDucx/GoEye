"""Qt widgets for GoEye: region picker, board canvas, screen marker, main panel."""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .engine import AnalysisResult
from .vision import BLACK, EMPTY, WHITE, GridSpec, gtp_to_rc

BOARD_COLOR = QColor("#e3b96b")
LINE_COLOR = QColor("#4a3520")
PANEL_BG = QColor("#fbfaf7")
TEXT_MAIN = QColor("#1d1c1a")
TEXT_DIM = QColor("#7a756d")
ACCENT = QColor("#c0392b")
SUGGEST_COLORS = ["#c0392b", "#e08a1e", "#2f8f5b", "#3b7dd8", "#8e6fc4"]
MARK_LETTERS = "ABCDE"


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """Convert an OpenCV BGR array into a QImage (copying the buffer)."""
    height, width = frame.shape[:2]
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


# ---------------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------------


class RegionSelector(QWidget):
    """Full-screen dimmed overlay; drag a rectangle to pick the board."""

    regionSelected = pyqtSignal(dict)
    cancelled = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._origin: Optional[QPoint] = None
        self._current: Optional[QPoint] = None

    def start(self) -> None:
        from PyQt6.QtWidgets import QApplication

        geometry = QRect()
        for screen in QApplication.screens():
            geometry = geometry.united(screen.geometry())
        self.setGeometry(geometry)
        self._origin = None
        self._current = None

        # On macOS showFullScreen() moves the overlay to its own Space,
        # leaving the board behind on the original desktop. Use show() so
        # the dim overlay stays on the current Space and the board remains
        # visible underneath.
        if sys.platform == "darwin":
            self.show()
        else:
            self.showFullScreen()
        self.raise_()
        self.activateWindow()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.pos()
            self._current = event.pos()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._origin is not None:
            self._current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._origin is None:
            return
        rect = QRect(self._origin, event.pos()).normalized()
        self.hide()
        if rect.width() < 40 or rect.height() < 40:
            self.cancelled.emit()
            return
        offset = self.geometry().topLeft()
        self.regionSelected.emit(
            {
                "left": rect.left() + offset.x(),
                "top": rect.top() + offset.y(),
                "width": rect.width(),
                "height": rect.height(),
            }
        )

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            self.cancelled.emit()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))

        if self._origin is not None and self._current is not None:
            rect = QRect(self._origin, self._current).normalized()
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            painter.setPen(QPen(QColor("#ffd24a"), 2))
            painter.drawRect(rect)
            painter.setPen(QPen(QColor("#ffffff")))
            painter.setFont(QFont("PingFang SC", 12))
            painter.drawText(
                rect.left(),
                max(rect.top() - 8, 14),
                f"{rect.width()} × {rect.height()}",
            )
        else:
            painter.setPen(QPen(QColor("#ffffff")))
            painter.setFont(QFont("PingFang SC", 18))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "拖动鼠标框住棋盘（尽量贴着最外圈线）\nEsc 取消",
            )


# ---------------------------------------------------------------------------
# Click-through marker drawn on top of the real board
# ---------------------------------------------------------------------------


class ScreenMarker(QWidget):
    """Transparent, click-through overlay that rings the recommended points.

    Only *hollow* rings are drawn, at a radius larger than the sampling disk
    used by the recogniser.  That way the overlay never contaminates the next
    screenshot and we avoid a hide/show flicker on every frame.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._grid: Optional[GridSpec] = None
        self._marks: list[tuple[int, int, int]] = []  # row, col, rank

    def update_marks(
        self, region: dict, grid: Optional[GridSpec], result: Optional[AnalysisResult]
    ) -> None:
        if not region or grid is None or result is None or not result.moves:
            self.hide()
            return
        self.setGeometry(
            region["left"], region["top"], region["width"], region["height"]
        )
        self._grid = grid
        self._marks = []
        for rank, move in enumerate(result.moves[:3]):
            rc = gtp_to_rc(move.move, grid.size)
            if rc is not None:
                self._marks.append((rc[0], rc[1], rank))
        self.show()
        self.raise_()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        if self._grid is None or not self._marks:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        xs, ys = self._grid.pixels(width, height)
        cell = self._grid.cell_px(width, height)
        radius = cell * 0.46

        for row, col, rank in self._marks:
            color = QColor(SUGGEST_COLORS[min(rank, len(SUGGEST_COLORS) - 1)])
            cx, cy = xs[col], ys[row]
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(color, 3 if rank == 0 else 2))
            painter.drawEllipse(
                QPoint(int(cx), int(cy)), int(radius), int(radius)
            )
            if rank == 0:
                painter.setPen(QPen(color, 2))
                painter.drawEllipse(
                    QPoint(int(cx), int(cy)), int(radius * 0.62), int(radius * 0.62)
                )


# ---------------------------------------------------------------------------
# Reconstructed board
# ---------------------------------------------------------------------------


class BoardCanvas(QWidget):
    """Draws the board GoEye *thinks* it sees, plus the engine's suggestions."""

    # Star-point indices for common board sizes (0-based).
    STARS: dict[int, list[int]] = {
        19: [3, 9, 15],
        13: [3, 6, 9],
        9: [2, 4, 6],
        7: [2, 4],
        6: [2],
        5: [2],
    }

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._stones: Optional[np.ndarray] = None
        self._result: Optional[AnalysisResult] = None
        self._size = 19

    def set_board(self, stones: np.ndarray) -> None:
        self._stones = stones
        self._size = stones.shape[0]
        self.update()

    def set_result(self, result: Optional[AnalysisResult]) -> None:
        self._result = result
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        side = min(self.width(), self.height())
        margin = side * 0.045
        board_rect = QRect(
            int((self.width() - side) / 2), int((self.height() - side) / 2), side, side
        )
        painter.fillRect(board_rect, BOARD_COLOR)

        n = self._size
        usable = side - 2 * margin
        cell = usable / (n - 1)
        ox = board_rect.left() + margin
        oy = board_rect.top() + margin

        painter.setPen(QPen(LINE_COLOR, 1))
        for i in range(n):
            painter.drawLine(
                int(ox), int(oy + i * cell), int(ox + usable), int(oy + i * cell)
            )
            painter.drawLine(
                int(ox + i * cell), int(oy), int(ox + i * cell), int(oy + usable)
            )

        star_indices = self.STARS.get(n)
        if star_indices:
            painter.setBrush(QBrush(LINE_COLOR))
            painter.setPen(Qt.PenStyle.NoPen)
            dot = max(2, int(cell * 0.11))
            for r in star_indices:
                if r >= n:
                    continue
                for c in star_indices:
                    if c >= n:
                        continue
                    painter.drawEllipse(
                        QPoint(int(ox + c * cell), int(oy + r * cell)), dot, dot
                    )

        if self._stones is None:
            return

        stone_r = cell * 0.47
        for row in range(n):
            for col in range(n):
                value = int(self._stones[row, col])
                if value == EMPTY:
                    continue
                cx, cy = ox + col * cell, oy + row * cell
                self._draw_stone(painter, cx, cy, stone_r, value == BLACK)

        self._draw_suggestions(painter, ox, oy, cell)

    @staticmethod
    def _draw_stone(
        painter: QPainter, cx: float, cy: float, radius: float, black: bool
    ) -> None:
        gradient = QRadialGradient(cx - radius * 0.3, cy - radius * 0.35, radius * 1.6)
        if black:
            gradient.setColorAt(0.0, QColor("#5c5c5c"))
            gradient.setColorAt(1.0, QColor("#0a0a0a"))
        else:
            gradient.setColorAt(0.0, QColor("#ffffff"))
            gradient.setColorAt(1.0, QColor("#c9c6bf"))
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(0, 0, 0, 90), 1))
        painter.drawEllipse(QPoint(int(cx), int(cy)), int(radius), int(radius))

    def _draw_suggestions(
        self, painter: QPainter, ox: float, oy: float, cell: float
    ) -> None:
        if self._result is None or not self._result.moves:
            return
        font = QFont("Menlo", max(7, int(cell * 0.42)))
        font.setBold(True)

        for rank, move in enumerate(self._result.moves[:5]):
            rc = gtp_to_rc(move.move, self._size)
            if rc is None:
                continue
            row, col = rc
            cx, cy = ox + col * cell, oy + row * cell
            color = QColor(SUGGEST_COLORS[min(rank, len(SUGGEST_COLORS) - 1)])

            radius = cell * 0.46
            fill = QColor(color)
            fill.setAlpha(215 if rank == 0 else 120)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(color, 2))
            painter.drawEllipse(QPoint(int(cx), int(cy)), int(radius), int(radius))

            painter.setFont(font)
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(
                QRect(
                    int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2)
                ),
                Qt.AlignmentFlag.AlignCenter,
                MARK_LETTERS[rank],
            )


# ---------------------------------------------------------------------------
# Colour calibration
# ---------------------------------------------------------------------------


class CalibrationDialog(QDialog):
    """Click a black stone, a white stone and an empty point to teach GoEye."""

    PROMPTS = [
        ("black", "请点击一颗【黑子】的中心"),
        ("white", "请点击一颗【白子】的中心"),
        ("empty", "请点击一个【空交叉点】"),
    ]

    def __init__(self, frame: np.ndarray, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("颜色校准")
        self._frame = frame
        self._step = 0
        self.picked: dict[str, tuple[float, float]] = {}

        import cv2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self._value = hsv[:, :, 2].astype(np.float32)
        self._saturation = hsv[:, :, 1].astype(np.float32)

        image = bgr_to_qimage(frame)
        self._pixmap = QPixmap.fromImage(image)
        max_side = 620
        if max(self._pixmap.width(), self._pixmap.height()) > max_side:
            self._pixmap = self._pixmap.scaled(
                max_side,
                max_side,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        self._prompt = QLabel(self.PROMPTS[0][1])
        self._prompt.setStyleSheet("font-size:14px;font-weight:600;padding:6px;")
        self._image_label = QLabel()
        self._image_label.setPixmap(self._pixmap)
        self._image_label.setCursor(Qt.CursorShape.CrossCursor)
        self._image_label.mousePressEvent = self._on_click  # type: ignore[assignment]

        layout = QVBoxLayout(self)
        layout.addWidget(self._prompt)
        layout.addWidget(self._image_label)

    def _on_click(self, event: QMouseEvent) -> None:
        if self._step >= len(self.PROMPTS):
            return
        scale_x = self._frame.shape[1] / self._pixmap.width()
        scale_y = self._frame.shape[0] / self._pixmap.height()
        x = int(event.pos().x() * scale_x)
        y = int(event.pos().y() * scale_y)
        height, width = self._value.shape
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        half = 3
        patch_v = self._value[
            max(0, y - half) : y + half + 1, max(0, x - half) : x + half + 1
        ]
        patch_s = self._saturation[
            max(0, y - half) : y + half + 1, max(0, x - half) : x + half + 1
        ]
        key = self.PROMPTS[self._step][0]
        self.picked[key] = (float(np.median(patch_v)), float(np.median(patch_s)))

        self._step += 1
        if self._step < len(self.PROMPTS):
            self._prompt.setText(self.PROMPTS[self._step][1])
        else:
            self.accept()


# ---------------------------------------------------------------------------
# Small helpers for the panel
# ---------------------------------------------------------------------------


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color:#e6e2da;")
    return line


def make_button(text: str, primary: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    if primary:
        button.setStyleSheet(
            "QPushButton{background:#1d1c1a;color:#fff;border:none;border-radius:6px;"
            "padding:7px 14px;font-size:13px;font-weight:600;}"
            "QPushButton:hover{background:#3a3733;}"
            "QPushButton:disabled{background:#c9c4bb;}"
        )
    else:
        button.setStyleSheet(
            "QPushButton{background:#f0ece4;color:#1d1c1a;border:1px solid #ddd7cc;"
            "border-radius:6px;padding:7px 12px;font-size:13px;}"
            "QPushButton:hover{background:#e6e1d7;}"
            "QPushButton:disabled{color:#a8a29a;}"
        )
    return button
