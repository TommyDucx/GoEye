"""The floating GoEye control panel."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QFont, QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .config import Settings
from .engine import AnalysisResult
from .ui import (
    MARK_LETTERS,
    SUGGEST_COLORS,
    BoardCanvas,
    CalibrationDialog,
    RegionSelector,
    ScreenMarker,
    _hline,
    make_button,
)
from .vision import BoardReading, GridSpec, board_to_sgf, gtp_to_rc
from .worker import AnalysisWorker

SIDE_MODES = [("自动判断", "auto"), ("轮到黑棋", "black"), ("轮到白棋", "white")]


class GoEyePanel(QWidget):
    def __init__(self, settings: Settings, models: Optional[list[str]] = None) -> None:
        super().__init__()
        self.settings = settings
        self._models = models or []
        self._result: Optional[AnalysisResult] = None
        self._grid: Optional[GridSpec] = None
        self._last_stones: Optional[np.ndarray] = None
        self._drag_offset = None

        self.setWindowTitle("GoEye 围棋助手")
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setStyleSheet(
            "QWidget{background:#fbfaf7;color:#1d1c1a;"
            "font-family:'PingFang SC','Helvetica Neue',sans-serif;}"
            "QLabel{color:#1d1c1a;}"
            "QComboBox,QSpinBox,QDoubleSpinBox{background:#fff;border:1px solid #ddd7cc;"
            "border-radius:5px;padding:3px 6px;font-size:12px;}"
        )
        self.resize(360, 620)

        self._selector = RegionSelector()
        self._selector.regionSelected.connect(self._on_region_selected)
        self._selector.cancelled.connect(self._on_select_cancelled)

        self._marker = ScreenMarker()

        self._build_ui()

        self.worker = AnalysisWorker(settings, models=self._models)
        self.worker.statusChanged.connect(self._set_status)
        self.worker.engineReady.connect(self._on_engine_ready)
        self.worker.engineFailed.connect(self._on_engine_failed)
        self.worker.boardUpdated.connect(self._on_board)
        self.worker.analysisReady.connect(self._on_analysis)
        self.worker.captureFailed.connect(self._set_status)
        self.worker.start()

        # If a board region was saved from a previous session, start watching
        # it immediately — the user should never have to re-pick the region
        # or press "开始分析" just to resume real-time monitoring.
        if self.settings.region:
            self._start_monitoring()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(9)

        title = QLabel("GoEye · 实时棋盘分析")
        title.setStyleSheet("font-size:15px;font-weight:700;")
        root.addWidget(title)

        self._status = QLabel("正在启动引擎…")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size:11px;color:#7a756d;")
        root.addWidget(self._status)

        root.addWidget(_hline())

        # Headline recommendation
        self._headline = QLabel("—")
        self._headline.setStyleSheet(
            "font-size:26px;font-weight:800;color:#c0392b;letter-spacing:1px;"
        )
        root.addWidget(self._headline)

        self._subline = QLabel("等待识别棋盘")
        self._subline.setStyleSheet("font-size:12px;color:#4a463f;")
        root.addWidget(self._subline)

        self._canvas = BoardCanvas()
        self._canvas.setMinimumHeight(320)
        root.addWidget(self._canvas, stretch=1)

        self._moves_label = QLabel("")
        self._moves_label.setTextFormat(Qt.TextFormat.RichText)
        self._moves_label.setStyleSheet("font-size:11.5px;")
        root.addWidget(self._moves_label)

        self._pv_label = QLabel("")
        self._pv_label.setWordWrap(True)
        self._pv_label.setStyleSheet(
            "font-size:11px;color:#7a756d;font-family:Menlo,monospace;"
        )
        root.addWidget(self._pv_label)

        root.addWidget(_hline())

        controls = QGridLayout()
        controls.setHorizontalSpacing(6)
        controls.setVerticalSpacing(6)

        self._btn_region = make_button("框选棋盘")
        self._btn_region.clicked.connect(self._pick_region)
        controls.addWidget(self._btn_region, 0, 0)

        self._btn_toggle = make_button("开始分析", primary=True)
        self._btn_toggle.clicked.connect(self._toggle)
        self._btn_toggle.setEnabled(False)
        controls.addWidget(self._btn_toggle, 0, 1)

        self._btn_refit = make_button("重新对齐")
        self._btn_refit.clicked.connect(self.worker_refit)
        controls.addWidget(self._btn_refit, 0, 2)

        self._btn_calib = make_button("颜色校准")
        self._btn_calib.clicked.connect(self._calibrate)
        controls.addWidget(self._btn_calib, 1, 0)

        self._side_box = QComboBox()
        for label, _ in SIDE_MODES:
            self._side_box.addItem(label)
        current = [m[1] for m in SIDE_MODES].index(self.settings.side_mode)
        self._side_box.setCurrentIndex(current)
        self._side_box.currentIndexChanged.connect(self._on_side_changed)
        controls.addWidget(self._side_box, 1, 1)

        self._btn_again = make_button("重新分析")
        self._btn_again.clicked.connect(lambda: self.worker.request_reanalyze())
        controls.addWidget(self._btn_again, 1, 2)

        # Second control row: export / convenience actions.
        self._btn_sgf = make_button("复制 SGF")
        self._btn_sgf.clicked.connect(self._copy_sgf)
        controls.addWidget(self._btn_sgf, 2, 0)

        self._btn_save_sgf = make_button("保存 SGF")
        self._btn_save_sgf.clicked.connect(self._save_sgf)
        controls.addWidget(self._btn_save_sgf, 2, 1)

        self._btn_play = make_button("替我落子")
        self._btn_play.clicked.connect(self._auto_play)
        controls.addWidget(self._btn_play, 2, 2)

        # Board size: auto-detect or manual override.
        self._auto_size = QCheckBox("自动识别尺寸")
        self._auto_size.setChecked(self.settings.auto_board_size)
        self._auto_size.stateChanged.connect(self._on_auto_size_changed)
        controls.addWidget(self._auto_size, 3, 0)

        size_layout = QHBoxLayout()
        size_layout.setSpacing(4)
        size_layout.addWidget(QLabel("棋盘"))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(2, 19)
        self._size_spin.setValue(self.settings.board_size)
        self._size_spin.setEnabled(not self.settings.auto_board_size)
        self._size_spin.valueChanged.connect(self._on_manual_size_changed)
        size_layout.addWidget(self._size_spin)
        size_layout.addWidget(QLabel("路"))
        size_layout.addStretch(1)
        controls.addLayout(size_layout, 3, 1)

        # spacer keeps the grid aligned
        controls.addWidget(QWidget(), 3, 2)

        root.addLayout(controls)

        tuning = QHBoxLayout()
        tuning.setSpacing(6)
        tuning.addWidget(QLabel("算力"))
        self._visits = QSpinBox()
        self._visits.setRange(20, 5000)
        self._visits.setSingleStep(50)
        self._visits.setValue(self.settings.max_visits)
        self._visits.valueChanged.connect(self._on_visits_changed)
        tuning.addWidget(self._visits)

        tuning.addWidget(QLabel("贴目"))
        self._komi = QDoubleSpinBox()
        self._komi.setRange(-30.0, 30.0)
        self._komi.setSingleStep(0.5)
        self._komi.setValue(self.settings.komi)
        self._komi.valueChanged.connect(self._on_komi_changed)
        tuning.addWidget(self._komi)
        tuning.addStretch(1)
        root.addLayout(tuning)

    # -- interactions ------------------------------------------------------

    def worker_refit(self) -> None:
        self.worker.request_refit()
        self._set_status("正在重新对齐棋盘格线…")

    def _pick_region(self) -> None:
        self.hide()
        self._marker.hide()
        self._selector.start()

    def _start_monitoring(self) -> None:
        """Begin continuous, real-time monitoring of the selected region.

        No manual "开始分析" press is required: this is called automatically
        as soon as a region is chosen (or restored from settings on launch).
        The worker keeps watching and auto-reanalyses the instant the board
        changes — e.g. the moment the opponent places a stone.
        """
        if not self.settings.region:
            self._set_status("请先框选棋盘区域")
            return
        self.worker.set_active(True)
        self._btn_toggle.setText("暂停")
        self._btn_toggle.setEnabled(True)
        self.worker.request_refit()
        self._set_status("实时监控已开启 · 棋盘一有变动就自动分析")

    @pyqtSlot()
    def _on_select_cancelled(self) -> None:
        # Bring the panel back; keep any existing monitoring running.
        self.show()
        self.raise_()

    @pyqtSlot(dict)
    def _on_region_selected(self, region: dict) -> None:
        self.settings.region = region
        self.settings.save()
        self.show()
        self.raise_()
        # Picking the region is enough to start watching — no extra click.
        self._start_monitoring()

    def _toggle(self) -> None:
        active = self._btn_toggle.text() == "开始分析"
        if active and not self.settings.region:
            self._set_status("请先框选棋盘区域")
            return
        self.worker.set_active(active)
        self._btn_toggle.setText("暂停" if active else "开始分析")
        if not active:
            self._marker.hide()

    def _calibrate(self) -> None:
        frame = self.worker.latest_frame()
        if frame is None:
            self._set_status("还没有截到画面，请先框选区域并开始分析")
            return
        dialog = CalibrationDialog(frame, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and len(dialog.picked) == 3:
            self.settings.color_refs = {k: list(v) for k, v in dialog.picked.items()}
            self.settings.save()
            self.worker.request_reanalyze()
            self._set_status("颜色校准已保存")

    def _on_side_changed(self, index: int) -> None:
        self.settings.side_mode = SIDE_MODES[index][1]
        self.settings.save()
        self.worker.request_reanalyze()

    def _on_visits_changed(self, value: int) -> None:
        self.settings.max_visits = value
        self.settings.save()

    def _on_komi_changed(self, value: float) -> None:
        self.settings.komi = value
        self.settings.save()
        self.worker.request_reanalyze()

    def _on_auto_size_changed(self, state: int) -> None:
        auto = bool(state)
        self.settings.auto_board_size = auto
        self._size_spin.setEnabled(not auto)
        self.settings.save()
        self.worker.request_refit()
        mode = "已切换到自动识别棋盘尺寸" if auto else "已切换到手动棋盘尺寸"
        self._set_status(mode)

    def _on_manual_size_changed(self, value: int) -> None:
        if self.settings.auto_board_size:
            return
        self.settings.board_size = value
        self.settings.save()
        self.worker.request_refit()

    # -- export / auto-play -------------------------------------------------

    def _current_sgf(self) -> "str | None":
        if self._last_stones is None or self._grid is None:
            return None
        to_move = self._result.side_to_move if self._result else "B"
        return board_to_sgf(
            self._last_stones,
            self._grid.size,
            self.settings.komi,
            to_move,
        )

    def _copy_sgf(self) -> None:
        sgf = self._current_sgf()
        if sgf is None:
            self._set_status("还没有识别到棋盘")
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(sgf)
        self._set_status("已复制 SGF 到剪贴板")

    def _save_sgf(self) -> None:
        sgf = self._current_sgf()
        if sgf is None:
            self._set_status("还没有识别到棋盘")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 SGF 棋谱", "goeye-board.sgf", "SGF (*.sgf)"
        )
        if path:
            from pathlib import Path

            Path(path).write_text(sgf, encoding="utf-8")
            self._set_status(f"已保存棋谱：{path}")

    def _auto_play(self) -> None:
        """Click the recommended point on the real board (PytoGo-style).

        Requires the terminal to have Accessibility permission, otherwise macOS
        blocks synthetic clicks.  Analysis is paused for the click so the next
        screenshot does not immediately re-trigger a move.
        """
        if (
            not self.settings.region
            or self._grid is None
            or self._result is None
            or not self._result.moves
        ):
            self._set_status("需要先识别棋盘并得到 AI 建议")
            return
        best = self._result.best
        rc = gtp_to_rc(best.move, self._grid.size)
        if rc is None:
            return
        row, col = rc
        region = self.settings.region
        xs, ys = self._grid.pixels(region["width"], region["height"])
        x = region["left"] + float(xs[col])
        y = region["top"] + float(ys[row])

        was_active = self._btn_toggle.text() == "暂停"
        if was_active:
            self.worker.set_active(False)
            self._btn_toggle.setText("开始分析")
            self._marker.hide()

        try:
            _click_at(x, y)
            self._set_status(
                f"已在屏幕 ({x:.0f}, {y:.0f}) 点击 {best.move}。"
                "若没反应，请在系统设置→隐私→辅助功能中授权本终端"
            )
        except Exception as exc:  # pragma: no cover - platform dependent
            self._set_status(f"自动落子失败：{exc}")

    # -- worker callbacks --------------------------------------------------

    @pyqtSlot(str)
    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    @pyqtSlot(str)
    def _on_engine_ready(self, text: str) -> None:
        if self.settings.region:
            # Already monitoring (auto-started on launch or region pick);
            # make sure the button reflects that state.
            self._btn_toggle.setEnabled(True)
            if self._btn_toggle.text() == "开始分析":
                self._start_monitoring()
            else:
                self._set_status(text + " · 实时监控中，棋盘变动会自动分析")
        else:
            self._set_status(text + " · 请先框选棋盘区域")

    @pyqtSlot(str)
    def _on_engine_failed(self, detail: str) -> None:
        # Kept non-blocking: engine issues are surfaced in the status line
        # rather than a modal dialog that could strand the capture loop.
        self._set_status("引擎启动遇到问题：" + detail[:160])

    @pyqtSlot(object, object)
    def _on_board(self, reading: BoardReading, grid: GridSpec) -> None:
        self._grid = grid
        self._last_stones = reading.stones
        self._canvas.set_board(reading.stones)
        black, white = reading.counts()
        warn = " ⚠ 识别可能不准，建议颜色校准" if reading.ambiguity > 0.05 else ""
        mode = "自动" if self.settings.auto_board_size else "手动"
        self._subline.setText(
            f"识别到 {grid.size}×{grid.size} 棋盘（{mode}）· "
            f"黑 {black} 子 / 白 {white} 子 · 对齐置信度 {grid.confidence:.2f}{warn}"
        )

    @pyqtSlot(object)
    def _on_analysis(self, result: AnalysisResult) -> None:
        self._result = result
        self._canvas.set_result(result)

        best = result.best
        if best is None:
            self._headline.setText("—")
            return

        mover = "黑" if result.side_to_move == "B" else "白"
        self._headline.setText(f"{mover}方下 {best.move}")

        lead = best.score_lead
        lead_text = f"领先 {lead:.1f} 目" if lead >= 0 else f"落后 {abs(lead):.1f} 目"
        self._subline.setText(
            f"胜率 {best.winrate * 100:.1f}% · {lead_text} · "
            f"{result.visits} 次模拟 · {result.elapsed:.1f}s"
        )

        rows = []
        for rank, move in enumerate(result.moves[: self.settings.top_moves]):
            color = SUGGEST_COLORS[min(rank, len(SUGGEST_COLORS) - 1)]
            rows.append(
                f"<span style='color:{color};font-weight:700'>{MARK_LETTERS[rank]}</span> "
                f"<b>{move.move:<4}</b> {move.winrate * 100:5.1f}%  "
                f"{move.score_lead:+.1f}目  {move.visits}次"
            )
        self._moves_label.setText("<br>".join(rows))

        pv = " ".join(best.pv[:10])
        self._pv_label.setText(f"后续变化: {pv}" if pv else "")

        if self.settings.show_screen_marker and self.settings.region:
            self._marker.update_marks(self.settings.region, self._grid, result)

    # -- window behaviour --------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_offset = None

    def closeEvent(self, event: QCloseEvent) -> None:
        self.worker.shutdown()
        self.worker.wait(4000)
        self._marker.close()
        self._selector.close()
        self.settings.save()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Native mouse click via CoreGraphics (no extra dependency needed on macOS).
# ---------------------------------------------------------------------------

import ctypes  # noqa: E402
from ctypes import c_double, c_int, c_uint32, c_void_p, Structure  # noqa: E402


class _CGPoint(Structure):
    _fields_ = [("x", c_double), ("y", c_double)]


def _click_at(x: float, y: float) -> None:
    """Synthesize a left-click at global screen coordinates (macOS only)."""
    lib = ctypes.CDLL(
        "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
    )
    lib.CGEventCreateMouseEvent.argtypes = [c_void_p, c_int, _CGPoint, c_uint32]
    lib.CGEventCreateMouseEvent.restype = c_void_p
    lib.CGEventPost.argtypes = [c_int, c_void_p]
    lib.CGEventPost.restype = None
    lib.CFRelease.argtypes = [c_void_p]
    lib.CFRelease.restype = None

    K_CG_EVENT_LEFT_DOWN = 1
    K_CG_EVENT_LEFT_UP = 2
    K_CG_HID_EVENT_TAP = 0

    point = _CGPoint(x, y)
    for etype in (K_CG_EVENT_LEFT_DOWN, K_CG_EVENT_LEFT_UP):
        event = lib.CGEventCreateMouseEvent(None, etype, point, 0)
        if not event:
            raise RuntimeError("CGEventCreateMouseEvent failed")
        lib.CGEventPost(K_CG_HID_EVENT_TAP, event)
        lib.CFRelease(event)
