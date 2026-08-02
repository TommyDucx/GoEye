"""Background capture-recognise-analyse loop.

Everything expensive happens on this thread so the Qt event loop never stalls:
screen grabs, OpenCV work and the (blocking) KataGo query all live here, and
results reach the UI purely through signals.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .config import Settings
from .engine import (
    AnalysisResult,
    EngineError,
    GnuGoEngine,
    KataGoEngine,
    find_gnugo,
    find_katago,
    stones_from_array,
)
from .vision import (
    BoardReading,
    ColorRefs,
    GridSpec,
    ScreenGrabber,
    board_signature,
    fit_grid,
    infer_side_to_move,
    read_board,
)


class AnalysisWorker(QThread):
    statusChanged = pyqtSignal(str)
    engineReady = pyqtSignal(str)
    engineFailed = pyqtSignal(str)
    boardUpdated = pyqtSignal(object, object)  # BoardReading, GridSpec
    analysisReady = pyqtSignal(object)  # AnalysisResult
    captureFailed = pyqtSignal(str)

    def __init__(self, settings: Settings, models: Optional[list[str]] = None) -> None:
        super().__init__()
        self.settings = settings
        self._models = models or [settings.model_path] if settings.model_path else []
        self._grabber = ScreenGrabber()
        self._engine: Optional[KataGoEngine] = None

        self._stop = False
        self._active = False
        self._refit = True
        self._force = False

        self._grid: Optional[GridSpec] = None
        self._prev_stones: Optional[np.ndarray] = None
        self._last_signature: Optional[bytes] = None
        self._pending_signature: Optional[bytes] = None
        self._stable_count = 0
        self._last_frame: Optional[np.ndarray] = None

    # -- external controls -------------------------------------------------

    def set_active(self, active: bool) -> None:
        self._active = active
        if active:
            self._force = True

    def request_refit(self) -> None:
        self._refit = True
        self._force = True

    def request_reanalyze(self) -> None:
        self._force = True

    def shutdown(self) -> None:
        self._stop = True

    def latest_frame(self) -> Optional[np.ndarray]:
        return self._last_frame

    def current_grid(self) -> Optional[GridSpec]:
        return self._grid

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        self._init_engine()

        while not self._stop:
            if not self._active or not self.settings.region:
                time.sleep(0.15)
                continue
            try:
                self._tick()
            except EngineError as exc:
                self.statusChanged.emit(f"引擎错误：{exc}")
                time.sleep(1.0)
            except Exception as exc:  # keep the loop alive on transient errors
                self.captureFailed.emit(f"{type(exc).__name__}: {exc}")
                time.sleep(0.8)
            time.sleep(max(self.settings.poll_interval, 0.15))

        if self._engine:
            self._engine.stop()
        self._grabber.close()

    def _init_engine(self) -> None:
        """Bring up the strongest available engine, or run in recognition-only
        mode if none can be found.  Never aborts the worker loop."""
        self._engine = None
        self._engine_kind = None

        katago = find_katago(self.settings.katago_path)
        if katago:
            try:
                self._engine = KataGoEngine(
                    katago_path=katago,
                    model_path=self._models,
                    search_threads=self.settings.search_threads,
                )
                self._engine.start(timeout=600.0)
                loaded = self._engine.model_path or self.settings.model_path or ""
                backend = self._engine.backend or "GPU"
                self._engine_kind = "katago"
                self.engineReady.emit(f"KataGo 就绪 · {loaded.split('/')[-1]} · {backend}")
                return
            except Exception as exc:
                tail = ""
                if self._engine is not None:
                    tail = "\n".join(getattr(self._engine, "_stderr_tail", [])[-12:])
                self._katago_error = f"{exc}\n{tail}" if tail else str(exc)
                self._engine = None

        gnugo = find_gnugo()
        if gnugo:
            try:
                self._engine = GnuGoEngine(gnugo)
                self._engine.start()
                self._engine_kind = "gnugo"
                self.engineReady.emit("GNU Go 就绪（轻量引擎，仅供参考）")
                return
            except Exception as exc:
                self._gnugo_error = str(exc)
                self._engine = None

        # No engine available: recognition still works, analysis is skipped.
        msg = (
            "仅识别模式 · 未检测到 KataGo / GNU Go。"
            "安装任一引擎后重启即可获得落子建议（详见说明文档）。"
        )
        if getattr(self, "_katago_error", None):
            msg += f" 已找到 KataGo 但启动失败：{self._katago_error[:160]}"
        self._engine_kind = None
        self.engineReady.emit(msg)

    def _tick(self) -> None:
        frame = self._grabber.grab(self.settings.region)
        self._last_frame = frame

        if frame.size == 0:
            self.captureFailed.emit("截屏返回空图像")
            return

        if self._grid is None or self._refit:
            fit_size = 0 if self.settings.auto_board_size else self.settings.board_size
            self._grid = fit_grid(frame, fit_size)
            self._refit = False
            if self._grid.confidence < 0.12:
                self.statusChanged.emit(
                    f"未能自动对齐棋盘格线（置信度 {self._grid.confidence:.2f}）"
                    "，请把框选范围收紧到棋盘边缘"
                )

        refs = _refs_from_settings(self.settings)
        reading = read_board(frame, self._grid, refs)
        self.boardUpdated.emit(reading, self._grid)

        signature = board_signature(reading.stones)

        # Require two identical consecutive frames before spending engine time:
        # most clients animate stone placement, and a half-drawn stone would
        # otherwise trigger an analysis of a position that never existed.
        if signature == self._pending_signature:
            self._stable_count += 1
        else:
            self._pending_signature = signature
            self._stable_count = 1

        if self._stable_count < 2 and not self._force:
            return
        if signature == self._last_signature and not self._force:
            return

        side = self._resolve_side(reading)
        stones = stones_from_array(reading.stones)

        if not stones:
            self.statusChanged.emit("当前区域没有识别到任何棋子")
            self._last_signature = signature
            return

        if self._engine is None:
            # Recognition-only mode: board is shown, no move suggestion.
            return

        board_size = int(reading.size)
        self.statusChanged.emit(
            f"分析中… {len(stones)} 子，{board_size}×{board_size}，"
            f"轮到{'黑' if side == 'B' else '白'}方"
        )
        assert self._engine is not None
        result: Optional[AnalysisResult] = self._engine.analyze(
            stones=stones,
            side_to_move=side,
            komi=self.settings.komi,
            rules=self.settings.rules,
            board_size=board_size,
            max_visits=self.settings.max_visits,
            timeout=60.0,
        )
        if result is None:
            self.statusChanged.emit("引擎超时，已跳过本次分析")
            return

        self._last_signature = signature
        self._prev_stones = reading.stones.copy()
        self._force = False
        self.analysisReady.emit(result)

    def _resolve_side(self, reading: BoardReading) -> str:
        mode = self.settings.side_mode
        if mode == "black":
            return "B"
        if mode == "white":
            return "W"
        return infer_side_to_move(reading.stones, self._prev_stones)


def _refs_from_settings(settings: Settings) -> Optional[ColorRefs]:
    data = settings.color_refs
    if not data:
        return None
    try:
        return ColorRefs(
            black=tuple(data["black"]),
            white=tuple(data["white"]),
            empty=tuple(data["empty"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
