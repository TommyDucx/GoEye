"""Persistent settings for GoEye, stored under ``~/.goeye/config.json``."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Optional

CONFIG_DIR = Path.home() / ".goeye"
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclasses.dataclass
class Settings:
    # Capture rectangle in logical screen points.
    region: Optional[dict] = None

    # Grid placement inside the capture, as fractions of the crop.
    grid: Optional[dict] = None

    # Engine
    katago_path: str = "katago"
    model_path: str = ""
    max_visits: int = 300
    komi: float = 7.5
    rules: str = "chinese"
    search_threads: int = 4

    # Behaviour
    auto_board_size: bool = True  # detect 9/13/19 (or cropped visible size)
    board_size: int = 19          # manual override when auto_board_size is False
    poll_interval: float = 0.7
    side_mode: str = "auto"  # auto | black | white
    show_screen_marker: bool = True
    top_moves: int = 5

    # Optional manual colour calibration: {"black": [v, s], ...}
    color_refs: Optional[dict] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def load(cls) -> "Settings":
        if not CONFIG_PATH.exists():
            return cls()
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
