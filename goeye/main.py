"""GoEye entry point."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from .config import Settings

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# Larger nets first: whichever is present and loads wins.
MODEL_PREFERENCE = ["g170e-b15c192", "g170e-b10c128"]


def find_katago(configured: str = "katago") -> Optional[str]:
    # Centralised in the engine module so discovery stays in one place.
    from .engine import find_katago as _find_katago

    return _find_katago(configured)


def find_models(configured: str = "") -> list[str]:
    """Return candidate model paths, larger/better nets first."""
    if configured and os.path.exists(configured):
        return [configured]
    if not MODELS_DIR.exists():
        return []
    available = sorted(MODELS_DIR.glob("*.gz"))
    ordered: list[str] = []
    for stem in MODEL_PREFERENCE:
        for path in available:
            if path.name.startswith(stem) and str(path) not in ordered:
                ordered.append(str(path))
    # Append any other weights we did not anticipate.
    for path in available:
        if str(path) not in ordered:
            ordered.append(str(path))
    return ordered


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("GoEye")

    settings = Settings.load()

    katago = find_katago(settings.katago_path)
    models = find_models(settings.model_path)

    # Launch regardless of engine availability: the worker falls back to
    # recognition-only mode and tells the user how to enable move analysis.
    if katago and models:
        settings.katago_path = katago
        settings.model_path = models[0]
    settings.save()

    from .panel import GoEyePanel

    use_models = models if (katago and models) else None
    panel = GoEyePanel(settings, models=use_models)
    panel.show()

    # Park the panel on the right edge so it does not sit on top of the board.
    screen = app.primaryScreen()
    if screen is not None:
        area = screen.availableGeometry()
        panel.move(area.right() - panel.width() - 24, area.top() + 60)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
