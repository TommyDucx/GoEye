"""Standalone launcher used by PyInstaller to build GoEye.app.

A top-level (non-package) entry avoids relative-import issues that arise when
PyInstaller freezes ``goeye/__main__.py`` directly.
"""

from goeye.main import main

if __name__ == "__main__":
    raise SystemExit(main())
