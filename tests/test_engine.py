"""Engine-layer tests (no display required).

* KataGo's JSON analysis parsing is checked against a synthetic payload, so the
  protocol mapping is validated without the (heavy) katago binary.
* GNU Go's GTP plumbing is exercised against a tiny *fake* gnugo subprocess,
  proving the request/response handling end-to-end.
* A real KataGo end-to-end check runs only when katago + a model are present;
  otherwise it prints SKIP and exits 0.
"""

from __future__ import annotations

import os
import stat
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from goeye.engine import (  # noqa: E402
    AnalysisResult,
    GnuGoEngine,
    KataGoEngine,
    find_gnugo,
    find_katago,
    stones_from_array,
)
from goeye.main import find_models  # noqa: E402
from goeye.vision import BLACK, WHITE  # noqa: E402

FAKE_GNUGO = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import sys

    def respond(body=""):
        sys.stdout.write("=1 " + body + "\\n\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        name = line.split()[0].lower()
        if name == "name":
            respond("Fake GNU Go 3.8")
        elif name in ("boardsize", "komi", "clear_board", "play"):
            respond()
        elif name == "genmove":
            respond("Q4")
        elif name == "estimate_score":
            respond("W+3.50")
        elif name == "quit":
            break
        else:
            respond()
    """
)


def _write_fake_gnugo(tmp_path: Path) -> str:
    p = tmp_path / "fake_gnugo"
    p.write_text(FAKE_GNUGO, encoding="utf-8")
    mode = p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    p.chmod(mode)
    return str(p)


def test_katago_parse() -> None:
    payload = {
        "id": "q1",
        "initialPlayer": "B",
        "moveInfos": [
            {
                "move": "Q4",
                "visits": 300,
                "winrate": 0.62,
                "scoreLead": 5.3,
                "prior": 0.3,
                "order": 0,
                "pv": ["Q4", "D16", "Q16"],
            },
            {
                "move": "D4",
                "visits": 120,
                "winrate": 0.55,
                "scoreLead": 3.1,
                "prior": 0.2,
                "order": 1,
                "pv": ["D4", "Q16"],
            },
        ],
        "rootInfo": {"winrate": 0.62, "scoreLead": 5.3, "visits": 420},
    }
    res = KataGoEngine._parse(payload, "B", 1.23)
    assert isinstance(res, AnalysisResult)
    assert res.side_to_move == "B"
    assert res.best.move == "Q4"
    assert abs(res.best.winrate - 0.62) < 1e-6
    assert abs(res.best.score_lead - 5.3) < 1e-6
    assert res.best.pv == ["Q4", "D16", "Q16"]
    assert len(res.moves) == 2
    print("test_katago_parse: PASS")


def test_gnugo_score_parse() -> None:
    assert GnuGoEngine._parse_score("0.00", "B") == 0.0
    assert GnuGoEngine._parse_score("", "W") == 0.0
    assert abs(GnuGoEngine._parse_score("W+3.50", "W") - 3.5) < 1e-6
    assert abs(GnuGoEngine._parse_score("W+3.50", "B") + 3.5) < 1e-6
    assert abs(GnuGoEngine._parse_score("B+10.00", "B") - 10.0) < 1e-6
    assert abs(GnuGoEngine._parse_score("B+10.00", "W") + 10.0) < 1e-6
    print("test_gnugo_score_parse: PASS")


def test_gnugo_fake(tmp_path: Path) -> None:
    path = _write_fake_gnugo(tmp_path)
    eng = GnuGoEngine(path)
    eng.start()
    try:
        board = np.full((19, 19), 0, dtype=np.uint8)
        board[3, 3] = BLACK
        board[15, 15] = WHITE
        stones = stones_from_array(board)
        res = eng.analyze(stones, "W", komi=7.5, board_size=19)
        assert isinstance(res, AnalysisResult)
        assert res.side_to_move == "W"
        assert res.best.move == "Q4"
        assert abs(res.best.score_lead - 3.5) < 1e-6  # W+3.5 for White to move
        assert res.best.visits == 0
    finally:
        eng.stop()
    print("test_gnugo_fake: PASS")


def test_discovery() -> None:
    # find_katago/find_gnugo must not raise and return either a path or None.
    assert find_katago() in (None,) or isinstance(find_katago(), str)
    assert find_gnugo() in (None,) or isinstance(find_gnugo(), str)
    print("test_discovery: PASS")


def test_katago_e2e() -> int:
    katago = find_katago()
    models = find_models()
    if not katago:
        print("SKIP katago e2e: katago not found")
        return 0
    if not models:
        print("SKIP katago e2e: no model weights")
        return 0

    print(f"katago : {katago}")
    print(f"model  : {models[0]}")
    engine = KataGoEngine(katago_path=katago, model_path=models, search_threads=4)
    try:
        engine.start(timeout=600.0)
    except Exception as exc:  # noqa: BLE001
        print("ENGINE START FAILED:", exc)
        return 1
    print(f"engine ready (backend={engine.backend}, model={engine.model_path})")

    result = engine.analyze([], "B", max_visits=200, timeout=60)
    assert result is not None, "no response"
    best = result.best
    print(
        f"empty board B to move -> best {best.move} "
        f"winrate={best.winrate*100:.1f}% lead={best.score_lead:+.1f} "
        f"pv={best.pv[:6]}"
    )

    board = np.full((19, 19), 0, dtype=np.uint8)
    board[3, 3] = BLACK
    board[15, 15] = WHITE
    result2 = engine.analyze(stones_from_array(board), "B", max_visits=200, timeout=60)
    assert result2 is not None
    print(
        f"2 stones    B to move -> best {result2.best.move} "
        f"winrate={result2.best.winrate*100:.1f}% lead={result2.best.score_lead:+.1f}"
    )
    engine.stop()
    print("katago e2e: PASS")
    return 0


def main() -> int:
    test_katago_parse()
    test_gnugo_score_parse()
    test_discovery()
    # tmp_path emulation for the fake-gnugo subprocess test
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_gnugo_fake(Path(d))
    rc = test_katago_e2e()
    if rc != 0:
        return rc
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
