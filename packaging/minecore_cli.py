"""PyInstaller entry point for the console program ``minecore-cli.exe``.

It runs the headless command-line interface (``minecore.cli``) and is built
next to ``MineCore.exe`` so both share one ``_internal`` folder.
"""

import sys

# A console redirected to a file uses the ANSI code page; never fail on a
# character it cannot show (Turkish field names, units).
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

from minecore.cli import main  # noqa: E402 - after the stream fix-up

raise SystemExit(main())
