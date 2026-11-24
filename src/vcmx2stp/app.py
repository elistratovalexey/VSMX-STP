"""Точка входа: без аргументов (или с путями к файлам) — окно; с --cli — командная строка."""
from __future__ import annotations

import multiprocessing
import os
import sys


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] in ("--cli", "-c"):
        from .cli import main as cli_main

        return cli_main(argv[1:])
    if argv and argv[0] in ("--version", "-V"):
        from . import __version__

        print(__version__)
        return 0
    from .gui import main as gui_main

    return gui_main([a for a in argv if os.path.isfile(a)])


if __name__ == "__main__":  # pragma: no cover
    multiprocessing.freeze_support()
    sys.exit(main())
