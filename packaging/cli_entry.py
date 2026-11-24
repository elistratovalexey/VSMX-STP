"""Точка входа консольной версии vcmx2stp-cli.exe (PyInstaller)."""
import multiprocessing
import sys

from vcmx2stp.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
