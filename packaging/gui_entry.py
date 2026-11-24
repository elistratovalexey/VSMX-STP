"""Точка входа оконного приложения VCMX2STP.exe (PyInstaller)."""
import multiprocessing
import sys

from vcmx2stp.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
