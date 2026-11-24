"""Командная строка: vcmx2stp.exe --cli файл.vcmx [-o файл.stp] ..."""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional

from . import __version__
from .converter import ConvertOptions, convert


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vcmx2stp", description="Конвертер KUKA.Sim/Visual Components .vcmx -> STEP (.stp)")
    p.add_argument("inputs", nargs="+", help="файлы .vcmx (допускаются маски *.vcmx)")
    p.add_argument("-o", "--output", help="выходной .stp (для одного файла) или папка")
    p.add_argument("-s", "--search-dir", action="append", default=[],
                   help="папка с исходными компонентами (eCatalog, My Models) для поиска невстроенной геометрии")
    p.add_argument("--hidden", action="store_true", help="включать скрытые элементы (Visible = 0)")
    p.add_argument("--envelopes", action="store_true", help="включать рабочие зоны роботов")
    p.add_argument("--no-colors", action="store_true", help="не записывать цвета (немного меньше размер)")
    p.add_argument("--angle-tol", type=float, default=0.5, help="допуск объединения компланарных треугольников, градусы")
    p.add_argument("--strict", action="store_true", help="код возврата 2, если в файле не хватает встроенных моделей")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _utf8_console() -> None:
    """Консоль Windows может быть в cp1251/cp866/cp1252 — выводим UTF-8 без падений."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    _utf8_console()
    args = build_parser().parse_args(argv)
    files: List[str] = []
    for i in args.inputs:
        files.extend(sorted(glob.glob(i)) or [i])
    opt = ConvertOptions(search_dirs=args.search_dir, include_hidden=args.hidden, include_envelopes=args.envelopes,
                         colors=not args.no_colors, angle_tol=args.angle_tol)
    code = 0
    for f in files:
        if not os.path.isfile(f):
            print(f"Файл не найден: {f}", file=sys.stderr)
            code = 1
            continue
        out = None
        if args.output:
            if len(files) == 1 and not os.path.isdir(args.output):
                out = args.output
            else:
                os.makedirs(args.output, exist_ok=True)
                out = os.path.join(args.output, os.path.splitext(os.path.basename(f))[0] + ".stp")
        try:
            res = convert(f, out, opt, log=print)
        except Exception as ex:  # pragma: no cover
            print(f"ОШИБКА: {f}: {ex}", file=sys.stderr)
            code = 1
            continue
        if res.has_missing:
            print("\nПРЕДУПРЕЖДЕНИЕ: в файле не хватает встроенных моделей (см. список выше).", file=sys.stderr)
            if args.strict:
                code = max(code, 2)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
