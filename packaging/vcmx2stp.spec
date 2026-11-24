# -*- mode: python ; coding: utf-8 -*-
# Сборка: pyinstaller packaging/vcmx2stp.spec --noconfirm
# Результат: dist/VCMX2STP/ (VCMX2STP.exe — окно, vcmx2stp-cli.exe — командная строка)
#
# Колёса cadquery-ocp (OpenCascade) для Windows хранят DLL в site-packages/cadquery_ocp.libs,
# а зависимые библиотеки VTK — в site-packages/vtk.libs. OCP/__init__.py вызывает
# os.add_dll_directory(../vtk.libs), поэтому обе папки переносятся в сборку под теми же именами.
# Чтобы не тащить все ~260 МБ VTK, берётся только транзитивное замыкание DLL,
# реально импортируемых модулем OCP (анализ таблиц импорта через pefile).
import os
import sys
import importlib.util

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

spec = importlib.util.find_spec("OCP")
OCP_DIR = os.path.dirname(spec.origin)
SITE = os.path.dirname(OCP_DIR)
LIB_DIRS = [d for d in (os.path.join(SITE, "cadquery_ocp.libs"), os.path.join(SITE, "vtk.libs")) if os.path.isdir(d)]


def _dll_closure():
    """Возвращает список (путь, папка_назначения) DLL, нужных модулю OCP."""
    if sys.platform != "win32":
        return []
    import pefile

    index = {}
    for d in LIB_DIRS:
        for n in os.listdir(d):
            if n.lower().endswith(".dll"):
                index[n.lower()] = os.path.join(d, n)
    roots = [os.path.join(OCP_DIR, n) for n in os.listdir(OCP_DIR) if n.lower().endswith(".pyd")]
    seen, todo, out = set(), list(roots), []
    while todo:
        path = todo.pop()
        try:
            pe = pefile.PE(path, fast_load=True)
            pe.parse_data_directories(directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
            ])
        except Exception as ex:  # pragma: no cover
            print("pefile:", path, ex)
            continue
        names = []
        for attr in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
            for e in getattr(pe, attr, []) or []:
                names.append(e.dll.decode("ascii", "ignore").lower())
        pe.close()
        for n in names:
            p = index.get(n)
            if p and p not in seen:
                seen.add(p)
                todo.append(p)
                out.append((p, os.path.basename(os.path.dirname(p))))
    print(f"[spec] OpenCascade/VTK DLLs bundled: {len(out)}")
    return out


binaries = _dll_closure()
# на случай ленивой загрузки плагинов OCCT — все DLL из cadquery_ocp.libs
if sys.platform == "win32" and LIB_DIRS:
    have = {os.path.normcase(p) for p, _ in binaries}
    d = LIB_DIRS[0]
    for n in os.listdir(d):
        p = os.path.join(d, n)
        if n.lower().endswith(".dll") and os.path.normcase(p) not in have:
            binaries.append((p, os.path.basename(d)))

hiddenimports = collect_submodules("OCP") + collect_submodules("vcmx2stp") + ["numpy"]
datas = []
icon = os.path.join(SRC, "vcmx2stp", "assets", "app.ico")
if os.path.isfile(icon):
    datas.append((icon, os.path.join("vcmx2stp", "assets")))
version_file = os.path.join(ROOT, "build", "version_info.txt")

excludes = ["vtkmodules", "vtk", "matplotlib", "IPython", "pytest", "PyQt5", "PySide6", "scipy", "pandas"]

a = Analysis(
    [os.path.join(SPECPATH, "gui_entry.py"), os.path.join(SPECPATH, "cli_entry.py")],
    pathex=[SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    runtime_hooks=[os.path.join(SPECPATH, "rthook_ocp.py")],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

common = dict(
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    icon=icon if os.path.isfile(icon) else None,
    version=version_file if os.path.isfile(version_file) else None,
)

gui_scripts = [s for s in a.scripts if s[0] not in ("cli_entry",)]
cli_scripts = [s for s in a.scripts if s[0] not in ("gui_entry",)]

exe_gui = EXE(pyz, gui_scripts, [], exclude_binaries=True, name="VCMX2STP", console=False, **common)
exe_cli = EXE(pyz, cli_scripts, [], exclude_binaries=True, name="vcmx2stp-cli", console=True, **common)

coll = COLLECT(exe_gui, exe_cli, a.binaries, a.datas, strip=False, upx=False, name="VCMX2STP")
