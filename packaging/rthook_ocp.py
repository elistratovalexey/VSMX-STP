# Runtime-hook PyInstaller: регистрирует папки с DLL OpenCascade/VTK до импорта OCP.
import os
import sys

_base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
for _d in ("cadquery_ocp.libs", "vtk.libs", ""):
    _p = os.path.join(_base, _d)
    if os.path.isdir(_p) and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_p)
        except OSError:
            pass
os.environ["PATH"] = os.pathsep.join(
    [os.path.join(_base, "cadquery_ocp.libs"), os.path.join(_base, "vtk.libs"), os.environ.get("PATH", "")])
