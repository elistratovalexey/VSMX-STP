"""Определение формата встроенной геометрии и её чтение."""
from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

from .mesh3ds import MeshData, is_3ds, read_3ds


def sniff(data: bytes) -> str:
    head = data[:256]
    if is_3ds(data):
        return "3ds"
    if head.lstrip().startswith(b"ISO-10303-21"):
        return "step"
    if len(head) > 72 and head[72:73] in (b"S", b"G") and b"S      1" in data[:400]:
        return "iges"
    if head.startswith(b"solid") and b"facet" in data[:2048]:
        return "stl_ascii"
    if len(data) > 84:
        import struct
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + n * 50 == len(data):
            return "stl_bin"
    return "unknown"


def _with_temp(data: bytes, suffix: str, fn):
    fd, path = tempfile.mkstemp(suffix=suffix)  # путь только из ASCII — безопасно для OCCT
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return fn(path)
    finally:
        try:
            os.remove(path)
        except OSError:  # pragma: no cover
            pass


def read_step_bytes(data: bytes):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    def go(path):
        r = STEPControl_Reader()
        if r.ReadFile(path) != IFSelect_RetDone:
            return None
        r.TransferRoots()
        s = r.OneShape()
        return None if s.IsNull() else s

    return _with_temp(data, ".stp", go)


def read_iges_bytes(data: bytes):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.IGESControl import IGESControl_Reader

    def go(path):
        r = IGESControl_Reader()
        if r.ReadFile(path) != IFSelect_RetDone:
            return None
        r.TransferRoots()
        s = r.OneShape()
        return None if s.IsNull() else s

    return _with_temp(data, ".igs", go)


def read_stl_bytes(data: bytes) -> Optional[MeshData]:
    import struct

    import numpy as np

    kind = sniff(data)
    pts = []
    if kind == "stl_bin":
        n = struct.unpack_from("<I", data, 80)[0]
        arr = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
        pts = arr["v"].reshape(-1, 3).astype(float)
    else:
        txt = data.decode("ascii", errors="ignore").split()
        coords = []
        for i, t in enumerate(txt):
            if t == "vertex":
                coords.append([float(txt[i + 1]), float(txt[i + 2]), float(txt[i + 3])])
        pts = np.array(coords, float).reshape(-1, 3)
    if len(pts) < 3:
        return None
    tris = np.arange(len(pts) - len(pts) % 3).reshape(-1, 3)
    return MeshData(vertices=pts, triangles=tris, tri_material=np.full(len(tris), -1, np.int32))


def load(data: bytes) -> Tuple[str, object]:
    """Возвращает (тип, объект): 'mesh' -> MeshData, 'brep' -> TopoDS_Shape, 'none' -> причина."""
    k = sniff(data)
    try:
        if k == "3ds":
            return "mesh", read_3ds(data)
        if k == "step":
            s = read_step_bytes(data)
            return ("brep", s) if s is not None else ("none", "не удалось прочитать STEP")
        if k == "iges":
            s = read_iges_bytes(data)
            return ("brep", s) if s is not None else ("none", "не удалось прочитать IGES")
        if k.startswith("stl"):
            m = read_stl_bytes(data)
            return ("mesh", m) if m is not None else ("none", "пустой STL")
    except Exception as ex:  # pragma: no cover - защитный код
        return "none", f"ошибка чтения геометрии: {ex}"
    return "none", "неизвестный формат геометрии"
