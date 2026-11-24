"""Чтение встроенной геометрии Visual Components (формат Autodesk 3DS + расширение VC 0x8001 для полилиний)."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MeshData:
    vertices: np.ndarray            # (N,3) float64
    triangles: np.ndarray           # (M,3) int64
    tri_material: np.ndarray        # (M,) int32 — индекс в materials (-1 — нет)
    materials: List[str] = field(default_factory=list)
    polylines: List[np.ndarray] = field(default_factory=list)  # список (K,3)
    colors: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return len(self.triangles) == 0 and not self.polylines

    def dominant_color(self) -> Optional[Tuple[float, float, float]]:
        if len(self.tri_material) == 0 or not self.materials:
            return None
        idx = self.tri_material[self.tri_material >= 0]
        if idx.size == 0:
            return None
        best = int(np.bincount(idx).argmax())
        return self.colors.get(self.materials[best])


M_MAIN, M_EDIT, M_OBJ, M_TRI, M_VERT, M_FACE, M_FMAT, M_MAT = 0x4D4D, 0x3D3D, 0x4000, 0x4100, 0x4110, 0x4120, 0x4130, 0xAFFF
M_MATNAME, M_DIFFUSE, M_RGBF, M_RGBB, M_RGBB_G, M_RGBF_G = 0xA000, 0xA020, 0x0010, 0x0011, 0x0012, 0x0013
M_VC_LINES = 0x8001


def is_3ds(data: bytes) -> bool:
    return len(data) >= 6 and struct.unpack_from("<H", data, 0)[0] == M_MAIN


def _cstr(b: bytes, off: int) -> Tuple[str, int]:
    z = b.find(b"\0", off)
    if z < 0:
        z = len(b)
    raw = b[off:z]
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(enc), z + 1
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), z + 1  # pragma: no cover


def _chunks(b: bytes, off: int, end: int):
    while off + 6 <= end:
        cid, ln = struct.unpack_from("<HI", b, off)
        if ln < 6 or off + ln > end:
            return
        yield cid, off + 6, off + ln
        off += ln


def _color(b: bytes, body: int, end: int):
    for cid, s, e in _chunks(b, body, end):
        if cid in (M_RGBB, M_RGBB_G) and e - s >= 3:
            r, g, bl = b[s], b[s + 1], b[s + 2]
            return (r / 255.0, g / 255.0, bl / 255.0)
        if cid in (M_RGBF, M_RGBF_G) and e - s >= 12:
            return tuple(float(min(1.0, max(0.0, v))) for v in struct.unpack_from("<3f", b, s))
    return None


def _polylines(b: bytes, s: int, e: int) -> List[np.ndarray]:
    """Расширение VC: u16 флаги, u16 число полилиний, далее (u16 n, n*3 double)."""
    out = []
    if e - s < 4:
        return out
    count = struct.unpack_from("<H", b, s + 2)[0]
    off = s + 4
    for _ in range(count):
        if off + 2 > e:
            break
        n = struct.unpack_from("<H", b, off)[0]
        off += 2
        need = n * 24
        if off + need > e:
            break
        pts = np.frombuffer(b, dtype="<f8", count=n * 3, offset=off).reshape(n, 3).astype(float)
        off += need
        if n >= 2:
            out.append(pts)
    return out


def read_3ds(data: bytes) -> MeshData:
    b = data
    if not is_3ds(b):
        raise ValueError("не формат 3DS")
    verts_all: List[np.ndarray] = []
    tris_all: List[np.ndarray] = []
    tmat_all: List[np.ndarray] = []
    materials: List[str] = []
    mat_index: Dict[str, int] = {}
    colors: Dict[str, Tuple[float, float, float]] = {}
    polylines: List[np.ndarray] = []
    base = 0

    def mat_id(name: str) -> int:
        if name not in mat_index:
            mat_index[name] = len(materials)
            materials.append(name)
        return mat_index[name]

    _, ms, me = next(_chunks(b, 0, len(b)))
    for cid, s, e in _chunks(b, ms, me):
        if cid != M_EDIT:
            continue
        for cid2, s2, e2 in _chunks(b, s, e):
            if cid2 == M_MAT:
                name, col = None, None
                for cid3, s3, e3 in _chunks(b, s2, e2):
                    if cid3 == M_MATNAME:
                        name, _ = _cstr(b, s3)
                    elif cid3 == M_DIFFUSE:
                        col = _color(b, s3, e3)
                if name is not None and col is not None:
                    colors[name] = col
            elif cid2 == M_OBJ:
                _, sub = _cstr(b, s2)
                for cid3, s3, e3 in _chunks(b, sub, e2):
                    if cid3 == M_VC_LINES:
                        polylines.extend(_polylines(b, s3, e3))
                    if cid3 != M_TRI:
                        continue
                    v = None
                    f = None
                    fm = None
                    for cid4, s4, e4 in _chunks(b, s3, e3):
                        if cid4 == M_VERT:
                            n = struct.unpack_from("<H", b, s4)[0]
                            v = np.frombuffer(b, dtype="<f4", count=n * 3, offset=s4 + 2).reshape(n, 3).astype(float)
                        elif cid4 == M_FACE:
                            n = struct.unpack_from("<H", b, s4)[0]
                            raw = np.frombuffer(b, dtype="<u2", count=n * 4, offset=s4 + 2).reshape(n, 4)
                            f = raw[:, :3].astype(np.int64)
                            fm = np.full(n, -1, np.int32)
                            fe = s4 + 2 + n * 8
                            for cid5, s5, e5 in _chunks(b, fe, e4):
                                if cid5 == M_FMAT:
                                    mname, p = _cstr(b, s5)
                                    cnt = struct.unpack_from("<H", b, p)[0]
                                    idx = np.frombuffer(b, dtype="<u2", count=cnt, offset=p + 2).astype(np.int64)
                                    idx = idx[idx < n]
                                    fm[idx] = mat_id(mname)
                        elif cid4 == M_VC_LINES:
                            polylines.extend(_polylines(b, s4, e4))
                    if v is not None and f is not None and len(f):
                        ok = (f < len(v)).all(axis=1)
                        verts_all.append(v)
                        tris_all.append(f[ok] + base)
                        tmat_all.append(fm[ok])
                        base += len(v)
    if verts_all:
        V = np.vstack(verts_all)
        F = np.vstack(tris_all) if tris_all else np.zeros((0, 3), np.int64)
        FM = np.concatenate(tmat_all) if tmat_all else np.zeros(0, np.int32)
    else:
        V = np.zeros((0, 3))
        F = np.zeros((0, 3), np.int64)
        FM = np.zeros(0, np.int32)
    return MeshData(V, F, FM, materials, polylines, colors)
