"""Построение точной граничной модели (B-rep) без треугольников.

* Примитивы VC (Block, Cylinder, Cone, Sphere, Wedge) строятся аналитически
  средствами OpenCASCADE — плоскости, цилиндры, конусы, сферы.
* Сеточная геометрия (3DS) преобразуется в многогранник: соседние компланарные
  треугольники объединяются в одну плоскую грань (многоугольник с отверстиями),
  коллинеарные вершины на рёбрах удаляются, общие рёбра и вершины делятся между
  гранями, замкнутые оболочки превращаются в тела (SOLID).
  В STEP не записываются триангуляции (TRIANGULATED_FACE и т.п.) — только
  ADVANCED_FACE на плоскостях.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from OCP.BRep import BRep_Builder
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_GTransform,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Transform,
)
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakePrism,
    BRepPrimAPI_MakeRevol,
    BRepPrimAPI_MakeSphere,
)
from OCP.Geom import Geom_Plane
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_GTrsf, gp_Mat, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec, gp_XYZ
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL, TopAbs_SOLID, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Face, TopoDS_Shape, TopoDS_Shell, TopoDS_Solid, TopoDS_Wire
try:  # OCP 7.x
    from OCP.TopTools import TopTools_HSequenceOfShape
except ImportError:  # OCP 8.x
    from OCP.OCP.collections import HSequence_TopoDS_Shape as TopTools_HSequenceOfShape

_B = BRep_Builder()


def _downcast(kind: str):
    f = getattr(TopoDS, kind + "_s", None) or getattr(TopoDS, kind)
    return f


to_face = _downcast("Face")
to_wire = _downcast("Wire")
to_shell = _downcast("Shell")
to_solid = _downcast("Solid")
to_edge = _downcast("Edge")


# ============================================================ утилиты
def compound(shapes: Sequence[TopoDS_Shape]) -> TopoDS_Compound:
    c = TopoDS_Compound()
    _B.MakeCompound(c)
    for s in shapes:
        if s is not None and not s.IsNull():
            _B.Add(c, s)
    return c


def is_rigid(m: np.ndarray, tol: float = 1e-6) -> bool:
    r = m[:3, :3]
    return bool(np.allclose(r.T @ r, np.eye(3), atol=tol) and np.linalg.det(r) > 0)


def trsf_from_matrix(m: np.ndarray) -> gp_Trsf:
    t = gp_Trsf()
    r = m[:3, :3]
    # ортонормализация (матрицы в файлах VC хранятся с ~1e-12 погрешностью)
    u, _s, vt = np.linalg.svd(r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        r = -r
    t.SetValues(
        float(r[0, 0]), float(r[0, 1]), float(r[0, 2]), float(m[0, 3]),
        float(r[1, 0]), float(r[1, 1]), float(r[1, 2]), float(m[1, 3]),
        float(r[2, 0]), float(r[2, 1]), float(r[2, 2]), float(m[2, 3]),
    )
    return t


def apply_matrix(shape: TopoDS_Shape, m: np.ndarray) -> TopoDS_Shape:
    """Применяет к форме произвольную аффинную матрицу 4x4 (с запеканием)."""
    if np.allclose(m, np.eye(4), atol=1e-12):
        return shape
    a = m[:3, :3]
    if is_rigid(m):
        return BRepBuilderAPI_Transform(shape, trsf_from_matrix(m), True).Shape()
    # равномерный масштаб * поворот — сохраняет аналитические поверхности
    sv = np.linalg.svd(a, compute_uv=False)
    det = np.linalg.det(a)
    if np.allclose(sv, sv[0], rtol=1e-9) and det > 0:
        s = float(sv[0])
        rot = a / s
        t = gp_Trsf()
        t.SetValues(
            float(rot[0, 0]) * s, float(rot[0, 1]) * s, float(rot[0, 2]) * s, float(m[0, 3]),
            float(rot[1, 0]) * s, float(rot[1, 1]) * s, float(rot[1, 2]) * s, float(m[1, 3]),
            float(rot[2, 0]) * s, float(rot[2, 1]) * s, float(rot[2, 2]) * s, float(m[2, 3]),
        )
        return BRepBuilderAPI_Transform(shape, t, True).Shape()
    # ортогональная матрица с отрицательным определителем (зеркало) * равномерный масштаб
    if np.allclose(sv, sv[0], rtol=1e-9) and det < 0:
        s = float(sv[0])
        rot = (a / s) @ np.diag([1.0, 1.0, -1.0])     # собственный поворот
        mir = gp_Trsf()
        mir.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)))
        t = gp_Trsf()
        t.SetValues(
            float(rot[0, 0]) * s, float(rot[0, 1]) * s, float(rot[0, 2]) * s, float(m[0, 3]),
            float(rot[1, 0]) * s, float(rot[1, 1]) * s, float(rot[1, 2]) * s, float(m[1, 3]),
            float(rot[2, 0]) * s, float(rot[2, 1]) * s, float(rot[2, 2]) * s, float(m[2, 3]),
        )
        return BRepBuilderAPI_Transform(shape, t.Multiplied(mir), True).Shape()
    poly = _affine_polyhedral(shape, m)
    if poly is not None:
        return poly
    g = gp_GTrsf()
    g.SetVectorialPart(gp_Mat(*[float(x) for x in a.reshape(-1)]))
    g.SetTranslationPart(gp_XYZ(float(m[0, 3]), float(m[1, 3]), float(m[2, 3])))
    return BRepBuilderAPI_GTransform(shape, g, True).Shape()


def _affine_polyhedral(shape: TopoDS_Shape, m: np.ndarray) -> Optional[TopoDS_Shape]:
    """Аффинное преобразование многогранника с сохранением плоских граней и прямых рёбер
    (BRepBuilderAPI_GTransform превратил бы их в B-сплайны)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.GeomAbs import GeomAbs_Line, GeomAbs_Plane

    flip = np.linalg.det(m[:3, :3]) < 0
    pts: Dict[tuple, int] = {}
    coords: List[np.ndarray] = []
    groups = []  # (индекс оболочки/тела, [контуры]) по граням

    def vid(p):
        k = (round(p.X(), 7), round(p.Y(), 7), round(p.Z(), 7))
        i = pts.get(k)
        if i is None:
            i = len(coords)
            pts[k] = i
            q = m @ np.array([p.X(), p.Y(), p.Z(), 1.0])
            coords.append(q[:3])
        return i

    from OCP.BRep import BRep_Tool
    solids = []
    exs = TopExp_Explorer(shape, TopAbs_SHELL)
    shells = []
    while exs.More():
        shells.append(exs.Current())
        exs.Next()
    if not shells:
        return None
    for sh in shells:
        faces = []
        exf = TopExp_Explorer(sh, TopAbs_FACE)
        while exf.More():
            fc = to_face(exf.Current())
            if BRepAdaptor_Surface(fc).GetType() != GeomAbs_Plane:
                return None
            loops = []
            exw = TopExp_Explorer(fc, TopAbs_WIRE)
            while exw.More():
                wire = to_wire(exw.Current())
                we = BRepTools_WireExplorer(wire, fc)
                lp = []
                while we.More():
                    e = we.Current()
                    if BRepAdaptor_Curve(e).GetType() != GeomAbs_Line:
                        return None
                    lp.append(vid(BRep_Tool.Pnt_s(we.CurrentVertex())))
                    we.Next()
                if len(lp) >= 3:
                    loops.append(lp[::-1] if flip else lp)
                exw.Next()
            if loops:
                faces.append(loops)
            exf.Next()
        shells.append(None) if False else None
        solids.append((faces, sh.Closed()))
    v = np.array(coords)
    diag = float(np.linalg.norm(np.ptp(v, axis=0))) or 1.0
    tol = max(diag * 1e-7, 1e-7)
    vcache: Dict[int, object] = {}
    ecache: Dict[Tuple[int, int], object] = {}

    def gv(i):
        x = vcache.get(i)
        if x is None:
            x = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, v[i]))).Vertex()
            vcache[i] = x
        return x

    def ge(i, j):
        k = (i, j) if i < j else (j, i)
        e = ecache.get(k)
        if e is None:
            e = BRepBuilderAPI_MakeEdge(gv(k[0]), gv(k[1])).Edge()
            ecache[k] = e
        return e if i < j else e.Reversed()

    out = []
    for faces, closed in solids:
        sh = TopoDS_Shell()
        _B.MakeShell(sh)
        for loops in faces:
            outer = v[loops[0]]
            nrm = np.zeros(3)
            for a_, b_ in zip(outer, np.roll(outer, -1, axis=0)):
                nrm += np.cross(a_, b_)
            if np.linalg.norm(nrm) < 1e-300:
                continue
            nrm /= np.linalg.norm(nrm)
            plane = Geom_Plane(gp_Pln(gp_Pnt(*map(float, outer[0])), gp_Dir(*map(float, nrm))))
            face = TopoDS_Face()
            _B.MakeFace(face, plane, tol)
            for lp in loops:
                w = TopoDS_Wire()
                _B.MakeWire(w)
                for a_, b_ in zip(lp, lp[1:] + lp[:1]):
                    if a_ != b_:
                        _B.Add(w, ge(a_, b_))
                w.Closed(True)
                _B.Add(face, w)
            _B.Add(sh, face)
        if closed:
            sh.Closed(True)
            so = TopoDS_Solid()
            _B.MakeSolid(so)
            _B.Add(so, sh)
            out.append(so)
        else:
            out.append(sh)
    if not out:
        return None
    res = out[0] if len(out) == 1 else compound(out)
    return res if BRepCheck_Analyzer(res).IsValid() else None


def count_faces(shape: TopoDS_Shape) -> int:
    n = 0
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        n += 1
        ex.Next()
    return n


# ============================================================ примитивы
def make_block(L: float, W: float, H: float) -> Optional[TopoDS_Shape]:
    dims = [abs(L) > 1e-9, abs(W) > 1e-9, abs(H) > 1e-9]
    if sum(dims) == 2:  # блок нулевой толщины = прямоугольная грань (используется как профиль)
        p = [(0, 0, 0)]
        if not dims[0]:
            p = [(0, 0, 0), (0, W, 0), (0, W, H), (0, 0, H)]
        elif not dims[1]:
            p = [(0, 0, 0), (0, 0, H), (L, 0, H), (L, 0, 0)]
        else:
            p = [(0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0)]
        poly = BRepBuilderAPI_MakePolygon()
        for q in p:
            poly.Add(gp_Pnt(*map(float, q)))
        poly.Close()
        return BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()
    if L <= 1e-9 or W <= 1e-9 or H <= 1e-9:
        return None
    return BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), float(L), float(W), float(H)).Shape()


def _sweep(a0: float, a1: float) -> Tuple[float, float]:
    a0, a1 = float(a0), float(a1)
    if a1 < a0:
        a0, a1 = a1, a0
    return a0, min(a1, a0 + 360.0)


def make_cylinder(r: float, h: float, a0: float = 0.0, a1: float = 360.0) -> Optional[TopoDS_Shape]:
    if r <= 1e-9 or abs(h) <= 1e-9:
        return None
    a0, a1 = _sweep(a0, a1)
    ang = math.radians(a1 - a0)
    z0 = 0.0 if h > 0 else h
    ax = gp_Ax2(gp_Pnt(0, 0, z0), gp_Dir(0, 0, 1), gp_Dir(math.cos(math.radians(a0)), math.sin(math.radians(a0)), 0))
    if ang >= 2 * math.pi - 1e-9:
        return BRepPrimAPI_MakeCylinder(ax, float(r), abs(float(h))).Shape()
    return BRepPrimAPI_MakeCylinder(ax, float(r), abs(float(h)), ang).Shape()


def make_cone(r1: float, r2: float, h: float, a0: float = 0.0, a1: float = 360.0) -> Optional[TopoDS_Shape]:
    if abs(h) <= 1e-9 or (r1 <= 1e-9 and r2 <= 1e-9):
        return None
    if abs(r1 - r2) < 1e-9:
        return make_cylinder(r1, h, a0, a1)
    a0, a1 = _sweep(a0, a1)
    ang = math.radians(a1 - a0)
    ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(math.cos(math.radians(a0)), math.sin(math.radians(a0)), 0))
    r1, r2 = max(0.0, float(r1)), max(0.0, float(r2))
    if ang >= 2 * math.pi - 1e-9:
        return BRepPrimAPI_MakeCone(ax, r1, r2, float(h)).Shape()
    return BRepPrimAPI_MakeCone(ax, r1, r2, float(h), ang).Shape()


def make_sphere(r: float, a0: float = 0.0, a1: float = 360.0) -> Optional[TopoDS_Shape]:
    if r <= 1e-9:
        return None
    a0, a1 = _sweep(a0, a1)
    ang = math.radians(a1 - a0)
    ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(math.cos(math.radians(a0)), math.sin(math.radians(a0)), 0))
    if ang >= 2 * math.pi - 1e-9:
        return BRepPrimAPI_MakeSphere(ax, float(r)).Shape()
    return BRepPrimAPI_MakeSphere(ax, float(r), ang).Shape()


def make_wedge(bl: float, sl: float, bh: float, sh: float, w: float, mirror_back: bool, mirror_down: bool):
    """Клин: профиль в плоскости XZ (нижняя длина bl, верхняя sl, высота сзади bh, спереди sh), толщина w по Y."""
    if w <= 1e-9 or bl <= 1e-9 or max(bh, sh) <= 1e-9:
        return None
    pts = [(0.0, 0.0), (bl, 0.0)]
    if sh > 1e-9:
        pts.append((sl if sl > 1e-9 else bl, sh) if sl > 1e-9 else (bl, sh))
    pts.append((0.0, bh))
    # удаляем совпадающие точки
    clean = []
    for p in pts:
        if not clean or abs(p[0] - clean[-1][0]) + abs(p[1] - clean[-1][1]) > 1e-9:
            clean.append(p)
    if len(clean) < 3:
        return None
    y0 = -w if mirror_back else 0.0
    zmul = [1.0, -1.0] if mirror_down else [1.0]
    solids = []
    for zm in zmul:
        poly = BRepBuilderAPI_MakePolygon()
        for x, z in clean:
            poly.Add(gp_Pnt(x, y0, z * zm))
        poly.Close()
        face = BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()
        depth = (2 * w) if mirror_back else w
        solids.append(BRepPrimAPI_MakePrism(face, gp_Vec(0, depth, 0)).Shape())
    return solids[0] if len(solids) == 1 else compound(solids)


# ============================================================ полилинии
def polylines_to_shape(polys: List[np.ndarray], tol: float = 1e-3) -> Optional[TopoDS_Shape]:
    """Полилинии профиля -> замкнутые плоские грани (если возможно) или провода."""
    edges = TopTools_HSequenceOfShape()
    n = 0
    for p in polys:
        for i in range(len(p) - 1):
            a, b = p[i], p[i + 1]
            if np.linalg.norm(b - a) < 1e-9:
                continue
            e = BRepBuilderAPI_MakeEdge(gp_Pnt(*map(float, a)), gp_Pnt(*map(float, b)))
            if e.IsDone():
                edges.Append(e.Edge())
                n += 1
    if n == 0:
        return None
    wires = TopTools_HSequenceOfShape()
    ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(edges, tol, False, wires)
    faces, open_wires = [], []
    for i in range(1, wires.Length() + 1):
        w = to_wire(wires.Value(i))
        mf = BRepBuilderAPI_MakeFace(w, True)
        if w.Closed() and mf.IsDone():
            faces.append(mf.Face())
        else:
            open_wires.append(w)
    if faces:
        # вложенные контуры (отверстия) — вычитаются по правилу чёт/нечет
        faces = _nest_faces(faces)
    return compound(faces + open_wires)


def _nest_faces(faces: List[TopoDS_Face]) -> List[TopoDS_Face]:
    if len(faces) < 2:
        return faces
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp

    def area(f):
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(f, g)
        return abs(g.Mass())

    faces = sorted(faces, key=area, reverse=True)
    result = []
    used = [False] * len(faces)
    for i, f in enumerate(faces):
        if used[i]:
            continue
        cur = f
        for j in range(i + 1, len(faces)):
            if used[j]:
                continue
            try:
                cut = BRepAlgoAPI_Cut(cur, faces[j])
                if cut.IsDone():
                    a0 = area(cur)
                    newf = cut.Shape()
                    if area(newf) < a0 - 1e-9:
                        cur = newf
                        used[j] = True
            except Exception:  # pragma: no cover
                pass
        result.append(cur)
    out = []
    for r in result:
        ex = TopExp_Explorer(r, TopAbs_FACE)
        while ex.More():
            out.append(to_face(ex.Current()))
            ex.Next()
    return out


# ============================================================ сетка -> B-rep
class MeshStats:
    def __init__(self):
        self.triangles_in = 0
        self.faces_out = 0
        self.solids = 0
        self.shells = 0


def _weld(v: np.ndarray, f: np.ndarray, tol: float):
    q = np.round(v / tol).astype(np.int64)
    _, idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    v2 = v[idx]
    f2 = inv[f]
    return v2, f2


def _clean_tris(v: np.ndarray, f: np.ndarray, mats: Optional[np.ndarray]):
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[ok]
    if mats is not None:
        mats = mats[ok]
    if len(f) == 0:
        return f, mats
    cr = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    area2 = np.linalg.norm(cr, axis=1)
    scale = max(1e-12, float(np.ptp(v, axis=0).max()) if len(v) else 1.0)
    ok = area2 > (scale * 1e-9) ** 2
    f = f[ok]
    if mats is not None:
        mats = mats[ok]
    # дубли (в т.ч. двусторонние копии)
    key = np.sort(f, axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    first.sort()
    f = f[first]
    if mats is not None:
        mats = mats[first]
    return f, mats


def _orient_consistently(f: np.ndarray, nverts: int) -> Tuple[np.ndarray, List[np.ndarray], List[bool]]:
    """Согласует ориентацию треугольников, разбивает на связные компоненты.

    Возвращает (треугольники, список индексов треугольников по компонентам, признак замкнутости)."""
    m = len(f)
    f = f.copy()
    a = f[:, [0, 1, 2]].reshape(-1)
    b = f[:, [1, 2, 0]].reshape(-1)
    tri_of = np.repeat(np.arange(m), 3)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    ukey = lo.astype(np.int64) * nverts + hi
    order = np.argsort(ukey, kind="stable")
    sk = ukey[order]
    # группы рёбер
    starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    counts = np.diff(np.r_[starts, len(sk)])
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(m)]  # (сосед, id полуребра)
    edge_count_of_he = np.empty(len(sk), np.int64)
    for s, c in zip(starts, counts):
        edge_count_of_he[order[s:s + c]] = c
        if c == 2:
            h1, h2 = order[s], order[s + 1]
            t1, t2 = tri_of[h1], tri_of[h2]
            if t1 != t2:
                adj[t1].append((t2, h1, h2))
                adj[t2].append((t1, h2, h1))
    # BFS
    comp = np.full(m, -1, np.int64)
    flip = np.zeros(m, bool)
    comps: List[np.ndarray] = []
    for seed in range(m):
        if comp[seed] >= 0:
            continue
        cid = len(comps)
        comp[seed] = cid
        stack = [seed]
        members = [seed]
        while stack:
            t = stack.pop()
            for (nb, h_t, h_nb) in adj[t]:
                # направление полуребра с учётом уже выполненных переворотов
                same_dir = (a[h_t] == a[h_nb])  # исходные направления совпадают -> несогласованы
                want_flip = flip[t] ^ same_dir
                if comp[nb] < 0:
                    comp[nb] = cid
                    flip[nb] = want_flip
                    stack.append(nb)
                    members.append(nb)
        comps.append(np.array(members, np.int64))
    f[flip] = f[flip][:, ::-1]
    # замкнутость: каждое ребро компоненты используется ровно 2 раза
    closed = []
    he_count_by_tri = edge_count_of_he.reshape(-1)
    cnt_tri = np.empty((m, 3), np.int64)
    cnt_tri.reshape(-1)[:] = 0
    cnt_flat = np.zeros(3 * m, np.int64)
    cnt_flat[np.arange(3 * m)] = he_count_by_tri[np.argsort(order)] if False else 0
    # edge_count_of_he индексирован по исходному id полуребра
    for idx in comps:
        hes = (idx[:, None] * 3 + np.arange(3)[None, :]).reshape(-1)
        closed.append(bool(np.all(edge_count_of_he[hes] == 2)))
    return f, comps, closed


def _signed_volume(v: np.ndarray, f: np.ndarray) -> float:
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0)


def mesh_topology(vertices: np.ndarray, triangles: np.ndarray, angle_tol_deg: float = 0.5):
    """Сварка, очистка, ориентация, полурёбра и плоские области сетки (общая часть построителей)."""
    if vertices is None or len(triangles) == 0:
        return None
    v = np.asarray(vertices, float)
    f = np.asarray(triangles, np.int64)
    diag = float(np.linalg.norm(np.ptp(v, axis=0))) or 1.0
    weld_tol = max(diag * 1e-7, 1e-6)
    v, f = _weld(v, f, weld_tol)
    f, _ = _clean_tris(v, f, None)
    if len(f) == 0:
        return None
    nv = len(v)
    f, comps, closed = _orient_consistently(f, nv)
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(p1 - p0, p2 - p0)
    area2 = np.linalg.norm(cr, axis=1)
    normals = cr / np.maximum(area2[:, None], 1e-300)
    cos_tol = math.cos(math.radians(angle_tol_deg))
    dist_tol = max(diag * 2e-6, 1e-5)
    geom_tol = max(dist_tol * 2.0, 1e-6)

    # полурёбра и их пары
    m = len(f)
    ha = f.reshape(-1)
    hb = f[:, [1, 2, 0]].reshape(-1)
    key = ha * nv + hb
    rkey = hb * nv + ha
    order = np.argsort(key)
    sk = key[order]
    pos = np.searchsorted(sk, rkey)
    pos_c = np.minimum(pos, len(sk) - 1)
    has = sk[pos_c] == rkey
    # уникальность ключей
    cnt_key = np.searchsorted(sk, key, side="right") - np.searchsorted(sk, key, side="left")
    cnt_r = np.where(has, np.searchsorted(sk, rkey, side="right") - pos, 0)
    twin = np.where(has & (cnt_key == 1) & (cnt_r == 1), order[pos_c], -1)
    twin_tri = np.where(twin >= 0, twin // 3, -1)

    # рост регионов (компланарные треугольники)
    region = np.full(m, -1, np.int64)
    reg_normal: List[np.ndarray] = []
    reg_point: List[np.ndarray] = []
    seeds = np.argsort(-area2)
    for s in seeds:
        if region[s] >= 0:
            continue
        rid = len(reg_normal)
        n0 = normals[s]
        d0 = float(np.dot(n0, p0[s]))
        region[s] = rid
        stack = [s]
        while stack:
            t = stack.pop()
            for k in range(3):
                nb = twin_tri[t * 3 + k]
                if nb < 0 or region[nb] >= 0:
                    continue
                if float(np.dot(normals[nb], n0)) < cos_tol:
                    continue
                q = v[f[nb]]
                if np.max(np.abs(q @ n0 - d0)) > dist_tol:
                    continue
                region[nb] = rid
                stack.append(nb)
        reg_normal.append(n0)
        reg_point.append(p0[s])

    return dict(v=v, f=f, nv=nv, comps=comps, closed=closed, normals=normals, area2=area2, diag=diag,
                dist_tol=dist_tol, geom_tol=geom_tol, ha=ha, hb=hb, twin=twin, twin_tri=twin_tri,
                region=region, reg_normal=reg_normal, reg_point=reg_point, m=m)


def mesh_to_brep(vertices: np.ndarray, triangles: np.ndarray, angle_tol_deg: float = 0.5,
                 stats: Optional[MeshStats] = None, log=None, faces_only: bool = False) -> Optional[TopoDS_Shape]:
    """Преобразует треугольную сетку в B-rep из плоских многоугольных граней."""
    T = mesh_topology(vertices, triangles, angle_tol_deg)
    if T is None:
        return None
    v, f, nv, comps, closed = T["v"], T["f"], T["nv"], T["comps"], T["closed"]
    normals, area2, diag = T["normals"], T["area2"], T["diag"]
    dist_tol, geom_tol, ha, hb = T["dist_tol"], T["geom_tol"], T["ha"], T["hb"]
    twin, twin_tri, region, reg_normal, reg_point, m = (T["twin"], T["twin_tri"], T["region"], T["reg_normal"],
                                                         T["reg_point"], T["m"])
    p0 = v[f[:, 0]]
    if stats:
        stats.triangles_in += len(f)

    # граничные полурёбра регионов
    he_region = np.repeat(region, 3)
    twin_region = np.where(twin >= 0, he_region[np.maximum(twin, 0)], -1)
    is_bnd = twin_region != he_region
    bnd_ids = np.flatnonzero(is_bnd)

    # глобальный граф границ — для удаления коллинеарных вершин
    ukey = np.minimum(ha, hb) * nv + np.maximum(ha, hb)
    edge_regions: Dict[int, set] = defaultdict(set)
    for h in bnd_ids:
        edge_regions[int(ukey[h])].add(int(he_region[h]))
    vert_edges: Dict[int, List[int]] = defaultdict(list)
    for ek in edge_regions:
        a_, b_ = divmod(ek, nv)
        vert_edges[a_].append(ek)
        vert_edges[b_].append(ek)
    removable = np.zeros(nv, bool)
    for vid, eks in vert_edges.items():
        if len(eks) != 2 or edge_regions[eks[0]] != edge_regions[eks[1]]:
            continue
        o = []
        for ek in eks:
            a_, b_ = divmod(ek, nv)
            o.append(b_ if a_ == vid else a_)
        d1 = v[vid] - v[o[0]]
        d2 = v[o[1]] - v[vid]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        if np.linalg.norm(np.cross(d1, d2)) / (n1 * n2) < 1e-7 and np.dot(d1, d2) > 0:
            # дополнительно — отклонение от прямой в абсолютных единицах
            seg = v[o[1]] - v[o[0]]
            dist = np.linalg.norm(np.cross(seg, v[vid] - v[o[0]])) / max(np.linalg.norm(seg), 1e-300)
            if dist < geom_tol * 0.5:
                removable[vid] = True

    # контуры регионов
    by_region: Dict[int, List[int]] = defaultdict(list)
    for h in bnd_ids:
        by_region[int(he_region[h])].append(int(h))

    occ_vert: Dict[int, object] = {}
    occ_edge: Dict[Tuple[int, int], object] = {}

    def get_vertex(i: int):
        vv = occ_vert.get(i)
        if vv is None:
            vv = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, v[i]))).Vertex()
            _B.UpdateVertex(vv, geom_tol)
            occ_vert[i] = vv
        return vv

    def get_edge(i: int, j: int):
        k = (i, j) if i < j else (j, i)
        e = occ_edge.get(k)
        if e is None:
            mk = BRepBuilderAPI_MakeEdge(get_vertex(k[0]), get_vertex(k[1]))
            if not mk.IsDone():
                return None
            e = mk.Edge()
            _B.UpdateEdge(e, geom_tol)
            occ_edge[k] = e
        return e if i < j else e.Reversed()

    faces_by_comp: Dict[int, List[TopoDS_Face]] = defaultdict(list)
    comp_of_tri = np.empty(m, np.int64)
    for ci, idx in enumerate(comps):
        comp_of_tri[idx] = ci
    reg_comp = {}
    for t in range(m):
        reg_comp.setdefault(int(region[t]), int(comp_of_tri[t]))

    bad_regions: set = set()
    for rid, hes in by_region.items():
        n0 = reg_normal[rid]
        ax_u = np.cross(n0, [1.0, 0, 0] if abs(n0[0]) < 0.9 else [0, 1.0, 0])
        ax_u /= np.linalg.norm(ax_u)
        ax_v = np.cross(n0, ax_u)
        out_map: Dict[int, List[int]] = defaultdict(list)
        for h in hes:
            out_map[int(ha[h])].append(h)
        used = set()
        loops: List[List[int]] = []
        for h0 in hes:
            if h0 in used:
                continue
            loop = []
            h = h0
            guard = 0
            while h not in used and guard < len(hes) + 5:
                guard += 1
                used.add(h)
                loop.append(int(ha[h]))
                nxt_v = int(hb[h])
                cands = [c for c in out_map.get(nxt_v, []) if c not in used]
                if not cands:
                    break
                if len(cands) > 1:
                    din = v[nxt_v] - v[int(ha[h])]
                    rin = -np.array([din @ ax_u, din @ ax_v])
                    best, best_ang = cands[0], 1e9
                    for c in cands:
                        dc = v[int(hb[c])] - v[nxt_v]
                        dd = np.array([dc @ ax_u, dc @ ax_v])
                        ang = (math.atan2(rin[1], rin[0]) - math.atan2(dd[1], dd[0])) % (2 * math.pi)
                        if ang < 1e-12:
                            ang = 2 * math.pi
                        if ang < best_ang:
                            best, best_ang = c, ang
                    h = best
                else:
                    h = cands[0]
            if len(loop) >= 3:
                loops.append(loop)
        if not loops:
            continue
        # удаляем коллинеарные вершины
        simp = []
        for lp in loops:
            kept = [x for x in lp if not removable[x]]
            simp.append(kept if len(kept) >= 3 else lp)
        # площадь со знаком в плоскости региона
        info = []
        for lp in simp:
            pts = v[lp]
            uu, vv_ = pts @ ax_u, pts @ ax_v
            ar = 0.5 * float(np.dot(uu, np.roll(vv_, -1)) - np.dot(np.roll(uu, -1), vv_))
            info.append((lp, ar, np.column_stack([uu, vv_])))
        outers = [x for x in info if x[1] > 0]
        holes = [x for x in info if x[1] <= 0]
        if not outers:
            outers, holes = [max(info, key=lambda x: abs(x[1]))], [x for x in info if x is not max(info, key=lambda x: abs(x[1]))]
        assign: Dict[int, List] = defaultdict(list)
        for hl in holes:
            p = hl[2][0]
            target = 0
            if len(outers) > 1:
                for oi, ot in enumerate(outers):
                    if _point_in_poly(p, ot[2]):
                        target = oi
                        break
            assign[target].append(hl)
        plane = Geom_Plane(gp_Pln(gp_Pnt(*map(float, reg_point[rid])), gp_Dir(*map(float, n0))))
        for oi, ot in enumerate(outers):
            face = TopoDS_Face()
            _B.MakeFace(face, plane, geom_tol)
            okf = True
            for lp in [ot[0]] + [h[0] for h in assign.get(oi, [])]:
                w = TopoDS_Wire()
                _B.MakeWire(w)
                for i in range(len(lp)):
                    e = get_edge(lp[i], lp[(i + 1) % len(lp)])
                    if e is None:
                        okf = False
                        break
                    _B.Add(w, e)
                w.Closed(True)
                _B.Add(face, w)
            if okf and BRepCheck_Analyzer(face, False).IsValid():
                faces_by_comp[reg_comp[rid]].append(face)
            else:
                bad_regions.add(rid)

    # регионы, для которых не удалось построить корректную грань-многоугольник,
    # строятся по исходным плоским треугольным фасетам (это B-rep грани, не триангуляция)
    if bad_regions:
        tri_ids = np.flatnonzero(np.isin(region, list(bad_regions)))
        for t in tri_ids:
            a_, b_, c_ = (int(x) for x in f[t])
            n_t = normals[t]
            plane = Geom_Plane(gp_Pln(gp_Pnt(*map(float, v[a_])), gp_Dir(*map(float, n_t))))
            face = TopoDS_Face()
            _B.MakeFace(face, plane, geom_tol)
            w = TopoDS_Wire()
            _B.MakeWire(w)
            okf = True
            for i, j in ((a_, b_), (b_, c_), (c_, a_)):
                e = get_edge(i, j)
                if e is None:
                    okf = False
                    break
                _B.Add(w, e)
            if not okf:
                continue
            w.Closed(True)
            _B.Add(face, w)
            faces_by_comp[int(comp_of_tri[t])].append(face)
        if log:
            log(f"  {len(bad_regions)} сложных плоских областей разбито на фасеты")

    if faces_only:  # профиль для выдавливания/вращения — набор граней
        allf = [fc for lst in faces_by_comp.values() for fc in lst]
        return compound(allf) if allf else None
    result = []
    for ci, idx in enumerate(comps):
        fl = faces_by_comp.get(ci, [])
        if not fl:
            continue
        sh = TopoDS_Shell()
        _B.MakeShell(sh)
        for fc in fl:
            _B.Add(sh, fc)
        if closed[ci]:
            sh.Closed(True)
            vol = _signed_volume(v, f[idx])
            if vol < 0:
                sh = to_shell(sh.Reversed())
            so = TopoDS_Solid()
            _B.MakeSolid(so)
            _B.Add(so, sh)
            result.append(so)
            if stats:
                stats.solids += 1
        else:
            result.append(sh)
            if stats:
                stats.shells += 1
        if stats:
            stats.faces_out += len(fl)
    if not result:
        return None
    return result[0] if len(result) == 1 else compound(result)


def _point_in_poly(p, poly) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / ((y2 - y1) or 1e-300)
            if xi > x:
                inside = not inside
    return inside


# ============================================================ операции
def _profile_faces(shape: TopoDS_Shape, d: np.ndarray) -> TopoDS_Shape:
    """Тела в профиле заменяются их «задней» гранью, перпендикулярной направлению выдавливания."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane

    has_solid = TopExp_Explorer(shape, TopAbs_SOLID).More() or TopExp_Explorer(shape, TopAbs_SHELL).More()
    if not has_solid:
        return shape
    out = []
    # одиночные грани/провода вне оболочек
    ex = TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SHELL)
    while ex.More():
        out.append(ex.Current())
        ex.Next()
    exs = TopExp_Explorer(shape, TopAbs_SHELL)
    while exs.More():
        best = []
        exf = TopExp_Explorer(exs.Current(), TopAbs_FACE)
        cands = []
        while exf.More():
            fc = to_face(exf.Current())
            ad = BRepAdaptor_Surface(fc)
            if ad.GetType() == GeomAbs_Plane:
                pl = ad.Plane()
                nn = pl.Axis().Direction()
                nv = np.array([nn.X(), nn.Y(), nn.Z()])
                if abs(abs(float(nv @ d)) - 1.0) < 1e-6:
                    o = pl.Location()
                    cands.append((float(np.array([o.X(), o.Y(), o.Z()]) @ d), fc))
            exf.Next()
        if cands:
            mn = min(c[0] for c in cands)
            best = [c[1] for c in cands if abs(c[0] - mn) < 1e-6]
        out.extend(best)
        exs.Next()
    return compound(out) if out else shape


def extrude(shape: TopoDS_Shape, direction: np.ndarray, length: float) -> Optional[TopoDS_Shape]:
    d = np.asarray(direction, float)
    n = np.linalg.norm(d)
    if n < 1e-12 or abs(length) < 1e-12:
        return None
    d = d / n
    shape = _profile_faces(shape, d)
    return BRepPrimAPI_MakePrism(shape, gp_Vec(*map(float, d * length)), True).Shape()


def revolve(shape: TopoDS_Shape, axis: np.ndarray, angle_deg: float, origin=(0.0, 0.0, 0.0)) -> Optional[TopoDS_Shape]:
    a = np.asarray(axis, float)
    if np.linalg.norm(a) < 1e-12 or abs(angle_deg) < 1e-9:
        return None
    ax = gp_Ax1(gp_Pnt(*map(float, origin)), gp_Dir(*map(float, a)))
    ang = math.radians(min(abs(angle_deg), 360.0)) * (1 if angle_deg > 0 else -1)
    if abs(abs(angle_deg) - 360.0) < 1e-9:
        return BRepPrimAPI_MakeRevol(shape, ax, True).Shape()
    return BRepPrimAPI_MakeRevol(shape, ax, ang, True).Shape()


def mirror_matrix(normal: np.ndarray) -> np.ndarray:
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    m = np.eye(4)
    m[:3, :3] = np.eye(3) - 2 * np.outer(n, n)
    return m
