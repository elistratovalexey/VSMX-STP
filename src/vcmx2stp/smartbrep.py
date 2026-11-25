"""Сетка -> B-rep с распознаванием поверхностей (основной построитель для 3DS-геометрии).

В отличие от «многогранного» построителя (brep.mesh_to_brep), где каждый треугольник
криволинейного участка становится отдельной плоской гранью, здесь:

  * плоские области      -> одна плоская грань (как и раньше);
  * цилиндр/конус/сфера/тор -> одна грань на точной аналитической поверхности;
  * прочие гладкие участки -> одна грань на B-сплайновой поверхности (отклонение
    от вершин исходной сетки не больше допуска);
  * границы между гранями -> одно ребро на всю цепочку (прямая, дуга окружности или
    B-сплайн), а не десятки коротких отрезков.

В STEP по-прежнему нет треугольников — только ADVANCED_FACE (B-rep). Если для
какого-то участка не удаётся построить корректную грань, он строится плоскими
фасетами; при общей ошибке вызывающий код использует многогранный построитель.
"""
from __future__ import annotations

import math
import os as _os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
_PF = 4.0   # предел числа полюсов B-сплайна: не больше 4 на вершину сетки участка

from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepLib import BRepLib
from OCP.GC import GC_MakeArcOfCircle
from OCP.Geom import (Geom_BSplineCurve, Geom_BSplineSurface, Geom_ConicalSurface, Geom_CylindricalSurface,
                      Geom_Plane, Geom_SphericalSurface, Geom_ToroidalSurface)
from OCP.Geom2d import Geom2d_BSplineCurve
from OCP.GeomAbs import GeomAbs_C2
from OCP.GeomAPI import GeomAPI_PointsToBSpline
from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt, gp_Pnt2d
from OCP.ShapeBuild import ShapeBuild_ReShape
from OCP.ShapeFix import ShapeFix_Face
from OCP.OCP.collections import (Array1_double as TColStd_Array1OfReal, Array1_gp_Pnt as TColgp_Array1OfPnt,
                                 Array1_gp_Pnt2d as TColgp_Array1OfPnt2d, Array1_int as TColStd_Array1OfInteger,
                                 Array2_gp_Pnt as TColgp_Array2OfPnt)
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shell, TopoDS_Solid, TopoDS_Wire

from . import brep as _brep
from .surfrec import HeightPatch, Surface, fit_height_patch, soft_points, segment, segment_freeform

_B = BRep_Builder()


class FitStats:
    def __init__(self):
        self.planar = 0
        self.analytic = 0
        self.freeform = 0
        self.facets = 0
        self.tris = {"plane": 0, "ana": 0, "bsp": 0, "facet": 0}


# ---------------------------------------------------------------------------
# вспомогательные построения OCCT
# ---------------------------------------------------------------------------
def _arr1_pnt(P: np.ndarray) -> TColgp_Array1OfPnt:
    a = TColgp_Array1OfPnt(1, len(P))
    for i, p in enumerate(P):
        a.SetValue(i + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    return a


def _polyline_bspline(P: np.ndarray) -> Geom_BSplineCurve:
    """Ломаная как B-сплайн 1-й степени (точно проходит через все точки)."""
    n = len(P)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    t = np.r_[0.0, np.cumsum(np.maximum(seg, 1e-9))]
    knots = TColStd_Array1OfReal(1, n)
    mults = TColStd_Array1OfInteger(1, n)
    for i in range(n):
        knots.SetValue(i + 1, float(t[i]))
        mults.SetValue(i + 1, 2 if i in (0, n - 1) else 1)
    return Geom_BSplineCurve(_arr1_pnt(P), knots, mults, 1)


def _curve_poles(c: Geom_BSplineCurve) -> np.ndarray:
    n = c.NbPoles()
    out = np.empty((n, 3))
    for i in range(1, n + 1):
        p = c.Pole(i)
        out[i - 1] = (p.X(), p.Y(), p.Z())
    return out


def _bspline_through(P: np.ndarray, tol: float) -> Optional[Geom_BSplineCurve]:
    try:
        ap = GeomAPI_PointsToBSpline(_arr1_pnt(P), 2, 8, GeomAbs_C2, tol * 0.5)
        if not ap.IsDone():
            return None
        c = ap.Curve()
    except Exception:
        return None
    # кривая не должна «гулять» от ломаной сетки
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    lim = max(tol, 0.1 * float(seg.max()))
    ts = np.linspace(c.FirstParameter(), c.LastParameter(), 8 * len(P))
    S = np.array([[q.X(), q.Y(), q.Z()] for q in (c.Value(float(t)) for t in ts)])
    A_, B_ = P[:-1], P[1:]
    AB = B_ - A_
    L2 = np.maximum((AB * AB).sum(1), 1e-18)
    tt = np.clip(np.einsum("sij,ij->si", S[:, None, :] - A_[None], AB) / L2, 0, 1)
    D = np.linalg.norm(S[:, None, :] - (A_[None] + tt[..., None] * AB[None]), axis=2).min(1)
    if D.max() > lim:
        return None
    p0, p1 = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
    if (np.linalg.norm(np.array([p0.X(), p0.Y(), p0.Z()]) - P[0]) > tol * 0.05 or
            np.linalg.norm(np.array([p1.X(), p1.Y(), p1.Z()]) - P[-1]) > tol * 0.05):
        return None
    return c


def _fit_circle3d(P: np.ndarray, tol: float):
    c0 = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c0).T))
    n = V[:, 0]
    if np.max(np.abs((P - c0) @ n)) > tol:
        return None
    u = V[:, 2]
    vv = np.cross(n, u)
    x, y = (P - c0) @ u, (P - c0) @ vv
    A = np.column_stack([x, y, np.ones_like(x)])
    try:
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    if np.max(np.abs(np.hypot(x - cx, y - cy) - r)) > tol:
        return None
    # угловая монотонность (дуга, а не «зигзаг»)
    ang = np.unwrap(np.arctan2(y - cy, x - cx))
    dif = np.diff(ang)
    if not (np.all(dif > 0) or np.all(dif < 0)) or abs(ang[-1] - ang[0]) > 2 * math.pi - 1e-3:
        return None
    return c0 + cx * u + cy * vv, r


# ---------------------------------------------------------------------------
def mesh_to_brep_fit(vertices: np.ndarray, triangles: np.ndarray, tol: float = 0.1, angle_tol_deg: float = 0.5,
                     stats=None, fstats: Optional[FitStats] = None, log=None, freeform: bool = True):
    """Возвращает форму OCCT или None (тогда используется многогранный построитель)."""
    T = _brep.mesh_topology(vertices, triangles, angle_tol_deg)
    if T is None:
        return None
    v, f, m = T["v"], T["f"], T["m"]
    if m < 8:
        return None
    normals, twin_tri = T["normals"], T["twin_tri"]
    region = T["region"]
    tol = max(tol, T["diag"] * 1e-6)

    # 1) аналитические поверхности, 2) свободные формы
    surf_of, surfaces = segment(v, f, normals, twin_tri, region, tol)
    sizes = np.bincount(region)
    cand = sizes[region] <= 4
    patches = [] if not freeform or _os.environ.get('VCMX2STP_NO_FREEFORM') else segment_freeform(v, f, normals, twin_tri, cand & (surf_of < 0), tol)

    label = region.copy()
    desc: Dict[int, tuple] = {}
    for rid in range(len(T["reg_normal"])):
        desc[rid] = ("plane", T["reg_normal"][rid], T["reg_point"][rid])
    nxt = len(T["reg_normal"])
    for sid, s in enumerate(surfaces):
        idx = np.flatnonzero(surf_of == sid)
        if len(idx):
            label[idx] = nxt
            desc[nxt] = ("ana", s)
            nxt += 1
    for tris, hp in patches:
        label[tris] = nxt
        desc[nxt] = ("bsp", hp)
        nxt += 1

    for attempt in range(8):
        res = _build(T, label, desc, tol)
        if res is None:
            return None
        shape, bad, counts = res
        if not bad:
            if fstats is not None:
                fstats.planar += counts["plane"]
                fstats.analytic += counts["ana"]
                fstats.freeform += counts["bsp"]
                fstats.facets += counts["facet"]
                kinds = np.array([desc[int(L)][0] for L in label])
                for k in fstats.tris:
                    fstats.tris[k] += int((kinds == k).sum())
            if stats is not None:
                stats.triangles_in += m
                stats.faces_out += sum(counts.values())
            return shape
        # неудачные участки: свободные формы — делим пополам и пробуем снова, остальное — плоские фасеты
        for L in bad:
            idx = np.flatnonzero(label == L)
            d = desc.get(L)
            if d is not None and d[0] == "bsp" and (len(d) < 3 or d[2] < 3) and len(idx) >= 12:
                depth = d[2] if len(d) > 2 else 0
                parts = _split_patch(v, f, normals, twin_tri, idx)
                rest = []
                for part in parts:
                    hp = None
                    if len(part) >= 6:
                        cr = np.cross(v[f[part, 1]] - v[f[part, 0]], v[f[part, 2]] - v[f[part, 0]])
                        W = cr.sum(0)
                        W = W / max(np.linalg.norm(W), 1e-12)
                        Nt = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-12)
                        Q, qt = soft_points(v, f, part, tol)
                        Pp = v[np.unique(f[part].reshape(-1))]
                        hp = fit_height_patch(Pp, Nt, tol, W=W, Q=Q, qtol=qt, max_poles=max(16, int(_PF * len(Pp))))
                    if hp is None:
                        rest.extend(part.tolist())
                        continue
                    label[part] = nxt
                    desc[nxt] = ("bsp", hp, depth + 1)
                    nxt += 1
                desc.pop(L, None)
                idx = np.array(rest, np.int64)
            for t in idx:
                label[t] = nxt
                desc[nxt] = ("facet", normals[t], v[f[t, 0]])
                nxt += 1
            desc.pop(L, None)
    return None


def _split_patch(v, f, normals, twin_tri, tris):
    """Делит участок пополам по наибольшей протяжённости; возвращает связные части."""
    cen = v[f[tris]].mean(1)
    w, V = np.linalg.eigh(np.cov(cen.T))
    key = cen @ V[:, -1]
    med = np.median(key)
    nb = twin_tri.reshape(-1, 3)
    out = []
    for half in (tris[key <= med], tris[key > med]):
        S = set(int(t) for t in half)
        seen = set()
        for t0 in half:
            t0 = int(t0)
            if t0 in seen:
                continue
            seen.add(t0)
            st, mem = [t0], [t0]
            while st:
                t = st.pop()
                for u in nb[t]:
                    u = int(u)
                    if u in S and u not in seen:
                        seen.add(u)
                        st.append(u)
                        mem.append(u)
            out.append(np.array(mem, np.int64))
    return out


def _fail(where):
    if _os.environ.get('VCMX2STP_DEBUG'):
        print('smartbrep:', where)
    return None


def _build(T, label: np.ndarray, desc: Dict[int, tuple], tol: float):
    v, f, nv, m = T["v"], T["f"], T["nv"], T["m"]
    normals, twin = T["normals"], T["twin"]
    ha, hb = T["ha"], T["hb"]
    comps, closed = T["comps"], T["closed"]

    he_label = np.repeat(label, 3)
    twin_label = np.where(twin >= 0, he_label[np.maximum(twin, 0)], -1)
    is_bnd = twin_label != he_label
    bnd = np.flatnonzero(is_bnd)
    ukey = np.minimum(ha, hb) * nv + np.maximum(ha, hb)

    edge_labels: Dict[int, set] = defaultdict(set)
    for h in bnd:
        edge_labels[int(ukey[h])].add(int(he_label[h]))
    vert_edges: Dict[int, List[int]] = defaultdict(list)
    for ek in edge_labels:
        a_, b_ = divmod(ek, nv)
        vert_edges[a_].append(ek)
        vert_edges[b_].append(ek)

    def lab(ek):
        return frozenset(edge_labels[ek])

    corner = set()
    # у мелких плоских граней (1–2 треугольника) каждое ребро сетки — отдельное прямое ребро
    cnt = np.bincount(label, minlength=int(label.max()) + 1)
    small = np.array([desc.get(L, ("",))[0] == "facet" or (desc.get(L, ("",))[0] == "plane" and cnt[L] <= 2)
                      for L in range(len(cnt))])
    corner.update(int(x) for x in np.unique(f[small[label]].reshape(-1)))
    for vid, eks in vert_edges.items():
        if len(eks) != 2 or lab(eks[0]) != lab(eks[1]):
            corner.add(vid)

    chains: List[List[int]] = []
    chain_lab: List[frozenset] = []
    edge_chain: Dict[int, Tuple[int, int]] = {}

    def other(ek, vid):
        a_, b_ = divmod(ek, nv)
        return b_ if a_ == vid else a_

    def walk(start, ek0):
        pts = [start]
        ek = ek0
        cur = start
        cid = len(chains)
        while True:
            nxt_v = other(ek, cur)
            edge_chain[ek] = (cid, len(pts) - 1)
            pts.append(nxt_v)
            if nxt_v in corner or nxt_v == start:
                break
            cand = [e for e in vert_edges[nxt_v] if e != ek and e not in edge_chain]
            if not cand:
                break
            ek = cand[0]
            cur = nxt_v
        chains.append(pts)
        chain_lab.append(lab(ek0))

    for c in corner:
        for ek in vert_edges[c]:
            if ek not in edge_chain:
                walk(c, ek)
    # циклы без углов (окружность отверстия и т.п.) — делим на две цепочки
    for ek in list(edge_labels.keys()):
        if ek in edge_chain:
            continue
        a_, _ = divmod(ek, nv)
        loop = [a_]
        cur, e = a_, ek
        guard = 0
        seen_e = []
        while guard < 10 ** 7:
            guard += 1
            seen_e.append(e)
            nx = other(e, cur)
            if nx == a_:
                break
            loop.append(nx)
            cand = [x for x in vert_edges[nx] if x != e]
            if not cand:
                return _fail(275)
            cur, e = nx, cand[0]
        mid = len(loop) // 2
        if mid < 1:
            return _fail(279)
        corner.add(loop[0])
        corner.add(loop[mid])
        walk(loop[0], seen_e[0])
        rest = [x for x in vert_edges[loop[mid]] if x not in edge_chain]
        if rest:
            walk(loop[mid], rest[0])
        if any(x not in edge_chain for x in seen_e):
            return _fail(287)

    # замкнутые цепочки (начало = конец) делим пополам
    for cid in range(len(chains)):
        pts = chains[cid]
        if pts[0] == pts[-1] and len(pts) >= 3:
            mid = len(pts) // 2
            a_pts, b_pts = pts[:mid + 1], pts[mid:]
            chains[cid] = a_pts
            nid = len(chains)
            chains.append(b_pts)
            chain_lab.append(chain_lab[cid])
            for j in range(len(b_pts) - 1):
                x, y = b_pts[j], b_pts[j + 1]
                edge_chain[min(x, y) * nv + max(x, y)] = (nid, j)
    for cid, pts in enumerate(chains):
        if pts[0] == pts[-1]:
            return _fail("closed chain")

    # ---- рёбра
    vtol = tol
    occ_v: Dict[int, object] = {}

    def V(i):
        x = occ_v.get(i)
        if x is None:
            x = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, v[i]))).Vertex()
            _B.UpdateVertex(x, vtol)
            occ_v[i] = x
        return x

    occ_e: List[object] = []
    e_curve: List[Optional[Geom_BSplineCurve]] = []
    for cid, pts in enumerate(chains):
        P = v[pts]
        need_bsp = any(desc.get(L, ("",))[0] == "bsp" for L in chain_lab[cid])
        curve = None
        bs = None
        if len(P) == 2 or np.max(np.linalg.norm(np.cross(P - P[0], P[-1] - P[0]), axis=1)) / \
                max(np.linalg.norm(P[-1] - P[0]), 1e-12) <= tol * 0.5:
            if need_bsp:
                bs = _polyline_bspline(P[[0, -1]])
                curve = bs
        elif not need_bsp:
            cf = _fit_circle3d(P, tol * 0.5)
            if cf is not None:
                mid = len(P) // 2
                try:
                    arc = GC_MakeArcOfCircle(gp_Pnt(*map(float, P[0])), gp_Pnt(*map(float, P[mid])),
                                             gp_Pnt(*map(float, P[-1])))
                    if arc.IsDone():
                        curve = arc.Value()
                except Exception:
                    curve = None
            if curve is None:
                bs = _bspline_through(P, tol) or _polyline_bspline(P)
                curve = bs
        else:
            bs = _bspline_through(P, tol) or _polyline_bspline(P)
            curve = bs
        try:
            if curve is None:
                mk = BRepBuilderAPI_MakeEdge(V(pts[0]), V(pts[-1]))
            else:
                mk = BRepBuilderAPI_MakeEdge(curve, V(pts[0]), V(pts[-1]))
            if not mk.IsDone():
                bs = _polyline_bspline(P)
                mk = BRepBuilderAPI_MakeEdge(bs, V(pts[0]), V(pts[-1]))
                if not mk.IsDone():
                    return _fail(339)
        except Exception:
            return _fail(341)
        e = mk.Edge()
        _B.UpdateEdge(e, vtol)
        occ_e.append(e)
        e_curve.append(bs)

    # ---- контуры областей
    by_label: Dict[int, List[int]] = defaultdict(list)
    for h in bnd:
        by_label[int(he_label[h])].append(int(h))

    def trace(hes: List[int]) -> Optional[List[List[int]]]:
        out_map: Dict[int, List[int]] = defaultdict(list)
        for h in hes:
            out_map[int(ha[h])].append(h)
        used = set()
        loops = []
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
                nv_ = int(hb[h])
                cands = [c for c in out_map.get(nv_, []) if c not in used]
                if not cands:
                    break
                if len(cands) > 1:
                    n0 = normals[h // 3]
                    ax_u = np.cross(n0, [1.0, 0, 0] if abs(n0[0]) < 0.9 else [0, 1.0, 0])
                    ax_u /= np.linalg.norm(ax_u)
                    ax_v = np.cross(n0, ax_u)
                    din = v[nv_] - v[int(ha[h])]
                    rin = -np.array([din @ ax_u, din @ ax_v])
                    best, best_ang = cands[0], 1e9
                    for c in cands:
                        dc = v[int(hb[c])] - v[nv_]
                        dd = np.array([dc @ ax_u, dc @ ax_v])
                        ang = (math.atan2(rin[1], rin[0]) - math.atan2(dd[1], dd[0])) % (2 * math.pi)
                        if ang < 1e-12:
                            ang = 2 * math.pi
                        if ang < best_ang:
                            best, best_ang = c, ang
                    h = best
                else:
                    h = cands[0]
            if len(loop) >= 2:
                loops.append(loop)
        return loops

    def loop_chains(loop: List[int]) -> Optional[List[Tuple[int, bool]]]:
        n = len(loop)
        start = None
        for i in range(n):
            a_, b_ = loop[i], loop[(i + 1) % n]
            ek = min(a_, b_) * nv + max(a_, b_)
            if ek not in edge_chain:
                return _fail(402)
            cid, j = edge_chain[ek]
            ch = chains[cid]
            fwd = ch[j] == a_
            if (fwd and j == 0) or (not fwd and j == len(ch) - 2):
                start = i
                break
        if start is None:
            return _fail(410)
        out = []
        i = 0
        while i < n:
            a_, b_ = loop[(start + i) % n], loop[(start + i + 1) % n]
            ek = min(a_, b_) * nv + max(a_, b_)
            cid, j = edge_chain[ek]
            ch = chains[cid]
            fwd = ch[j] == a_
            out.append((cid, fwd))
            i += len(ch) - 1
        if i != n:
            return _fail(422)
        return out

    def make_wire(chs: List[Tuple[int, bool]], reverse: bool) -> TopoDS_Wire:
        w = TopoDS_Wire()
        _B.MakeWire(w)
        seq = chs[::-1] if reverse else chs
        for cid, fwd in seq:
            if reverse:
                fwd = not fwd
            e = occ_e[cid]
            _B.Add(w, e if fwd else e.Reversed())
        w.Closed(True)
        return w

    ctx = ShapeBuild_ReShape()
    faces_by_comp: Dict[int, List[TopoDS_Face]] = defaultdict(list)
    comp_of_tri = np.empty(m, np.int64)
    for ci, idx in enumerate(comps):
        comp_of_tri[idx] = ci
    tris_of: Dict[int, List[int]] = defaultdict(list)
    for t in range(m):
        tris_of[int(label[t])].append(t)
    bad: List[int] = []
    face_lab: List[tuple] = []
    counts = {"plane": 0, "ana": 0, "bsp": 0, "facet": 0}

    for L, hes in by_label.items():
        d = desc.get(L)
        if d is None:
            return _fail(452)
        loops = trace(hes)
        if not loops:
            _fail(("noloops", d[0]))
            bad.append(L)
            continue
        lch = [loop_chains(lp) for lp in loops]
        if any(x is None for x in lch):
            _fail(("loopchains", d[0]))
            bad.append(L)
            continue
        ci = int(comp_of_tri[tris_of[L][0]])
        kind = d[0]
        try:
            if kind in ("plane", "facet"):
                faces = _planar_faces(v, d[1], d[2], loops, lch, make_wire, tol)
            elif kind == "ana":
                faces = _analytic_face(v, f, normals, tris_of[L], d[1], lch, make_wire, tol, ctx)
            else:
                faces = _bspline_face(v, f, normals, tris_of[L], d[1], loops, lch, make_wire, occ_e, e_curve, tol, ctx, chains)
        except Exception as ex:
            _fail(("exc", kind, repr(ex)))
            faces = None
        if not faces:
            _fail(("badface", kind))
            bad.append(L)
            continue
        counts[kind] += len(faces)
        faces_by_comp[ci].extend(faces)
        for fc in faces:
            face_lab.append((fc, L))

    if bad:
        return None, bad, counts

    result = []
    for ci, idx in enumerate(comps):
        fl = faces_by_comp.get(ci, [])
        if not fl:
            continue
        sh = TopoDS_Shell()
        _B.MakeShell(sh)
        for fc in fl:
            _B.Add(sh, fc)
        sh = ctx.Apply(sh)
        if closed[ci]:
            sh.Closed(True)
            so = TopoDS_Solid()
            _B.MakeSolid(so)
            _B.Add(so, sh)
            result.append(so)
        else:
            result.append(sh)
    if not result:
        return _fail(500)
    shape = result[0] if len(result) == 1 else _brep.compound(result)
    try:
        BRepLib.SameParameter_s(shape, tol, False)
        BRepLib.UpdateTolerances_s(shape, False)
    except Exception:
        pass
    an = BRepCheck_Analyzer(shape, True)
    if not an.IsValid():
        # находим неверные грани -> соответствующие области в фасеты
        from OCP.TopAbs import TopAbs_FORWARD
        key = defaultdict(list)
        for fc, L in face_lab:
            g = ctx.Value(fc)
            key[hash(g.Oriented(TopAbs_FORWARD))].append((g, L))
        bad_l = set()
        ex = TopExp_Explorer(shape, TopAbs_FACE)
        while ex.More():
            cf = ex.Current()
            if not an.IsValid(cf):
                L = next((LL for g, LL in key.get(hash(cf.Oriented(TopAbs_FORWARD)), []) if g.IsSame(cf)), None)
                if L is None:
                    return _fail("unmapped bad face")
                _fail(("assembled bad", desc[L][0], _face_status(TopoDS.Face(cf))))
                bad_l.add(L)
            ex.Next()
        bad_l = [L for L in bad_l if desc[L][0] != "facet"] or list(bad_l)
        if not bad_l:
            return _fail("invalid without bad faces")
        return None, bad_l, counts
    return shape, [], counts


# ---------------------------------------------------------------------------
def _planar_faces(v, n0, p0, loops, lch, make_wire, tol):
    n0 = np.asarray(n0, float)
    ax_u = np.cross(n0, [1.0, 0, 0] if abs(n0[0]) < 0.9 else [0, 1.0, 0])
    ax_u /= np.linalg.norm(ax_u)
    ax_v = np.cross(n0, ax_u)
    info = []
    for lp, ch in zip(loops, lch):
        pts = v[lp]
        uu, vv = pts @ ax_u, pts @ ax_v
        ar = 0.5 * float(np.dot(uu, np.roll(vv, -1)) - np.dot(np.roll(uu, -1), vv))
        info.append((ch, ar, np.column_stack([uu, vv])))
    outers = [x for x in info if x[1] > 0]
    holes = [x for x in info if x[1] <= 0]
    if not outers:
        return None
    assign: Dict[int, List] = defaultdict(list)
    for hl in holes:
        p = hl[2][0]
        target = 0
        if len(outers) > 1:
            for oi, ot in enumerate(outers):
                if _brep._point_in_poly(p, ot[2]):
                    target = oi
                    break
        assign[target].append(hl)
    plane = Geom_Plane(gp_Pln(gp_Pnt(*map(float, p0)), gp_Dir(*map(float, n0))))
    faces = []
    for oi, ot in enumerate(outers):
        face = TopoDS_Face()
        _B.MakeFace(face, plane, tol)
        for ch in [ot[0]] + [h[0] for h in assign.get(oi, [])]:
            _B.Add(face, make_wire(ch, False))
        if not BRepCheck_Analyzer(face, False).IsValid():
            return _fail(("plane invalid", len(loops), _face_status(face)))
        faces.append(face)
    return faces


def _face_status(face):
    from OCP.TopAbs import TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
    an = BRepCheck_Analyzer(face, True)
    out = {}
    for T, n in ((TopAbs_VERTEX, "V"), (TopAbs_EDGE, "E"), (TopAbs_WIRE, "W")):
        ex = TopExp_Explorer(face, T)
        k = 0
        tot = 0
        while ex.More():
            tot += 1
            if not an.IsValid(ex.Current()):
                k += 1
            ex.Next()
        out[n] = (k, tot)
    from OCP.BRepCheck import BRepCheck_Face
    fc = BRepCheck_Face(face)
    try:
        out["ori"] = fc.OrientationOfWires(False).name
        out["int"] = fc.IntersectWires(False).name
        out["cls"] = fc.ClassifyWires(False).name
    except Exception as e:
        out["exc"] = repr(e)[:80]
    return out


def _max_edge_tol(face) -> float:
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE
    t = 0.0
    ex = TopExp_Explorer(face, TopAbs_EDGE)
    while ex.More():
        t = max(t, BRep_Tool.Tolerance_s(TopoDS.Edge(ex.Current())))
        ex.Next()
    return t


def _gap_xdir(P: np.ndarray, c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Направление X системы координат, при котором шов поверхности попадает в «пустой» сектор."""
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    d = P - c
    ang = np.sort(np.arctan2(d @ w, d @ u))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
    i = int(np.argmax(gaps))
    mid = ang[i] + gaps[i] / 2
    return math.cos(mid) * u + math.sin(mid) * w


def _make_analytic(s: Surface, P: np.ndarray, Nmean: np.ndarray):
    if s.kind == "sphere":
        nm = Nmean / max(np.linalg.norm(Nmean), 1e-12)
        ax = np.cross(nm, [1.0, 0, 0] if abs(nm[0]) < 0.9 else [0, 1.0, 0])
        ax /= np.linalg.norm(ax)
        xd = -nm
        a3 = gp_Ax3(gp_Pnt(*map(float, s.c)), gp_Dir(*map(float, ax)), gp_Dir(*map(float, xd)))
        return Geom_SphericalSurface(a3, float(s.r))
    xd = _gap_xdir(P, s.c, s.a)
    a3 = gp_Ax3(gp_Pnt(*map(float, s.c)), gp_Dir(*map(float, s.a)), gp_Dir(*map(float, xd)))
    if s.kind == "cyl":
        return Geom_CylindricalSurface(a3, float(s.r))
    if s.kind == "cone":
        return Geom_ConicalSurface(a3, float(math.atan(s.k)), float(s.r))
    if s.kind == "torus":
        return Geom_ToroidalSurface(a3, float(s.r), float(s.r2))
    return None


def _analytic_face(v, f, normals, tris, s: Surface, lch, make_wire, tol, ctx):
    tris = np.asarray(tris)
    vids = np.unique(f[tris].reshape(-1))
    P = v[vids]
    cen = v[f[tris]].mean(1)
    Nt = normals[tris]
    surf = _make_analytic(s, P, Nt.sum(0))
    if surf is None:
        return None
    sn = s.normal(cen)
    rev = float(np.einsum("ij,ij->i", sn, Nt).mean()) < 0
    face = TopoDS_Face()
    _B.MakeFace(face, surf, tol)
    for ch in lch:
        _B.Add(face, make_wire(ch, rev))
    sff = ShapeFix_Face(face)
    sff.SetContext(ctx)
    sff.SetPrecision(tol)
    sff.SetMaxTolerance(tol * 20)
    sff.Perform()
    face = sff.Face()
    if not BRepCheck_Analyzer(face, True).IsValid():
        return _fail(("face invalid", surf.__class__.__name__, _face_status(face)))
    return [face.Reversed() if rev else face]


def _bspline_face(v, f, normals, tris, hp: HeightPatch, loops, lch, make_wire, occ_e, e_curve, tol, ctx, chains):
    # уточнение поверхности: кроме вершин сетки она должна проходить вдоль граничных кривых
    tris = np.asarray(tris)
    pts = [v[np.unique(f[tris].reshape(-1))]]
    for ch in lch:
        for cid, _fwd in ch:
            c3 = e_curve[cid]
            if c3 is None:
                return _fail("bsp: edge without bspline")
            a, b = c3.FirstParameter(), c3.LastParameter()
            n = max(5, 4 * (len(chains[cid]) - 1) + 1)
            S = np.empty((n, 3))
            for i, t in enumerate(np.linspace(a, b, n)):
                p = c3.Value(float(t))
                S[i] = (p.X(), p.Y(), p.Z())
            pts.append(S)
    Q, qt = soft_points(v, f, tris, tol)
    hp2 = fit_height_patch(np.vstack(pts), normals[tris], tol, max_n=24, W=hp.W, Q=Q, qtol=qt,
                           max_poles=max(16, int(1.2 * _PF * len(pts[0]))))
    if hp2 is None:
        return _fail("bsp: refit failed")
    hp = hp2
    nu, nv_ = hp.H.shape
    poles = np.round(hp.poles(), 3)      # полюса — приближение в пределах допуска, 1e-3 мм достаточно
    arr = TColgp_Array2OfPnt(1, nu, 1, nv_)
    for i in range(nu):
        for j in range(nv_):
            p = poles[i, j]
            arr.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))

    def knots(t):
        u, c = np.unique(np.round(t, 12), return_counts=True)
        K = TColStd_Array1OfReal(1, len(u))
        M = TColStd_Array1OfInteger(1, len(u))
        for i, (x, k) in enumerate(zip(u, c)):
            K.SetValue(i + 1, float(x))
            M.SetValue(i + 1, int(k))
        return K, M

    Ku, Mu = knots(hp.ku)
    Kv, Mv = knots(hp.kv)
    surf = Geom_BSplineSurface(arr, Ku, Kv, Mu, Mv, 3, 3)
    # ориентация: нормаль поверхности ≈ +W
    rev = float((normals[np.asarray(tris)] @ hp.W).mean()) < 0
    # outer/holes по площадям в (u, v)
    info = []
    for lp, ch in zip(loops, lch):
        uv = hp.uv(v[lp])
        ar = 0.5 * float(np.dot(uv[:, 0], np.roll(uv[:, 1], -1)) - np.dot(np.roll(uv[:, 0], -1), uv[:, 1]))
        if rev:
            ar = -ar
        info.append((ch, ar))
    if sum(1 for x in info if x[1] > 0) != 1:
        return _fail(("bsp outers", [round(x[1], 3) for x in info]))
    info.sort(key=lambda x: -x[1])
    face = TopoDS_Face()
    _B.MakeFace(face, surf, tol)
    for ch, _ in info:
        _B.Add(face, make_wire(ch, rev))
    # p-кривые: проекция полюсов 3D B-сплайна ребра на плоскость (u, v) — та же параметризация
    done = set()
    for ch, _ in info:
        for cid, _fwd in ch:
            if cid in done:
                continue
            done.add(cid)
            c3 = e_curve[cid]
            if c3 is None:
                return None
            pl = _curve_poles(c3)
            uv = hp.uv(pl)
            n = len(uv)
            a2 = TColgp_Array1OfPnt2d(1, n)
            for i in range(n):
                a2.SetValue(i + 1, gp_Pnt2d(float(uv[i, 0]), float(uv[i, 1])))
            nk = c3.NbKnots()
            K = TColStd_Array1OfReal(1, nk)
            M = TColStd_Array1OfInteger(1, nk)
            for i in range(1, nk + 1):
                K.SetValue(i, c3.Knot(i))
                M.SetValue(i, c3.Multiplicity(i))
            c2 = Geom2d_BSplineCurve(a2, K, M, c3.Degree())
            _B.UpdateEdge(occ_e[cid], c2, face, tol)
    sff = ShapeFix_Face(face)
    sff.SetContext(ctx)
    sff.SetPrecision(tol)
    sff.SetMaxTolerance(tol * 20)
    sff.Perform()
    face = sff.Face()
    if not BRepCheck_Analyzer(face, True).IsValid():
        return _fail(("face invalid", surf.__class__.__name__, _face_status(face)))
    return [face.Reversed() if rev else face]
