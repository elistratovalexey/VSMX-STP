"""Распознавание аналитических поверхностей в треугольной сетке.

KUKA.Sim хранит CAD-модели тесселированными: цилиндр превращается в ленту из десятков
плоских четырёхугольников, скругление — в сотни треугольников. Модуль находит в сетке
участки, вершины которых лежат на одной поверхности вращения (в пределах допуска):

  * цилиндр, конус, сфера, тор (скругления), а также плоскость.

Каждый такой участок затем строится одной гранью B-rep на точной поверхности, что
кардинально уменьшает число граней/рёбер и размер STEP, а модель в CAD получает
настоящие цилиндрические отверстия, скругления и т.п.

Алгоритм:
  1. Плоские области (компланарные треугольники) — как и раньше.
  2. «Мелкие» плоские области (1–4 треугольника), соединённые негладкими рёбрами
     с углом < 45°, образуют гладкие участки-кандидаты.
  3. На участке выращивается область от затравки: подбор поверхности (МНК +
     уточнение Гаусса–Ньютона), добавление соседних треугольников, вершины которых
     лежат на поверхности, периодическое переуточнение. Неподошедшие треугольники
     остаются плоскими гранями.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
_PF = 4.0   # предел числа полюсов B-сплайна: не больше 4 на вершину сетки участка


@dataclass
class Surface:
    kind: str                       # 'plane' | 'cyl' | 'cone' | 'sphere' | 'torus'
    c: np.ndarray                   # точка на оси / центр
    a: np.ndarray                   # ось (единичный вектор); для сферы — произвольная
    r: float = 0.0                  # радиус (цилиндр, сфера, конус — радиус в точке c, тор — большой)
    k: float = 0.0                  # конус: tg половины угла; тор: малый радиус хранится в r2
    r2: float = 0.0

    def residual(self, P: np.ndarray) -> np.ndarray:
        return _residual(self.kind, self.c, self.a, self.r, self.k, self.r2, P)

    def normal(self, P: np.ndarray) -> np.ndarray:
        """Наружная (в смысле параметризации OCCT) нормаль поверхности в точках P."""
        d = P - self.c
        if self.kind == "plane":
            return np.repeat(self.a[None, :], len(P), 0)
        if self.kind == "sphere":
            return _unit(d)
        z = d @ self.a
        radial = d - z[:, None] * self.a
        rho = np.linalg.norm(radial, axis=1)
        u = radial / np.maximum(rho, 1e-300)[:, None]
        if self.kind == "cyl":
            return u
        if self.kind == "cone":
            ang = math.atan(self.k)
            return _unit(u * math.cos(ang) - self.a[None, :] * math.sin(ang))
        if self.kind == "torus":
            q = (rho - self.r)[:, None] * u + z[:, None] * self.a
            return _unit(q)
        raise ValueError(self.kind)


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-300)


def _residual(kind, c, a, r, k, r2, P):
    d = P - c
    if kind == "plane":
        return d @ a
    if kind == "sphere":
        return np.linalg.norm(d, axis=1) - r
    z = d @ a
    rho = np.linalg.norm(d - z[:, None] * a, axis=1)
    if kind == "cyl":
        return rho - r
    if kind == "cone":
        return (rho - (r + k * z)) / math.sqrt(1.0 + k * k)
    if kind == "torus":
        return np.sqrt((rho - r) ** 2 + z ** 2) - r2
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# начальные приближения
# ---------------------------------------------------------------------------
def _axis_point(P: np.ndarray, N: np.ndarray, a: np.ndarray) -> Optional[np.ndarray]:
    """Точка оси (направление a известно): нормали поверхности вращения пересекают ось."""
    w = np.cross(N, a)                       # (p - c)·(n × a) = 0
    A = w
    b = np.einsum("ij,ij->i", w, P)
    # ограничение c·a = 0 (точка оси ближайшая к началу координат, затем сдвигаем)
    A2 = np.vstack([A, a[None, :] * 1.0])
    b2 = np.r_[b, float(P.mean(0) @ a)]
    try:
        c, *_ = np.linalg.lstsq(A2, b2, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return c


def _plucker_axis(P: np.ndarray, N: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Ось поверхности вращения общего вида (тор): прямая, пересекающая все нормали."""
    M = np.hstack([np.cross(P, N), N])       # a·(p×n) + n·m = 0
    try:
        _, s, vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    x = vt[-1]
    a, m = x[:3], x[3:]
    na = np.linalg.norm(a)
    if na < 1e-9:
        return None
    a, m = a / na, m / na
    c = np.cross(a, m)
    return c, a


def _circle2d(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float]]:
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return float(cx), float(cy), float(math.sqrt(r2))


def _frame(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, t)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


# ---------------------------------------------------------------------------
# уточнение (Гаусс–Ньютон с демпфированием) по расстояниям вершин
# ---------------------------------------------------------------------------
def _pack(s: Surface) -> np.ndarray:
    return np.r_[s.c, s.a, s.r, s.k, s.r2]


def _refine(s: Surface, P: np.ndarray, iters: int = 12) -> Surface:
    kind = s.kind
    idx = {"plane": [0, 1, 2, 3, 4, 5], "cyl": [0, 1, 2, 3, 4, 5, 6], "cone": [0, 1, 2, 3, 4, 5, 6, 7],
           "sphere": [0, 1, 2, 6], "torus": [0, 1, 2, 3, 4, 5, 6, 8]}[kind]
    x = _pack(s)
    scale = float(np.ptp(P, axis=0).max()) or 1.0

    def res(xx):
        a = xx[3:6]
        na = np.linalg.norm(a)
        a = a / na if na > 1e-12 else np.array([0, 0, 1.0])
        return _residual(kind, xx[0:3], a, xx[6], xx[7], xx[8], P)

    lam = 1e-3
    r0 = res(x)
    f0 = float(r0 @ r0)
    for _ in range(iters):
        J = np.empty((len(P), len(idx)))
        for j, i in enumerate(idx):
            h = 1e-6 * (scale if i < 3 or i in (6, 8) else 1.0)
            xp = x.copy()
            xp[i] += h
            J[:, j] = (res(xp) - r0) / h
        g = J.T @ r0
        H = J.T @ J
        improved = False
        for _try in range(5):
            try:
                dx = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
            except np.linalg.LinAlgError:
                break
            xn = x.copy()
            xn[idx] += dx
            if kind != "sphere":
                xn[3:6] /= max(np.linalg.norm(xn[3:6]), 1e-12)
            rn = res(xn)
            fn = float(rn @ rn)
            if fn < f0:
                x, r0, f0 = xn, rn, fn
                lam = max(lam * 0.3, 1e-9)
                improved = True
                break
            lam *= 10
        if not improved or f0 < 1e-20:
            break
    a = x[3:6] / max(np.linalg.norm(x[3:6]), 1e-12)
    return Surface(kind, x[0:3].copy(), a, float(x[6]), float(x[7]), float(x[8]))


# ---------------------------------------------------------------------------
# подбор поверхности для набора вершин
# ---------------------------------------------------------------------------
def fit_surface(P: np.ndarray, N: np.ndarray, tol: float, kinds: Sequence[str] = ("cyl", "cone", "sphere", "torus"),
                size: Optional[float] = None) -> Optional[Surface]:
    """Подбирает поверхность, на которой лежат точки P (нормали N — оценки по сетке)."""
    if len(P) < 6:
        return None
    size = size or float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    cands: List[Surface] = []
    # ось по нормалям: для цилиндра n ⟂ a, для конуса n·a = const
    C = np.cov(N.T) if len(N) > 3 else None
    if "cyl" in kinds:
        w, V = np.linalg.eigh(N.T @ N)
        a = V[:, 0]
        if np.max(np.abs(N @ a)) < 0.2:
            c = _axis_point(P, N, a)
            if c is not None:
                u, v = _frame(a)
                d = P - c
                cf = _circle2d(d @ u, d @ v)
                if cf:
                    c = c + cf[0] * u + cf[1] * v
                    cands.append(Surface("cyl", c, a, cf[2]))
    if "cone" in kinds and C is not None:
        w, V = np.linalg.eigh(C)
        a = V[:, 0]
        cosang = N @ a
        if np.ptp(cosang) < 0.25 and 0.02 < abs(cosang.mean()) < 0.98:
            c = _axis_point(P, N, a)
            if c is not None:
                d = P - c
                z = d @ a
                rho = np.linalg.norm(d - z[:, None] * a, axis=1)
                A = np.column_stack([np.ones_like(z), z])
                (r0, k), *_ = np.linalg.lstsq(A, rho, rcond=None)
                if abs(k) > 1e-3:
                    zm = float(z.mean())
                    cands.append(Surface("cone", c + zm * a, a, float(r0 + k * zm), float(k)))
    if "sphere" in kinds:
        # точка, ближайшая ко всем нормалям
        I = np.eye(3)
        Ms = I[None] - N[:, :, None] * N[:, None, :]
        A = Ms.sum(0)
        b = np.einsum("nij,nj->i", Ms, P)
        try:
            c = np.linalg.solve(A, b)
            R = float(np.linalg.norm(P - c, axis=1).mean())
            cands.append(Surface("sphere", c, np.array([0, 0, 1.0]), R))
        except np.linalg.LinAlgError:
            pass
    if "torus" in kinds:
        pa = _plucker_axis(P, N)
        if pa is not None:
            c, a = pa
            d = P - c
            z = d @ a
            rho = np.linalg.norm(d - z[:, None] * a, axis=1)
            cf = _circle2d(rho, z)
            if cf:
                R, z0, rr = cf
                if R > 1e-6 and rr > 1e-6:
                    cands.append(Surface("torus", c + z0 * a, a, R, 0.0, rr))
    best = None
    for s in cands:
        try:
            s2 = _refine(s, P)
        except Exception:
            continue
        if not _sane(s2, size):
            continue
        err = float(np.max(np.abs(s2.residual(P))))
        if err <= tol:
            # нормали треугольников должны быть близки к нормалям поверхности
            nn = s2.normal(P)
            if np.min(np.abs(np.einsum("ij,ij->i", nn, N))) < 0.5:
                continue
            best = s2
            break           # порядок kinds: от простого к сложному
    return best


def _sane(s: Surface, size: float) -> bool:
    if not np.all(np.isfinite(_pack(s))):
        return False
    lim = 50.0 * size
    if s.kind in ("cyl", "sphere"):
        return 1e-4 < s.r < lim
    if s.kind == "cone":
        return 0.0 <= s.r < lim and 1e-3 < abs(s.k) < 50
    if s.kind == "torus":
        return 1e-4 < s.r2 < lim and s.r > s.r2 * 1.001 and s.r < lim
    return True


# ---------------------------------------------------------------------------
# сегментация сетки
# ---------------------------------------------------------------------------
def segment(v: np.ndarray, f: np.ndarray, normals: np.ndarray, twin_tri: np.ndarray, plane_region: np.ndarray,
            tol: float, min_tris: int = 4, small_region: int = 4, max_dihedral_deg: float = 45.0,
            ) -> Tuple[np.ndarray, List[Surface]]:
    """Возвращает (номер аналитической поверхности для каждого треугольника или -1, список поверхностей)."""
    m = len(f)
    surf_of = np.full(m, -1, np.int64)
    surfaces: List[Surface] = []
    if m < min_tris:
        return surf_of, surfaces
    sizes = np.bincount(plane_region, minlength=int(plane_region.max()) + 1)
    cand = sizes[plane_region] <= small_region
    if cand.sum() < min_tris:
        return surf_of, surfaces
    cos_max = math.cos(math.radians(max_dihedral_deg))
    nb = twin_tri.reshape(m, 3)

    # нормали в вершинах (по треугольникам-кандидатам)
    def vertex_normals(tris: np.ndarray) -> Dict[int, np.ndarray]:
        acc: Dict[int, np.ndarray] = {}
        cr = np.cross(v[f[tris, 1]] - v[f[tris, 0]], v[f[tris, 2]] - v[f[tris, 0]])
        for t, c_ in zip(tris, cr):
            for vid in f[t]:
                if vid in acc:
                    acc[vid] = acc[vid] + c_
                else:
                    acc[vid] = c_.copy()
        return acc

    def ok_edge(t: int, u: int) -> bool:
        if u < 0 or not cand[u] or surf_of[u] >= 0:
            return False
        return float(normals[t] @ normals[u]) > cos_max

    # гладкие участки
    patch = np.full(m, -1, np.int64)
    patches: List[np.ndarray] = []
    for s in np.flatnonzero(cand):
        if patch[s] >= 0:
            continue
        pid = len(patches)
        patch[s] = pid
        q = [s]
        mem = [s]
        while q:
            t = q.pop()
            for u in nb[t]:
                if u >= 0 and patch[u] < 0 and cand[u] and float(normals[t] @ normals[u]) > cos_max:
                    patch[u] = pid
                    q.append(u)
                    mem.append(u)
        patches.append(np.array(mem, np.int64))

    for mem in patches:
        if len(mem) < min_tris:
            continue
        _grow_in_patch(v, f, normals, nb, mem, patch, cand, tol, min_tris, surf_of, surfaces, cos_max)
    return surf_of, surfaces


def _grow_in_patch(v, f, normals, nb, mem, patch, cand, tol, min_tris, surf_of, surfaces, cos_max):
    attempts = [0]
    max_attempts = max(8, len(mem) // 24)
    inpatch = set(int(t) for t in mem)
    tried_seed = set()
    size_all = float(np.linalg.norm(np.ptp(v[f[mem].reshape(-1)], axis=0))) or 1.0

    def tri_ok(s: Surface, t: int) -> bool:
        P = v[f[t]]
        if np.max(np.abs(s.residual(P))) > tol:
            return False
        nn = s.normal(P.mean(0)[None, :])[0]
        return abs(float(nn @ normals[t])) > 0.5

    # сначала пробуем весь участок целиком
    def collect(tris):
        vids = np.unique(f[tris].reshape(-1))
        P = v[vids]
        acc = np.zeros((len(vids), 3))
        pos = {int(x): i for i, x in enumerate(vids)}
        cr = np.cross(v[f[tris, 1]] - v[f[tris, 0]], v[f[tris, 2]] - v[f[tris, 0]])
        for t, c_ in zip(tris, cr):
            for vid in f[t]:
                acc[pos[int(vid)]] += c_
        return P, _unit(acc)

    P, N = collect(mem)
    s = fit_surface(P, N, tol, size=size_all)
    if s is not None:
        sid = len(surfaces)
        surfaces.append(s)
        surf_of[mem] = sid
        return

    # выращивание от затравок (крупные треугольники первыми)
    area = np.linalg.norm(np.cross(v[f[mem, 1]] - v[f[mem, 0]], v[f[mem, 2]] - v[f[mem, 0]]), axis=1)
    order = mem[np.argsort(-area)]
    for seed in order:
        seed = int(seed)
        if surf_of[seed] >= 0 or seed in tried_seed:
            continue
        tried_seed.add(seed)
        # затравочная область — ближайшие ~16 треугольников
        region = [seed]
        seen = {seed}
        dq = deque([seed])
        while dq and len(region) < 16:
            t = dq.popleft()
            for u in nb[t]:
                u = int(u)
                if u >= 0 and u in inpatch and u not in seen and surf_of[u] < 0 and \
                        float(normals[t] @ normals[u]) > cos_max:
                    seen.add(u)
                    region.append(u)
                    dq.append(u)
        tried_seed.update(region)
        if len(region) < min_tris:
            continue
        attempts[0] += 1
        if attempts[0] > max_attempts:
            break
        P, N = collect(np.array(region))
        s = fit_surface(P, N, tol, size=size_all)
        if s is None:
            continue
        # рост
        members = set(region)
        front = deque(region)
        last_refit = len(members)
        while front:
            t = front.popleft()
            for u in nb[t]:
                u = int(u)
                if u < 0 or u in members or u not in inpatch or surf_of[u] >= 0:
                    continue
                if float(normals[t] @ normals[u]) <= cos_max:
                    continue
                if tri_ok(s, u):
                    members.add(u)
                    front.append(u)
            if len(members) > last_refit * 1.6:
                arr = np.array(sorted(members))
                P, N = collect(arr)
                s2 = fit_surface(P, N, tol, kinds=(s.kind,), size=size_all)
                if s2 is not None:
                    s = s2
                last_refit = len(members)
        arr = np.array(sorted(members))
        if len(arr) < min_tris:
            continue
        P, _ = collect(arr)
        if np.max(np.abs(s.residual(P))) > tol:
            continue
        sid = len(surfaces)
        surfaces.append(s)
        surf_of[arr] = sid


# ---------------------------------------------------------------------------
# поверхности свободной формы: B-сплайн «карта высот» над средней плоскостью
# ---------------------------------------------------------------------------
@dataclass
class HeightPatch:
    O: np.ndarray
    U: np.ndarray
    V: np.ndarray
    W: np.ndarray
    ku: np.ndarray          # полный (с повторами) узловой вектор по u, в мм
    kv: np.ndarray
    H: np.ndarray           # высоты контрольных точек (nu, nv)
    gu: np.ndarray          # абсциссы Гревиля (координаты контрольных точек по u)
    gv: np.ndarray

    def uv(self, P: np.ndarray) -> np.ndarray:
        d = P - self.O
        return np.column_stack([d @ self.U, d @ self.V])

    def height(self, uv: np.ndarray) -> np.ndarray:
        nu, nv = self.H.shape
        Bu, Bv = _basis(uv[:, 0], self.ku, nu), _basis(uv[:, 1], self.kv, nv)
        return np.einsum("pi,ij,pj->p", Bu, self.H, Bv)

    def poles(self) -> np.ndarray:
        nu, nv = self.H.shape
        X = self.gu[:, None, None] * self.U + self.gv[None, :, None] * self.V
        return self.O + X + self.H[:, :, None] * self.W


def _clamped_knots(a: float, b: float, n: int, p: int = 3) -> np.ndarray:
    inner = np.linspace(a, b, n - p + 1)[1:-1]
    return np.r_[[a] * (p + 1), inner, [b] * (p + 1)]


def _basis(x: np.ndarray, t: np.ndarray, n: int, p: int = 3) -> np.ndarray:
    """Матрица значений B-сплайновых базисных функций (len(x), n)."""
    x = np.clip(x, t[p], t[n])
    m = len(t) - 1
    Nb = np.zeros((len(x), m))
    for i in range(m):
        if t[i + 1] > t[i]:
            Nb[:, i] = (x >= t[i]) & (x < t[i + 1])
    # правый конец
    last = np.flatnonzero(t[1:] > t[:-1])[-1]
    Nb[x >= t[n], :] = 0
    Nb[x >= t[n], last] = 1
    for k in range(1, p + 1):
        Nn = np.zeros((len(x), m - k))
        for i in range(m - k):
            d1 = t[i + k] - t[i]
            d2 = t[i + k + 1] - t[i + 1]
            term = 0.0
            if d1 > 0:
                term = (x - t[i]) / d1 * Nb[:, i]
            if d2 > 0:
                term = term + (t[i + k + 1] - x) / d2 * Nb[:, i + 1]
            Nn[:, i] = term
        Nb = Nn
    return Nb[:, :n]


def _greville(t: np.ndarray, n: int, p: int = 3) -> np.ndarray:
    return np.array([t[i + 1:i + p + 1].mean() for i in range(n)])


def fit_height_patch(P: np.ndarray, N: np.ndarray, tol: float, max_n: int = 20,
                     W: Optional[np.ndarray] = None, Q: Optional[np.ndarray] = None,
                     qtol: Optional[np.ndarray] = None, qw: float = 0.3,
                     max_poles: Optional[int] = None) -> Optional[HeightPatch]:
    """Q — «мягкие» точки (на гранях сетки): вес qw, допуск qtol (поэлементно) — защищают от волн."""
    """B-сплайн z = h(u, v) над плоскостью ⟂ W, проходящий через вершины в пределах tol."""
    if len(P) < 6:
        return None
    if W is None:
        W = _unit(N.sum(0))
    if np.min(N @ W) < 0.3:          # не однозначная проекция — участок надо делить
        return None
    U, V = _frame(W)
    O = P.mean(0)
    nP = len(P)
    if Q is not None and len(Q):
        P = np.vstack([P, Q])
    else:
        Q = None
    d = P - O
    x, y, h = d @ U, d @ V, d @ W
    wts = np.ones(len(P))
    if Q is not None:
        wts[nP:] = qw
    span = max(np.ptp(x), np.ptp(y), 1e-9)
    pad = 0.02 * span + 1e-6
    x0, x1, y0, y1 = x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad
    # размеры сетки — пропорционально сторонам
    ratio = (x1 - x0) / max(y1 - y0, 1e-9)
    for base in (4, 6, 8, 11, 15, 20, 26):
        if base > max_n * 1.3:
            break
        nu = int(min(max_n, max(4, round(base * math.sqrt(ratio)))))
        nv = int(min(max_n, max(4, round(base / math.sqrt(ratio)))))
        if nu * nv > (max_poles if max_poles else 3 * nP + 16):
            break
        ku, kv = _clamped_knots(x0, x1, nu), _clamped_knots(y0, y1, nv)
        Bu, Bv = _basis(x, ku, nu), _basis(y, kv, nv)
        A = (Bu[:, :, None] * Bv[:, None, :]).reshape(len(P), nu * nv)
        # сглаживание (вторые разности) — устойчивость для точек вне данных
        lam = 1e-4 * math.sqrt(len(P))
        rows = []
        for i in range(nu):
            for j in range(nv):
                if 0 < i < nu - 1:
                    r = np.zeros(nu * nv); r[(i - 1) * nv + j] = 1; r[i * nv + j] = -2; r[(i + 1) * nv + j] = 1
                    rows.append(r)
                if 0 < j < nv - 1:
                    r = np.zeros(nu * nv); r[i * nv + j - 1] = 1; r[i * nv + j] = -2; r[i * nv + j + 1] = 1
                    rows.append(r)
        R = np.array(rows) * lam if rows else np.zeros((0, nu * nv))
        wl = wts.copy()
        ok = False
        for _it in range(5):
            M = np.vstack([A * wl[:, None], R])
            b = np.r_[h * wl, np.zeros(len(R))]
            try:
                sol, *_ = np.linalg.lstsq(M, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
            err = np.abs(A @ sol - h)
            vh = err[:nP] > tol * 0.9
            vq = (err[nP:] > qtol * 0.9) if Q is not None else np.zeros(0, bool)
            if err[:nP].max() <= tol and (Q is None or np.all(err[nP:] <= qtol)):
                ok = True
                break
            if err[:nP].max() > 8 * tol:
                break
            wl[:nP][vh] *= 4.0
            if Q is not None:
                wl[nP:][vq] *= 3.0
        if ok:
            Hm = sol.reshape(nu, nv)
            return HeightPatch(O, U, V, W, ku, kv, Hm, _greville(ku, nu), _greville(kv, nv))
    return None


def soft_points(v: np.ndarray, f: np.ndarray, tris: np.ndarray, tol: float):
    """Центры треугольников и середины рёбер с допуском tol + 0.12·(длина ребра)."""
    T = v[f[tris]]
    L = np.linalg.norm(T - np.roll(T, -1, axis=1), axis=2)       # (n, 3)
    mids = 0.5 * (T + np.roll(T, -1, axis=1))
    cen = T.mean(1)
    Q = np.vstack([cen, mids.reshape(-1, 3)])
    qt = np.r_[tol + 0.12 * L.max(1), tol + 0.12 * L.reshape(-1)]
    return Q, qt


def segment_freeform(v: np.ndarray, f: np.ndarray, normals: np.ndarray, twin_tri: np.ndarray, mask: np.ndarray,
                     tol: float, min_tris: int = 6, max_dihedral_deg: float = 45.0, max_depth: int = 6,
                     ) -> List[Tuple[np.ndarray, HeightPatch]]:
    """Гладкие участки из треугольников mask -> B-сплайновые «карты высот» (с делением при необходимости)."""
    m = len(f)
    nb = twin_tri.reshape(m, 3)
    cos_max = math.cos(math.radians(max_dihedral_deg))
    comp = np.full(m, -1, np.int64)
    comps: List[np.ndarray] = []
    for s in np.flatnonzero(mask):
        if comp[s] >= 0:
            continue
        cid = len(comps)
        comp[s] = cid
        st = [s]
        mem = [s]
        while st:
            t = st.pop()
            for u in nb[t]:
                if u >= 0 and mask[u] and comp[u] < 0 and float(normals[t] @ normals[u]) > cos_max:
                    comp[u] = cid
                    st.append(u)
                    mem.append(u)
        comps.append(np.array(mem, np.int64))

    out: List[Tuple[np.ndarray, HeightPatch]] = []

    def connected_parts(tris: np.ndarray) -> List[np.ndarray]:
        S = set(int(t) for t in tris)
        seen = set()
        parts = []
        for t0 in tris:
            t0 = int(t0)
            if t0 in seen:
                continue
            seen.add(t0)
            st = [t0]
            mem = [t0]
            while st:
                t = st.pop()
                for u in nb[t]:
                    u = int(u)
                    if u in S and u not in seen and float(normals[t] @ normals[u]) > cos_max:
                        seen.add(u)
                        st.append(u)
                        mem.append(u)
            parts.append(np.array(mem, np.int64))
        return parts

    def area_normals(tris):
        cr = np.cross(v[f[tris, 1]] - v[f[tris, 0]], v[f[tris, 2]] - v[f[tris, 0]])
        return cr

    def rec(tris: np.ndarray, depth: int):
        if len(tris) < min_tris:
            return
        vids = np.unique(f[tris].reshape(-1))
        P = v[vids]
        cr = area_normals(tris)
        W = _unit(cr.sum(0))
        Nt = _unit(cr)
        hp = None
        if np.min(Nt @ W) >= 0.3:
            Q, qt = soft_points(v, f, tris, tol)
            hp = fit_height_patch(P, Nt, tol, W=W, Q=Q, qtol=qt, max_poles=max(16, int(_PF * len(P))))
        if hp is not None:
            out.append((tris, hp))
            return
        if depth >= max_depth:
            return
        # деление: по нормалям (если разброс велик) или пополам по наибольшей протяжённости
        if np.min(Nt @ W) < 0.3:
            w, Vv = np.linalg.eigh(np.cov(Nt.T))
            axis = Vv[:, -1]
            key = Nt @ axis
        else:
            cen = v[f[tris]].mean(1)
            w, Vv = np.linalg.eigh(np.cov(cen.T))
            key = cen @ Vv[:, -1]
        med = np.median(key)
        for half in (tris[key <= med], tris[key > med]):
            for part in connected_parts(half):
                rec(part, depth + 1)

    for c in comps:
        rec(c, 0)
    return out
