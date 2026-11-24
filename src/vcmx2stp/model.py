"""Построение дерева сцены (сборки) из rsc-описания Visual Components / KUKA.Sim.

Иерархия STEP повторяет иерархию VC::

    Layout (сборка)
      └─ Компонент (rSimResource)            — сборка
           ├─ детали узла (features)          — тела
           └─ звено (rSimLink)                — подсборка
                ├─ детали звена
                └─ присоединённый компонент   — подсборка ...
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import expr as X
from .rsc_parser import El

# ---------------------------------------------------------------- данные


@dataclass
class GeoRef:
    """Описание геометрии-прототипа (одинаковые ключи -> один экземпляр в STEP)."""

    key: tuple                      # ключ прототипа
    kind: str                       # geo|box|cyl|cone|sphere|wedge|extrude|revolve
    params: dict = field(default_factory=dict)


@dataclass
class Part:
    name: str
    geo: GeoRef
    matrix: np.ndarray              # положение относительно узла
    color: Optional[Tuple[float, float, float]] = None
    source: str = ""                # компонент / feature (для отчёта)


@dataclass
class SceneNode:
    name: str
    kind: str                       # layout|component|link|group
    matrix: np.ndarray
    parts: List[Part] = field(default_factory=list)
    children: List["SceneNode"] = field(default_factory=list)
    vcid: str = ""

    def count_parts(self) -> int:
        return len(self.parts) + sum(c.count_parts() for c in self.children)


@dataclass
class Issue:
    component: str
    feature: str
    uri: str
    reason: str


@dataclass
class BuildOptions:
    include_hidden: bool = False
    include_envelopes: bool = False     # рабочие зоны роботов (switch WorkSpace::...)


# ---------------------------------------------------------------- цвета
_PALETTE = {
    "white": (0.95, 0.95, 0.95), "black": (0.1, 0.1, 0.1), "red": (0.8, 0.1, 0.1),
    "green": (0.1, 0.6, 0.1), "blue": (0.1, 0.25, 0.8), "yellow": (0.95, 0.8, 0.05),
    "orange": (1.0, 0.45, 0.0), "grey": (0.55, 0.55, 0.55), "gray": (0.55, 0.55, 0.55),
    "lightgrey": (0.78, 0.78, 0.78), "darkgrey": (0.3, 0.3, 0.3), "silver": (0.75, 0.75, 0.78),
    "aluminium": (0.8, 0.8, 0.82), "aluminum": (0.8, 0.8, 0.82), "steel": (0.6, 0.62, 0.65),
    "chrome": (0.85, 0.85, 0.88), "magenta": (0.8, 0.1, 0.7), "cyan": (0.1, 0.75, 0.8),
    "brown": (0.45, 0.3, 0.15), "beige": (0.9, 0.85, 0.7), "glass": (0.7, 0.85, 0.9),
    "plexi": (0.75, 0.85, 0.95), "kuka": (1.0, 0.4, 0.0), "fanuc": (0.95, 0.8, 0.05),
}


def palette_color(name: str) -> Optional[Tuple[float, float, float]]:
    low = (name or "").lower()
    for k in sorted(_PALETTE, key=len, reverse=True):
        if k in low:
            return _PALETTE[k]
    return None


def _diffuse_of(el: El) -> Optional[Tuple[float, float, float]]:
    d = el.first("Diffuse")
    if d is None:
        return None
    if d.children:
        try:
            return (float(d.value("Red", "0")), float(d.value("Green", "0")), float(d.value("Blue", "0")))
        except ValueError:
            return None
    if len(d.args) >= 3:
        try:
            return tuple(float(x) for x in d.args[:3])  # type: ignore[return-value]
        except ValueError:
            return None
    return None


def parse_materials(els: List[El]) -> Dict[str, Tuple[float, float, float]]:
    out: Dict[str, Tuple[float, float, float]] = {}
    from .rsc_parser import walk

    for e in walk(els):
        if e.key == "Material" and e.children:
            nm = e.name or e.arg
            c = _diffuse_of(e)
            if nm and c:
                out.setdefault(nm, c)
        if e.key == "Variable" and e.arg == "rMaterialVariable":
            v = e.first("Value")
            if v is not None and v.arg and v.children:
                c = _diffuse_of(v)
                if c:
                    out.setdefault(v.arg, c)
    return out


# ---------------------------------------------------------------- контекст компонента
class CompContext:
    """Таблица символов компонента для вычисления выражений."""

    def __init__(self, node: El, name: str, warn: Callable[[str], None]):
        self.name = name
        self.raw: Dict[str, object] = {}
        self.cache: Dict[str, object] = {}
        self.busy: set = set()
        self.locals: List[Dict[str, object]] = []
        self.ev = X.Evaluator(self.resolve, lambda m: warn(f"[{name}] {m}"))
        vs = node.first("VariableSpace")
        if vs is not None:
            for v in vs.kids("Variable"):
                if v.name:
                    self.raw[v.name] = v.first("Value")
        for fn in node.kids("Functionality"):
            blk_name = fn.value("Name")
            if not blk_name or not fn.children:
                continue
            for c in fn.kids():
                if c.key in ("Name", "Id", "Visible") or c.key[0].islower():
                    continue
                if c.children is None and c.args:
                    self.raw.setdefault(f"{blk_name}::{c.key}", c)
        self.dofs: Dict[str, El] = {}

    def add_dof(self, name: str, dof: El):
        if name:
            self.dofs.setdefault(name, dof)

    def _value_of(self, el) -> object:
        if el is None:
            return None
        if isinstance(el, El):
            ex = el.first("Expression") if el.children else None
            if ex is not None:
                return self.ev.eval(" ".join(ex.args), 0.0)
            args = el.args
            if not args:
                return 0.0
            if len(args) == 16:
                try:
                    return X.matrix_from_vc(args)
                except ValueError:
                    pass
            if len(args) in (3, 4):
                try:
                    return X.vec(*[float(a) for a in args[:3]])
                except ValueError:
                    pass
            a = args[0]
            low = a.lower()
            if low in ("true", "yes"):
                return 1.0
            if low in ("false", "no"):
                return 0.0
            try:
                return float(a)
            except ValueError:
                return a
        return el

    def resolve(self, name: str):
        for loc in reversed(self.locals):
            if name in loc:
                return loc[name]
        if name in self.cache:
            return self.cache[name]
        if name in self.busy:
            return None
        el = self.raw.get(name)
        if el is None and name in self.dofs:
            try:
                return float(self.dofs[name].value("JointValue", "0"))
            except ValueError:
                return 0.0
        if el is None and "::" in name:
            el = self.raw.get(name.split("::")[-1])
        if el is None:
            # регистронезависимый поиск
            low = name.lower()
            for k, v in self.raw.items():
                if k.lower() == low:
                    el = v
                    break
        if el is None:
            return None
        self.busy.add(name)
        try:
            val = self._value_of(el)
        finally:
            self.busy.discard(name)
        self.cache[name] = val
        return val

    # удобные обёртки
    def num(self, el: El, key: str, default: float = 0.0) -> float:
        src = el.expression(key)
        if src is None or src == "":
            return default
        return self.ev.num(src, default)

    def vector(self, el: El, key: str, default=(0.0, 0.0, 1.0)) -> np.ndarray:
        src = el.expression(key)
        if not src:
            return np.array(default, float)
        return self.ev.vector(src, default)


# ---------------------------------------------------------------- построитель
_AXES = {0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1), 3: (-1, 0, 0), 4: (0, -1, 0), 5: (0, 0, -1)}
_PRIMS = {
    "rPrimitiveBoxFeature", "rPrimitiveCylinderFeature", "rPrimitiveConeFeature",
    "rPrimitiveSphereFeature", "rPrimitiveWedgeFeature",
}
_IGNORED = {"rFrameFeature", "rTextFeature", "rLabelFeature", "rPointCloudFeature", "rDimensionFeature"}


def _mat_of(el: El, key: str = "Matrix") -> np.ndarray:
    e = el.first(key)
    if e is None or len(e.args) < 16:
        return np.eye(4)
    try:
        return X.matrix_from_vc(e.args)
    except ValueError:
        return np.eye(4)


def _r(x: float) -> float:
    return round(float(x), 6)


class SceneBuilder:
    def __init__(self, archive, materials: Dict[str, Tuple[float, float, float]], options: BuildOptions, log=print):
        self.ar = archive
        self.materials = materials
        self.opt = options
        self.log = log
        self.issues: List[Issue] = []
        self.warnings: List[str] = []
        self.empty_components: List[str] = []
        self.stats = {"components": 0, "links": 0, "features": 0, "parts": 0}
        self._warned = set()

    def warn(self, msg: str):
        if msg not in self._warned:
            self._warned.add(msg)
            self.warnings.append(msg)
            self.log("Предупреждение: " + msg)

    # ---------------------------------------------------------- узлы
    def build(self, roots: List[El], name: str) -> SceneNode:
        top = SceneNode(name=name, kind="layout", matrix=np.eye(4))
        for el in roots:
            if el.key == "Node":
                n = self._node(el, None, None)
                if n is not None:
                    top.children.extend(n if isinstance(n, list) else [n])
        # «прозрачные» контейнеры верхнего уровня (LAYOUT/WorldContainer/ROOT) раскрываем
        top.children = self._flatten_groups(top.children)
        return top

    def _flatten_groups(self, nodes: List[SceneNode]) -> List[SceneNode]:
        out = []
        for n in nodes:
            if n.kind == "group" and not n.parts:
                for c in self._flatten_groups(n.children):
                    c.matrix = n.matrix @ c.matrix
                    out.append(c)
            else:
                out.append(n)
        return out

    def _node(self, el: El, ctx: Optional[CompContext], comp_vcid: Optional[str], inherited=None):
        typ = el.arg
        name = el.name or typ
        if typ == "rSimResource" and not self.opt.include_hidden and _hidden_component(el):
            self.log(f"Скрытый компонент пропущен: {name}")
            return None
        if typ == "rSimResource":
            self.stats["components"] += 1
            vcid = (el.value("VCID") or "").lower() or None
            nc = el.first("NodeClass")
            if not vcid and nc is not None:
                vcid = (nc.value("VCID") or "").lower() or None
            ctx = CompContext(el, name, self.warn)
            self._collect_dofs(el, ctx)
            comp_vcid = vcid or comp_vcid
            m = _mat_of(el, "Location")
            kind = "component"
            color = self._component_color(nc)
        elif typ == "rSimLink":
            self.stats["links"] += 1
            m = self._offset(el, ctx) @ self._dof_motion(el, ctx)
            kind = "link"
            color = inherited
        elif typ in ("rSimGroup", "rSimWorld"):
            m = _mat_of(el, "Location")
            kind = "group"
            color = inherited
        else:
            m = self._offset(el, ctx) if el.first("Offset") is not None else _mat_of(el, "Location")
            kind = "group" if typ != "rBaseNode" else "link"
            color = inherited
        node = SceneNode(name=name, kind=kind, matrix=m, vcid=comp_vcid or "")
        nc = el.first("NodeClass")
        comp_name = ctx.name if ctx is not None else name
        if nc is not None and ctx is not None:
            self._features(list(nc.kids("Feature")), np.eye(4), ctx, comp_vcid, node, comp_name, color)
        for c in el.kids("Node"):
            ch = self._node(c, ctx, comp_vcid, color if c.arg != "rSimResource" else None)
            if ch is not None:
                node.children.append(ch)
        if typ == "rSimResource":
            if node.count_parts() == 0:
                self.empty_components.append(name)
        if node.count_parts() == 0:
            return None
        return node

    def _collect_dofs(self, comp: El, ctx: CompContext):
        stack = [comp]
        while stack:
            e = stack.pop()
            for c in e.kids("Node"):
                if c.arg == "rSimResource":
                    continue
                d = c.first("Dof")
                if d is not None:
                    ctx.add_dof(d.value("Name") or c.name, d)
                stack.append(c)

    def _offset(self, el: El, ctx: Optional[CompContext]) -> np.ndarray:
        e = el.first("Offset")
        if e is None:
            return np.eye(4)
        if e.children:
            src = el.expression("Offset")
            if ctx is None:
                return np.eye(4)
            return ctx.ev.matrix(src)
        if len(e.args) >= 16:
            return X.matrix_from_vc(e.args)
        return np.eye(4)

    def _dof_motion(self, el: El, ctx: Optional[CompContext]) -> np.ndarray:
        d = el.first("Dof")
        if d is None or ctx is None:
            return np.eye(4)
        typ = d.arg
        try:
            val = float(d.value("JointValue", "0"))
        except ValueError:
            val = 0.0
        ctx.locals.append({"VALUE": val, "Position": val, "Value": val})
        try:
            if typ == "Custom":
                src = d.expression("Joint")
                return ctx.ev.matrix(src) if src else np.eye(4)
            defl = d.expression("JointDeflection")
            q = ctx.ev.num(defl, val) if defl else val
            if abs(q) < 1e-12:
                return np.eye(4)
            try:
                ax = _AXES.get(int(d.value("AxisType", "2")), None)
            except ValueError:
                ax = None
            if ax is None:
                self.warn(f"[{ctx.name}] неизвестный тип оси сочленения {d.value('AxisType')} — принята ось Z")
                ax = (0, 0, 1)
            if typ == "Rotational":
                return X.axis_angle(ax[0], ax[1], ax[2], q)
            if typ == "Translational":
                return X.T(ax[0] * q, ax[1] * q, ax[2] * q)
            return np.eye(4)
        finally:
            ctx.locals.pop()

    def _component_color(self, nc: Optional[El]):
        if nc is None:
            return None
        for sa in nc.kids("SimAttribute"):
            mat = sa.first("Material")
            if mat is not None:
                return self._material_color(mat)
        return None

    def _material_color(self, mat: El):
        c = _diffuse_of(mat) if mat.children else None
        if c:
            return c
        nm = mat.arg
        if nm in self.materials:
            return self.materials[nm]
        return palette_color(nm)

    # ---------------------------------------------------------- features
    def _visible(self, f: El) -> bool:
        if self.opt.include_hidden:
            return True
        v = f.value("Visible", "1")
        return v not in ("0", "false", "False")

    def _features(self, feats: List[El], acc: np.ndarray, ctx: CompContext, vcid, node: SceneNode,
                  comp: str, color):
        for f in feats:
            self._feature(f, acc, ctx, vcid, node, comp, color)

    def _feature(self, f: El, acc: np.ndarray, ctx: CompContext, vcid, node: SceneNode, comp: str, color):
        typ = f.arg
        fname = f.name or typ
        self.stats["features"] += 1
        if not self._visible(f):
            return
        mat_el = f.first("Material")
        if mat_el is not None and (mat_el.arg or mat_el.children):
            c = self._material_color(mat_el)
            if c:
                color = c
        m = acc @ _mat_of(f)
        kids = list(f.kids("Feature"))
        if typ == "rTransformFeature":
            src = f.expression("Transform")
            if src:
                m = m @ ctx.ev.matrix(src)
            self._features(kids, m, ctx, vcid, node, comp, color)
        elif typ == "rSwitchFeature":
            src = f.expression("ValExpression") or ""
            if not self.opt.include_envelopes and ("WorkSpace::" in src or "envelope" in src.lower()):
                return
            val = ctx.ev.eval(src, 1.0) if src else 1.0
            chosen: List[El] = []
            if isinstance(val, str):
                chosen = [k for k in kids if k.name == val]
            else:
                try:
                    iv = int(round(float(val)))
                except (TypeError, ValueError):
                    iv = 1
                if 1 <= iv <= len(kids):
                    chosen = [kids[iv - 1]]
            self._features(chosen, m, ctx, vcid, node, comp, color)
        elif typ == "rLinearCloneFeature":
            cnt = max(0, int(round(ctx.num(f, "CountExpression", 1))))
            ctx.locals.append({"Count": float(cnt)})
            step = ctx.num(f, "StepExpression", 0.0)
            ctx.locals[-1]["Step"] = step
            d = ctx.vector(f, "DirectionExpression", (1, 0, 0))
            n = np.linalg.norm(d)
            d = d / n if n > 1e-12 else np.array([1.0, 0, 0])
            for i in range(cnt):
                ctx.locals[-1]["Index"] = float(i)
                off = d * step * i
                self._features(kids, m @ X.T(*off), ctx, vcid, node, comp, color)
            ctx.locals.pop()
        elif typ == "rAngularCloneFeature":
            cnt = max(0, int(round(ctx.num(f, "CountExpression", 1))))
            ctx.locals.append({"Count": float(cnt)})
            step = ctx.num(f, "StepExpression", 360.0 / max(cnt, 1))
            ctx.locals[-1]["Step"] = step
            o = ctx.vector(f, "OffsetExpression", (0, 0, 0))
            for i in range(cnt):
                ctx.locals[-1]["Index"] = float(i)
                r = X.T(*o) @ X.R("z", step * i) @ X.T(*(-o))
                self._features(kids, m @ r, ctx, vcid, node, comp, color)
            ctx.locals.pop()
        elif typ == "rMirrorFeature":
            d = ctx.vector(f, "DirectionExpression", (1, 1, -1))
            hide = f.value("HideOriginal", "0") in ("1", "true")
            if not hide:
                self._features(kids, m, ctx, vcid, node, comp, color)
            s = np.eye(4)
            if np.all(np.abs(np.abs(d) - 1.0) < 1e-9):
                s[:3, :3] = np.diag(np.sign(d))  # зеркалирование масштабом (-1 по осям)
            else:
                from .brep import mirror_matrix
                s = mirror_matrix(d)
            self._features(kids, m @ s, ctx, vcid, node, comp, color)
        elif typ == "rGeoFeature":
            uri = f.value("Uri") or _var_value(f, "Uri")
            self._add_part(node, fname, GeoRef(("geo", vcid or "", uri), "geo", {"uri": uri, "vcid": vcid}),
                           m, color, comp)
            if kids:
                self._features(kids, m, ctx, vcid, node, comp, color)
        elif typ in _PRIMS:
            g = self._primitive(f, ctx)
            if g is not None:
                self._add_part(node, fname, g, m, color, comp)
            if kids:
                self._features(kids, m, ctx, vcid, node, comp, color)
        elif typ in ("rExtrudeFeature", "rRevolveFeature"):
            profile = self._collect_profile(kids, ctx, vcid, comp)
            if not profile:
                self.warn(f"[{comp}] {fname}: нет профиля для операции {typ}")
                return
            if typ == "rExtrudeFeature":
                params = {
                    "profile": profile,
                    "dir": tuple(_r(x) for x in ctx.vector(f, "DirectionExpression", (0, 0, 1))),
                    "length": _r(ctx.num(f, "LengthExpression", 0.0)),
                    "reverse": f.value("Reverse", "0") == "1",
                }
                kind = "extrude"
            else:
                params = {
                    "profile": profile,
                    "axis": tuple(_r(x) for x in ctx.vector(f, "RevolveAxis", (0, 0, 1))),
                    "angle": _r(ctx.num(f, "RevolveAngle", 360.0)),
                    "offset": _r(ctx.num(f, "RevolveOffset", 0.0)),
                    "reverse": f.value("Reverse", "0") == "1",
                }
                kind = "revolve"
            key = (kind, _profile_key(profile)) + tuple(
                (k, v) for k, v in sorted(params.items()) if k != "profile")
            self._add_part(node, fname, GeoRef(key, kind, params), m, color, comp)
        elif typ in _IGNORED:
            return
        else:
            if kids:
                self.warn(f"[{comp}] тип элемента «{typ}» не поддерживается — обработаны только вложенные элементы")
                self._features(kids, m, ctx, vcid, node, comp, color)
            else:
                self.warn(f"[{comp}] тип элемента «{typ}» ({fname}) не поддерживается и пропущен")

    def _collect_profile(self, feats: List[El], ctx: CompContext, vcid, comp: str):
        """Список (uri, vcid, матрица-кортеж) геометрий профиля, вложенных в Extrude/Revolve."""
        tmp = SceneNode("tmp", "group", np.eye(4))
        self._features(feats, np.eye(4), ctx, vcid, tmp, comp, None)
        out = []
        for p in tmp.parts:
            out.append((p.geo, tuple(_r(x) for x in p.matrix.reshape(-1))))
        # убираем части профиля из статистики деталей
        self.stats["parts"] -= len(tmp.parts)
        return out

    def _primitive(self, f: El, ctx: CompContext) -> Optional[GeoRef]:
        t = f.arg
        if t == "rPrimitiveBoxFeature":
            L, W, H = (ctx.num(f, k, 0.0) for k in ("Length", "Width", "Height"))
            return GeoRef(("box", _r(L), _r(W), _r(H)), "box", {"L": L, "W": W, "H": H})
        if t == "rPrimitiveCylinderFeature":
            r, h = ctx.num(f, "Radius", 0.0), ctx.num(f, "Height", 0.0)
            a0, a1 = ctx.num(f, "StartSweep", 0.0), ctx.num(f, "EndSweep", 360.0)
            return GeoRef(("cyl", _r(r), _r(h), _r(a0), _r(a1)), "cyl", {"r": r, "h": h, "a0": a0, "a1": a1})
        if t == "rPrimitiveConeFeature":
            r1, r2, h = ctx.num(f, "BottomRadius", 0.0), ctx.num(f, "TopRadius", 0.0), ctx.num(f, "Height", 0.0)
            a0, a1 = ctx.num(f, "StartSweep", 0.0), ctx.num(f, "EndSweep", 360.0)
            return GeoRef(("cone", _r(r1), _r(r2), _r(h), _r(a0), _r(a1)), "cone",
                          {"r1": r1, "r2": r2, "h": h, "a0": a0, "a1": a1})
        if t == "rPrimitiveSphereFeature":
            r = ctx.num(f, "Radius", 0.0)
            a0, a1 = ctx.num(f, "StartSweep", 0.0), ctx.num(f, "EndSweep", 360.0)
            return GeoRef(("sphere", _r(r), _r(a0), _r(a1)), "sphere", {"r": r, "a0": a0, "a1": a1})
        if t == "rPrimitiveWedgeFeature":
            p = {k: ctx.num(f, k, 0.0) for k in ("BiggerLength", "SmallerLength", "BiggerHeight", "SmallerHeight", "Width")}
            p["MirrorBack"] = f.value("MirrorBack", "0") == "1"
            p["MirrorDown"] = f.value("MirrorDown", "0") == "1"
            return GeoRef(("wedge",) + tuple((k, _r(v) if isinstance(v, float) else v) for k, v in sorted(p.items())),
                          "wedge", p)
        return None

    def _add_part(self, node: SceneNode, name: str, geo: GeoRef, m: np.ndarray, color, comp: str):
        node.parts.append(Part(name=name, geo=geo, matrix=m, color=color, source=comp))
        self.stats["parts"] += 1


def _var_value(el: El, name: str) -> str:
    """Значение переменной ``name`` из VariableSpace элемента (запасной путь чтения Uri)."""
    vs = el.first("VariableSpace")
    if vs is None:
        return ""
    for v in vs.kids("Variable"):
        if v.value("Name") == name:
            return v.value("Value")
    return ""


def _hidden_component(el: El) -> bool:
    vs = el.first("VariableSpace")
    if vs is None:
        return False
    for v in vs.kids("Variable"):
        if v.name == "Visible":
            return v.value("Value", "1").lower() in ("0", "false")
    return False


def _profile_key(profile) -> str:
    h = hashlib.sha1()
    for g, mt in profile:
        h.update(repr(g.key).encode())
        h.update(repr(mt).encode())
    return h.hexdigest()


def tree_signature(n: SceneNode, cache: Dict[int, str]) -> str:
    """Подпись поддерева: одинаковые поддеревья -> один экземпляр сборки в STEP."""
    s = cache.get(id(n))
    if s is not None:
        return s
    h = hashlib.sha1()
    h.update(n.name.encode("utf-8", "ignore"))
    h.update(n.kind.encode())
    for p in n.parts:
        h.update(repr(p.geo.key).encode())
        h.update(np.round(p.matrix, 6).tobytes())
        h.update(repr(p.color and tuple(round(c, 3) for c in p.color)).encode())
        h.update(p.name.encode("utf-8", "ignore"))
    for c in n.children:
        h.update(tree_signature(c, cache).encode())
        h.update(np.round(c.matrix, 6).tobytes())
    s = h.hexdigest()
    cache[id(n)] = s
    return s


__all__ = ["SceneBuilder", "SceneNode", "Part", "GeoRef", "Issue", "BuildOptions", "parse_materials",
           "tree_signature", "palette_color"]
_ = math
