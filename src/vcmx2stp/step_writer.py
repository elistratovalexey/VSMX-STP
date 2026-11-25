"""Создание геометрии деталей и запись STEP со структурой сборок (OpenCASCADE XDE)."""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import brep, smartbrep, step_compact
from . import geometry as G
from .model import GeoRef, Issue, Part, SceneNode, tree_signature


# ============================================================ фабрика форм
class ShapeFactory:
    """Строит и кэширует B-rep для GeoRef (один прототип на одинаковую геометрию)."""

    def __init__(self, archive, log: Callable[[str], None], issues: List[Issue], angle_tol: float = 0.5,
                 keep_polylines: bool = False, errors: Optional[List[Issue]] = None,
                 surface_fit: bool = True, fit_tol: float = 0.1):
        self.ar = archive
        self.errors = errors if errors is not None else []
        self.log = log
        self.issues = issues
        self.angle_tol = angle_tol
        self.keep_polylines = keep_polylines
        self.surface_fit = surface_fit
        self.fit_tol = fit_tol
        self.fit_stats = smartbrep.FitStats()
        self._raw: Dict[tuple, Tuple[str, object]] = {}
        self._shape: Dict[tuple, Optional[object]] = {}
        self._color: Dict[tuple, Optional[tuple]] = {}
        self.mesh_stats = brep.MeshStats()
        self.exact_count = 0

    # ---------------------------------------------------------- загрузка
    def _load_geo(self, geo: GeoRef, comp: str, fname: str):
        uri = geo.params.get("uri", "")
        vcid = geo.params.get("vcid")
        k = ("raw", vcid or "", uri)
        if k in self._raw:
            return self._raw[k]
        data, how = self.ar.resolve(uri, vcid)
        if data is None:
            self.issues.append(Issue(comp, fname, uri, how))
            res = ("none", how)
        else:
            res = G.load(data)
            if res[0] == "none":
                self.issues.append(Issue(comp, fname, uri, str(res[1])))
        self._raw[k] = res
        return res

    def color_of(self, geo: GeoRef) -> Optional[tuple]:
        return self._color.get(geo.key)

    def shape(self, geo: GeoRef, comp: str = "", fname: str = "", profile: bool = False,
              lin: Optional[np.ndarray] = None):
        """B-rep геометрии; ``lin`` — дополнительная (в т.ч. нежёсткая) матрица, «запекаемая» в форму."""
        if lin is not None and np.allclose(lin, np.eye(4), atol=1e-12):
            lin = None
        lkey = None if lin is None else tuple(np.round(lin, 6).reshape(-1))
        key = geo.key + (("profile",) if profile else ()) + ((lkey,) if lkey else ())
        if key in self._shape:
            return self._shape[key]
        s = None
        try:
            s = self._build(geo, comp, fname, profile, lin)
        except Exception as ex:  # защитный код: одна деталь не должна ломать конвертацию
            self.log(f"Ошибка построения «{fname}» ({comp}): {ex}")
            self.errors.append(Issue(comp, fname, geo.params.get("uri", geo.kind), f"ошибка построения: {ex}"))
        if s is not None:
            s = _relax_tolerance(s)
        self._shape[key] = s
        return s

    def _build(self, geo: GeoRef, comp: str, fname: str, profile: bool, lin: Optional[np.ndarray]):
        k, p = geo.kind, geo.params
        if k == "geo":
            return self._build_geo(geo, comp, fname, profile, lin)
        base = self._build_exact(geo, comp, fname)
        if base is None or lin is None:
            return base
        return brep.apply_matrix(base, lin)

    def _build_geo(self, geo: GeoRef, comp: str, fname: str, profile: bool, lin: Optional[np.ndarray]):
        if True:
            kind, obj = self._load_geo(geo, comp, fname)
            if kind == "brep":
                self.exact_count += 1
                return obj if lin is None else brep.apply_matrix(obj, lin)
            if kind != "mesh":
                return None
            mesh = obj
            if lin is not None:  # нежёсткое преобразование — к вершинам сетки (грани остаются плоскими)
                mesh = _transform_mesh(mesh, lin)
            c = mesh.dominant_color()
            if c:
                self._color[geo.key] = c
            shapes = []
            if profile:
                s = None
                if mesh.polylines:
                    s = brep.polylines_to_shape(mesh.polylines)
                if s is None and len(mesh.triangles):
                    s = brep.mesh_to_brep(mesh.vertices, mesh.triangles, self.angle_tol, None)
                if s is not None:
                    shapes.append(s)
            else:
                if len(mesh.triangles):
                    s = None
                    if self.surface_fit:
                        try:
                            s = smartbrep.mesh_to_brep_fit(mesh.vertices, mesh.triangles, self.fit_tol, self.angle_tol,
                                                           self.mesh_stats, self.fit_stats)
                        except Exception as ex:  # защитный код: при сбое — многогранный построитель
                            self.log(f"Распознавание поверхностей «{fname}»: {ex}")
                            s = None
                    if s is None:
                        s = brep.mesh_to_brep(mesh.vertices, mesh.triangles, self.angle_tol, self.mesh_stats)
                    if s is not None:
                        shapes.append(s)
                if self.keep_polylines and mesh.polylines:
                    s = brep.polylines_to_shape(mesh.polylines)
                    if s is not None:
                        shapes.append(s)
            if not shapes:
                return None
            return shapes[0] if len(shapes) == 1 else brep.compound(shapes)

    def _build_exact(self, geo: GeoRef, comp: str, fname: str):
        k, p = geo.kind, geo.params
        self.exact_count += 1
        if k == "box":
            return brep.make_block(p["L"], p["W"], p["H"])
        if k == "cyl":
            return brep.make_cylinder(p["r"], p["h"], p["a0"], p["a1"])
        if k == "cone":
            return brep.make_cone(p["r1"], p["r2"], p["h"], p["a0"], p["a1"])
        if k == "sphere":
            return brep.make_sphere(p["r"], p["a0"], p["a1"])
        if k == "wedge":
            return brep.make_wedge(p["BiggerLength"], p["SmallerLength"], p["BiggerHeight"], p["SmallerHeight"],
                                   p["Width"], p["MirrorBack"], p["MirrorDown"])
        if k in ("extrude", "revolve"):
            prof = self._profile_shape(p["profile"], comp, fname)
            if prof is None:
                return None
            if k == "extrude":
                d = np.array(p["dir"], float)
                if p.get("reverse"):
                    d = -d
                return brep.extrude(prof, d, p["length"])
            if p.get("offset"):
                prof = brep.apply_matrix(prof, _tmat(p["offset"], 0, 0))
            ang = -p["angle"] if p.get("reverse") else p["angle"]
            return brep.revolve(prof, np.array(p["axis"], float), ang)
        return None

    def _profile_shape(self, profile, comp, fname):
        shapes = []
        for g, mt in profile:
            m = np.array(mt, float).reshape(4, 4)
            s = self.shape(g, comp, fname, profile=True, lin=m)
            if s is not None:
                shapes.append(s)
        if not shapes:
            return None
        return shapes[0] if len(shapes) == 1 else brep.compound(shapes)


def _transform_mesh(mesh, m: np.ndarray):
    from dataclasses import replace

    v = mesh.vertices @ m[:3, :3].T + m[:3, 3]
    tris = mesh.triangles
    if np.linalg.det(m[:3, :3]) < 0 and len(tris):
        tris = tris[:, ::-1].copy()
    polys = [p @ m[:3, :3].T + m[:3, 3] for p in mesh.polylines]
    return replace(mesh, vertices=v, triangles=tris, polylines=polys)


def _tmat(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def split_matrix(m: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """M = Rigid * A. Возвращает (жёсткая часть, нежёсткая линейная часть или None)."""
    a = m[:3, :3]
    if brep.is_rigid(m, 1e-6):
        return m, None
    u, s, vt = np.linalg.svd(a)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    rest = r.T @ a
    rig = np.eye(4)
    rig[:3, :3] = r
    rig[:3, 3] = m[:3, 3]
    lin = np.eye(4)
    lin[:3, :3] = rest
    if np.allclose(rest, np.eye(3), atol=1e-6):
        return rig, None
    return rig, lin


# ============================================================ запись STEP
class StepWriter:
    def __init__(self, factory: ShapeFactory, log: Callable[[str], None], colors: bool = True,
                 schema: str = "AP214IS"):
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDocStd import TDocStd_Document
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import XCAFDoc_DocumentTool

        self.f = factory
        self.log = log
        self.colors = colors
        self.schema = schema
        self.app = XCAFApp_Application.GetApplication_s()
        self.doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
        self.app.InitDocument(self.doc)
        self.st = XCAFDoc_DocumentTool.ShapeTool_s(self.doc.Main())
        self.ct = XCAFDoc_DocumentTool.ColorTool_s(self.doc.Main())
        self._proto: Dict[tuple, object] = {}
        self._asm: Dict[str, object] = {}
        self._sig: Dict[int, str] = {}
        self.instances = 0
        self.prototypes = 0

    # ---------------------------------------------------------- утилиты
    @staticmethod
    def _loc(m: np.ndarray):
        from OCP.TopLoc import TopLoc_Location

        if np.allclose(m, np.eye(4), atol=1e-12):
            return TopLoc_Location()
        return TopLoc_Location(brep.trsf_from_matrix(m))

    @staticmethod
    def _name(label, name: str):
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDataStd import TDataStd_Name

        TDataStd_Name.Set_s(label, TCollection_ExtendedString(name or "unnamed", True))

    def _set_color(self, label, c):
        if not self.colors or c is None:
            return
        from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
        from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf

        col = Quantity_Color(float(c[0]), float(c[1]), float(c[2]), Quantity_TOC_RGB)
        self.ct.SetColor(label, col, XCAFDoc_ColorSurf)
        self.ct.SetColor(label, col, XCAFDoc_ColorGen)

    # ---------------------------------------------------------- прототипы
    def _prototype(self, part: Part, lin: Optional[np.ndarray]):
        color = part.color or self.f.color_of(part.geo)
        lkey = None if lin is None else tuple(np.round(lin[:3, :3], 6).reshape(-1))
        ckey = None if color is None else tuple(round(x, 3) for x in color)
        key = (part.geo.key, lkey, ckey)
        if key in self._proto:
            return self._proto[key]
        s = self.f.shape(part.geo, part.source, part.name, lin=lin)
        if s is None:
            self._proto[key] = None
            return None
        label = self.st.AddShape(s, False, False)
        self._name(label, part.name)
        self._set_color(label, color)
        self._proto[key] = label
        self.prototypes += 1
        return label

    # ---------------------------------------------------------- сборки
    def _assembly(self, node: SceneNode):
        sig = tree_signature(node, self._sig)
        if sig in self._asm:
            return self._asm[sig]
        label = self.st.NewShape()
        self._name(label, node.name)
        added = 0
        for p in node.parts:
            rig, lin = split_matrix(p.matrix)
            proto = self._prototype(p, lin)
            if proto is None:
                continue
            comp = self.st.AddComponent(label, proto, self._loc(rig))
            self._name(comp, p.name)
            added += 1
            self.instances += 1
        for ch in node.children:
            sub = self._assembly(ch)
            if sub is None:
                continue
            rig, lin = split_matrix(ch.matrix)
            if lin is not None:
                self.log(f"Узел «{ch.name}» имеет масштаб/зеркалирование в матрице положения — "
                         "оно применено к геометрии деталей узла")
            comp = self.st.AddComponent(label, sub, self._loc(rig))
            self._name(comp, ch.name)
            added += 1
        if added == 0:
            self._asm[sig] = None
            return None
        self._asm[sig] = label
        return label

    def build(self, root: SceneNode):
        _push_down_nonrigid(root)
        lab = self._assembly(root)
        self.st.UpdateAssemblies()
        return lab

    # ---------------------------------------------------------- запись
    def write(self, path: str):
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.Interface import Interface_Static
        from OCP.STEPCAFControl import STEPCAFControl_Writer
        from OCP.STEPControl import STEPControl_AsIs

        w = STEPCAFControl_Writer()  # создание writer инициализирует параметры Interface_Static
        _set(Interface_Static.SetCVal_s("write.step.schema", self.schema))
        _set(Interface_Static.SetIVal_s("write.surfacecurve.mode", 0))   # без p-кривых — меньше размер
        _set(Interface_Static.SetCVal_s("write.step.unit", "MM"))
        _set(Interface_Static.SetIVal_s("write.step.assembly", 1))
        _set(Interface_Static.SetCVal_s("write.step.product.name", "VCMX2STP"))
        try:
            _set(Interface_Static.SetIVal_s("write.step.tessellated", 0))  # никаких треугольников (OCCT >= 7.6)
        except Exception:  # pragma: no cover
            pass
        w.SetColorMode(self.colors)
        w.SetNameMode(True)
        w.SetLayerMode(False)
        w.SetPropsMode(False)
        if not w.Transfer(self.doc, STEPControl_AsIs):
            raise RuntimeError("Не удалось передать модель в STEP")
        # OCCT плохо работает с не-ASCII путями в Windows: пишем во временный файл и переносим
        fd, tmp = tempfile.mkstemp(suffix=".stp")
        os.close(fd)
        try:
            st = w.Write(tmp)
            if st != IFSelect_RetDone:
                raise RuntimeError("Ошибка записи STEP")
            if not os.environ.get("VCMX2STP_NO_COMPACT"):
                step_compact.compact_file(tmp)
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            shutil.move(tmp, path)
            try:
                os.chmod(path, 0o644)
            except OSError:  # pragma: no cover
                pass
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def _set(ok):
    return ok


def _push_down_nonrigid(node: SceneNode):
    """Нежёсткие матрицы узлов (Sy(-1) и т.п.) переносятся вниз — на детали и дочерние узлы."""
    for ch in node.children:
        rig, lin = split_matrix(ch.matrix)
        if lin is not None:
            ch.matrix = rig
            for p in ch.parts:
                p.matrix = lin @ p.matrix
            for g in ch.children:
                g.matrix = lin @ g.matrix
        _push_down_nonrigid(ch)


def _relax_tolerance(shape, tmin: float = 1e-4):
    """Минимальный допуск 1e-4 мм: запись координат с точностью 1e-5 мм не нарушает корректность."""
    try:
        from OCP.ShapeFix import ShapeFix_ShapeTolerance
        ShapeFix_ShapeTolerance().LimitTolerance(shape, tmin, 1e6)
    except Exception:  # pragma: no cover
        pass
    return shape
