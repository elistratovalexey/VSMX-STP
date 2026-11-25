"""Проверка распознавания аналитических поверхностей (цилиндр из треугольной сетки)."""
import numpy as np
import pytest

pytest.importorskip("OCP")

from vcmx2stp import smartbrep  # noqa: E402


def _cylinder_mesh(r=20.0, h=50.0, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(a), r * np.sin(a)], 1)
    v = np.vstack([np.c_[ring, np.zeros(n)], np.c_[ring, np.full(n, h)], [[0, 0, 0], [0, 0, h]]])
    c0, c1 = 2 * n, 2 * n + 1
    t = []
    for i in range(n):
        j = (i + 1) % n
        t += [[i, j, n + j], [i, n + j, n + i], [c0, j, i], [c1, n + i, n + j]]
    return v, np.array(t, dtype=np.int64)


def test_cylinder_recognized():
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import TopoDS
    from OCP.BRepCheck import BRepCheck_Analyzer

    v, t = _cylinder_mesh()
    fs = smartbrep.FitStats()
    shape = smartbrep.mesh_to_brep_fit(v, t, tol=0.1, fstats=fs)
    assert shape is not None
    assert BRepCheck_Analyzer(shape).IsValid()
    kinds = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        srf = BRep_Tool.Surface_s(TopoDS.Face(ex.Current()))
        if hasattr(srf, "BasisSurface"):
            srf = srf.BasisSurface()
        kinds.append(srf.__class__.__name__)
        ex.Next()
    # 2 торца + боковая цилиндрическая поверхность вместо 96 треугольников
    assert len(kinds) <= 4
    assert any("Cylindrical" in k for k in kinds)
