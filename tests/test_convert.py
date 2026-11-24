"""Сквозная проверка: синтетический .vcmx -> STEP."""
import os
import re

import numpy as np
import pytest

from make_sample import MISSING_VCID, build
from vcmx2stp.brep import count_faces, mesh_to_brep
from vcmx2stp.converter import ConvertOptions, convert


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    d = tmp_path_factory.mktemp("conv")
    src = build(str(d / "sample.vcmx"))
    dst = str(d / "sample.stp")
    res = convert(src, dst, ConvertOptions(), log=lambda *_: None)
    with open(dst, encoding="latin-1") as f:
        text = f.read()
    return res, dst, text


def test_output_written(converted):
    res, dst, _ = converted
    assert res.ok
    assert os.path.getsize(dst) > 1000


def test_no_triangles(converted):
    _, _, text = converted
    assert not re.search(r"TRIANGULATED|TESSELLATED|POLY_LOOP", text)
    assert "ADVANCED_FACE" in text or "MANIFOLD_SOLID_BREP" in text


def test_assembly_structure(converted):
    _, _, text = converted
    assert text.count("NEXT_ASSEMBLY_USAGE_OCCURRENCE") >= 4
    for name in ("TestCell", "J1", "Base", "Column", "EmbeddedCube", "Arm"):
        assert f"'{name}'" in text, name


def test_missing_warning(converted):
    res, _, _ = converted
    assert res.has_missing
    msg = res.missing_message()
    assert msg.startswith("В файле не хватает встроенных моделей.")
    assert MISSING_VCID in msg
    assert "MissingPart" in msg


def test_cube_mesh_becomes_six_faces():
    s = 10.0
    v = np.array([(0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0), (0, 0, s), (s, 0, s), (s, s, s), (0, s, s)], float)
    f = np.array([(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
                  (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)])
    shape = mesh_to_brep(v, f)
    assert shape is not None
    assert count_faces(shape) == 6
