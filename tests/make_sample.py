"""Создание синтетического .vcmx для автотестов и проверки сборки в GitHub Actions.

Файл содержит:
  * примитивы (Block, Cylinder);
  * встроенную 3DS-геометрию (куб) — rGeoFeature с Uri "geo-...";
  * ссылку на НЕвстроенный компонент (vcid:...#part) — должно появиться предупреждение;
  * дочернее звено rSimLink с вращательной осью.

Запуск: python tests/make_sample.py путь/sample.vcmx
"""
from __future__ import annotations

import struct
import sys
import zipfile

GEO_NAME = "geo-11111111-2222-3333-4444-555555555555"
MISSING_VCID = "deadbeef-0000-4000-8000-000000000001"
COMP_VCID = "0a1b2c3d-1111-4222-8333-444455556666"


def _chunk(cid: int, body: bytes) -> bytes:
    return struct.pack("<HI", cid, 6 + len(body)) + body


def cube_3ds(size: float = 100.0) -> bytes:
    s = size
    v = [(0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0), (0, 0, s), (s, 0, s), (s, s, s), (0, s, s)]
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    vert = struct.pack("<H", len(v)) + b"".join(struct.pack("<3f", *p) for p in v)
    face = struct.pack("<H", len(f)) + b"".join(struct.pack("<4H", a, b, c, 0) for a, b, c in f)
    tri = _chunk(0x4100, _chunk(0x4110, vert) + _chunk(0x4120, face))
    obj = _chunk(0x4000, b"cube\0" + tri)
    return _chunk(0x4D4D, _chunk(0x3D3D, obj))


def _geo(name: str, uri: str, matrix: str = "") -> str:
    m = f"Matrix {matrix}\n" if matrix else ""
    return f'''Feature "rGeoFeature"
{{
Name "{name}"
{m}Visible 1
VariableSpace
{{
  Variable "rTVariable<rUri>"
  {{
    Name "Uri"
    Value "{uri}"
  }}
}}
Uri "{uri}"
}}
'''


RSC = f'''VCMD0028040100000000COMPONENT
Node "rSimResource"
{{
Name "TestCell"
Id 1
NodeClass
{{
Id 1
Feature "rPrimitiveBoxFeature"
{{
Name "Base"
Visible 1
Length
{{
  Expression "400"
}}
Width
{{
  Expression "300"
}}
Height
{{
  Expression "50"
}}
}}
Feature "rTransformFeature"
{{
Name "Up"
Matrix 1 0 0 0 0 1 0 0 0 0 1 0 0 0 50 1
Visible 1
Feature "rPrimitiveCylinderFeature"
{{
Name "Column"
Visible 1
Radius
{{
  Expression "40"
}}
Height
{{
  Expression "200"
}}
}}
}}
{_geo("EmbeddedCube", GEO_NAME, "1 0 0 0 0 1 0 0 0 0 1 0 100 50 50 1")}
{_geo("MissingPart", "vcid:" + MISSING_VCID + "#part_1")}
VCID "{COMP_VCID}"
}}
VCID {COMP_VCID}
Location 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
Node "rSimLink"
{{
Name "J1"
Id 2
NodeClass
{{
Id 2
Feature "rPrimitiveBoxFeature"
{{
Name "Arm"
Visible 1
Length
{{
  Expression "300"
}}
Width
{{
  Expression "40"
}}
Height
{{
  Expression "40"
}}
}}
}}
Location 1 0 0 0 0 1 0 0 0 0 1 0 0 0 250 1
}}
}}
'''

COMPONENT_DAT = f'VcId "{COMP_VCID}"\nRevision 1\n'


def build(path: str) -> str:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("component.rsc", RSC)
        z.writestr("component.dat", COMPONENT_DAT)
        z.writestr(GEO_NAME, cube_3ds())
    return path


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "sample.vcmx"))
