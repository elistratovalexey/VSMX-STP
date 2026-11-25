"""Компактная запись STEP (ISO 10303-21) после OpenCascade.

Что делается (геометрия и структура сборок не меняются):
  * координаты точек округляются до 1e-5 мм (исходные сетки KUKA.Sim хранятся во float32,
    их «шум» вида 159.000000381 не несёт информации), направления — до 1e-10;
  * одинаковые геометрические сущности (CARTESIAN_POINT, DIRECTION, VECTOR,
    AXIS2_PLACEMENT_3D, LINE, PLANE, CIRCLE и т.п.) записываются один раз,
    ссылки перенаправляются на единственный экземпляр;
  * сущности перенумеровываются подряд (короче ссылки #N);
  * лишние пробелы, переносы строк и нули удаляются.
Строковые литералы ('...') не изменяются. Топологические сущности (вершины, рёбра, грани)
не объединяются — это изменило бы модель.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

_ENTITY_RE = re.compile(r"#(\d+)\s*=\s*", re.S)
_REF_RE = re.compile(r"#(\d+)")
_NUM_RE = re.compile(r"(?<![#\w.])[-+]?\d+\.\d*(?:[Ee][-+]?\d+)?")
_STR_SPLIT = re.compile(r"('(?:[^']|'')*')")

# только чистая геометрия (без топологии и без семантики сборок)
_DEDUP_TYPES = {
    "CARTESIAN_POINT", "DIRECTION", "VECTOR", "AXIS2_PLACEMENT_3D", "AXIS1_PLACEMENT", "LINE",
    "PLANE", "CIRCLE", "ELLIPSE", "CYLINDRICAL_SURFACE", "CONICAL_SURFACE", "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE", "COLOUR_RGB", "DRAUGHTING_PRE_DEFINED_CURVE_FONT", "FILL_AREA_STYLE_COLOUR",
    "FILL_AREA_STYLE", "SURFACE_STYLE_FILL_AREA", "SURFACE_SIDE_STYLE", "SURFACE_STYLE_USAGE",
    "CURVE_STYLE", "PRESENTATION_STYLE_ASSIGNMENT",
}


def _fmt(x: float, decimals: int) -> str:
    s = f"{round(x, decimals):.{decimals}f}".rstrip("0")
    if s in ("-0.", "+0."):
        s = "0."
    return s


def _round_numbers(body: str, decimals: int) -> str:
    return _NUM_RE.sub(lambda m: _fmt(float(m.group(0)), decimals), body)


def _trim_zeros(body: str) -> str:
    # 1.500000 -> 1.5 ; 2.000 -> 2.
    return re.sub(r"(\d\.\d*?)0+(?=[,)E])", r"\1", body)


def _split_entities(data: str) -> Tuple[List[Tuple[int, str]], bool]:
    """Разбивает секцию DATA на (номер, тело) с учётом строк."""
    parts = _STR_SPLIT.split(data)
    # собираем текст, заменяя строки маркерами, чтобы ';' внутри строк не мешали
    strings: List[str] = []
    buf = []
    for i, p in enumerate(parts):
        if i % 2:
            buf.append(f"\x00{len(strings)}\x00")
            strings.append(p)
        else:
            buf.append(p)
    text = "".join(buf)
    out: List[Tuple[int, str]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _ENTITY_RE.match(chunk)
        if not m:
            return [], False
        body = chunk[m.end():]
        body = re.sub(r"\s+", "", body)  # пробелы вне строк не значимы
        body = re.sub(r"\x00(\d+)\x00", lambda mm: strings[int(mm.group(1))], body)
        out.append((int(m.group(1)), body))
    return out, True


def _etype(body: str) -> str:
    i = body.find("(")
    return body[:i] if i > 0 else ""


def compact_text(text: str) -> str:
    head, sep, rest = text.partition("DATA;")
    if not sep:
        return text
    data, sep2, tail = rest.partition("ENDSEC;")
    if not sep2:
        return text
    ents, ok = _split_entities(data)
    if not ok or not ents:
        return text

    bodies: Dict[int, str] = {}
    order: List[int] = []
    for eid, body in ents:
        t = _etype(body)
        if t == "CARTESIAN_POINT":
            body = _round_numbers(body, 5)
        elif t == "DIRECTION":
            body = _round_numbers(body, 10)
        else:
            segs = _STR_SPLIT.split(body)
            segs = [s if k % 2 else _trim_zeros(s) for k, s in enumerate(segs)]
            body = "".join(segs)
        bodies[eid] = body
        order.append(eid)

    # объединение одинаковых геометрических сущностей (несколько проходов: точки -> векторы -> оси ...)
    alias: Dict[int, int] = {}

    def resolve(i: int) -> int:
        while i in alias:
            i = alias[i]
        return i

    def rewrite(body: str) -> str:
        segs = _STR_SPLIT.split(body)
        for k in range(0, len(segs), 2):
            segs[k] = _REF_RE.sub(lambda m: "#" + str(resolve(int(m.group(1)))), segs[k])
        return "".join(segs)

    for _ in range(6):
        seen: Dict[str, int] = {}
        changed = False
        for eid in order:
            if eid in alias:
                continue
            b = bodies[eid]
            if _etype(b) not in _DEDUP_TYPES:
                continue
            b = rewrite(b)
            bodies[eid] = b
            first = seen.get(b)
            if first is None:
                seen[b] = eid
            else:
                alias[eid] = first
                changed = True
        if not changed:
            break

    # перенумерация подряд
    keep = [eid for eid in order if eid not in alias]
    newid = {eid: k + 1 for k, eid in enumerate(keep)}

    def renum(body: str) -> str:
        segs = _STR_SPLIT.split(body)
        for k in range(0, len(segs), 2):
            segs[k] = _REF_RE.sub(lambda m: "#" + str(newid.get(resolve(int(m.group(1))), int(m.group(1)))), segs[k])
        return "".join(segs)

    lines = [f"#{newid[eid]}={renum(bodies[eid])};" for eid in keep]
    return head + "DATA;\n" + "\n".join(lines) + "\nENDSEC;" + tail


def compact_file(path: str) -> None:
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("latin-1")
    out = compact_text(text)
    with open(path, "wb") as f:
        f.write(out.encode("latin-1"))
