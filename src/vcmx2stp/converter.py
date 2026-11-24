"""Конвертация .vcmx -> .stp: оркестрация, отчёт, предупреждения."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .archive import VcmxArchive
from .model import BuildOptions, Issue, SceneBuilder, parse_materials
from .rsc_parser import parse_bytes
from .step_writer import ShapeFactory, StepWriter


@dataclass
class ConvertOptions:
    search_dirs: List[str] = field(default_factory=list)   # папки с исходными .vcmx (eCatalog, My Models ...)
    include_hidden: bool = False
    include_envelopes: bool = False
    colors: bool = True
    angle_tol: float = 0.5                                   # допуск компланарности, градусы
    keep_polylines: bool = False
    schema: str = "AP214IS"


@dataclass
class ConvertResult:
    output: str
    ok: bool
    size_bytes: int = 0
    components: int = 0
    parts: int = 0
    prototypes: int = 0
    instances: int = 0
    triangles_in: int = 0
    faces_out: int = 0
    missing: List[Issue] = field(default_factory=list)
    errors: List[Issue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    empty_components: List[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def has_missing(self) -> bool:
        return bool(self.missing)

    @property
    def needs_attention(self) -> bool:
        return bool(self.missing or self.empty_components)

    def missing_message(self) -> str:
        if not self.missing:
            if self.empty_components:
                return self._empty_message()
            return ""
        lines = ["В файле не хватает встроенных моделей.",
                 "Следующие объекты ссылаются на геометрию, которой нет в .vcmx "
                 "(сохраните layout в KUKA.Sim с опцией «Include Components»/«Pack and Go» "
                 "или укажите папку с исходными компонентами):", ""]
        seen = set()
        for i in self.missing:
            k = (i.component, i.uri)
            if k in seen:
                continue
            seen.add(k)
            lines.append(f"  - {i.component} / {i.feature}: {i.uri} - {i.reason}")
        if self.empty_components:
            lines += ["", self._empty_message()]
        return "\n".join(lines)

    def _empty_message(self) -> str:
        names = sorted(set(self.empty_components))
        more = f" и ещё {len(names) - 30}" if len(names) > 30 else ""
        return ("Компоненты без геометрии (в файле нет ни встроенной модели, ни ссылки на неё; "
                "возможно, модели не были встроены при сохранении): " + ", ".join(names[:30]) + more)

    def summary(self) -> str:
        s = [
            f"Файл: {self.output}",
            f"Размер: {self.size_bytes / 1024:.1f} КБ",
            f"Компонентов: {self.components}, деталей: {self.parts} "
            f"(уникальных тел: {self.prototypes}, экземпляров: {self.instances})",
        ]
        if self.triangles_in:
            s.append(f"Сетка: {self.triangles_in} треугольников -> {self.faces_out} плоских граней B-rep")
        s.append(f"Время: {self.seconds:.1f} с")
        if self.missing:
            s.append(f"ВНИМАНИЕ: отсутствует геометрия для {len({(i.component, i.uri) for i in self.missing})} объектов")
        return "\n".join(s)


def quiet_occt():
    """Отключает служебный вывод OpenCASCADE в консоль."""
    try:
        from OCP.Message import Message, Message_Gravity

        ps = Message.DefaultMessenger_s().Printers()
        for i in range(1, ps.Size() + 1):
            ps.Value(i).SetTraceLevel(Message_Gravity.Message_Fail)
    except Exception:  # pragma: no cover
        pass


def default_output(path: str) -> str:
    return os.path.splitext(path)[0] + ".stp"


def convert(src: str, dst: Optional[str] = None, options: Optional[ConvertOptions] = None,
            log: Callable[[str], None] = print) -> ConvertResult:
    t0 = time.time()
    quiet_occt()
    opt = options or ConvertOptions()
    dst = dst or default_output(src)
    log(f"Чтение {src}")
    with VcmxArchive(src, opt.search_dirs, log=log) as ar:
        els = parse_bytes(ar.rsc_bytes)
        materials = {}
        mdat = ar.root.get("materials.dat")
        if mdat:
            try:
                materials.update(parse_materials(parse_bytes(mdat)))
            except Exception:  # pragma: no cover
                pass
        materials.update({k: v for k, v in parse_materials(els).items() if k not in materials})
        name = os.path.splitext(os.path.basename(src))[0]
        sb = SceneBuilder(ar, materials, BuildOptions(opt.include_hidden, opt.include_envelopes), log=log)
        root = sb.build(els, name)
        log(f"Компонентов: {sb.stats['components']}, звеньев: {sb.stats['links']}, деталей: {sb.stats['parts']}")
        issues: List[Issue] = []
        errors: List[Issue] = []
        factory = ShapeFactory(ar, log, issues, opt.angle_tol, opt.keep_polylines, errors)
        writer = StepWriter(factory, log, opt.colors, opt.schema)
        log("Построение B-rep и структуры сборок...")
        writer.build(root)
        log(f"Запись STEP: {dst}")
        writer.write(dst)
    res = ConvertResult(
        output=dst, ok=True, size_bytes=os.path.getsize(dst), components=sb.stats["components"],
        parts=sb.stats["parts"], prototypes=writer.prototypes, instances=writer.instances,
        triangles_in=factory.mesh_stats.triangles_in, faces_out=factory.mesh_stats.faces_out,
        missing=issues, errors=errors, warnings=sb.warnings, empty_components=sb.empty_components,
        seconds=time.time() - t0,
    )
    for e in res.errors:
        log(f"Ошибка построения: {e.component} / {e.feature}: {e.reason}")
    if res.needs_attention:
        log(res.missing_message())
    log(res.summary())
    return res
