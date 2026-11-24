"""Чтение архива .vcmx и поиск встроенных файлов геометрии.

Структура архива (ZIP):
  layout.rsc / component.rsc      — дерево компонентов, узлов и features;
  component.dat                   — метаданные (VcId) самого файла;
  package.dat                     — список встроенных компонентов (папки component, component_1 ...);
  component_N/component.vcm       — вложенный ZIP с исходным компонентом и его геометрией;
  geo-<uuid>                      — встроенная геометрия (формат 3DS).

Ссылки на геометрию в rsc:
  Uri "geo-<uuid>"                — файл в корне архива;
  Uri "vcid:<VCID>#<имя>"         — файл <имя> в компоненте с данным VCID;
  Uri "openbeam"                  — файл рядом с rsc (внутри component.vcm);
  Uri "C:/путь/деталь.stp"        — внешний файл.
"""
from __future__ import annotations

import io
import json
import os
import re
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple

_VCID_RE = re.compile(r'VcId\s+"?([0-9a-fA-F-]{36})"?')


def _read_vcid(data: bytes) -> Optional[str]:
    try:
        m = _VCID_RE.search(data.decode("utf-8", errors="ignore"))
    except Exception:  # pragma: no cover
        return None
    return m.group(1).lower() if m else None


class _Store:
    """Набор файлов (имя в нижнем регистре -> функция чтения)."""

    def __init__(self, label: str):
        self.label = label
        self.files: Dict[str, Tuple[zipfile.ZipFile, str]] = {}

    def add_zip(self, zf: zipfile.ZipFile, prefix: str = ""):
        for info in zf.infolist():
            if info.is_dir():
                continue
            n = info.filename
            if prefix and not n.startswith(prefix):
                continue
            rel = n[len(prefix):]
            if "/" in rel:
                continue
            self.files.setdefault(rel.lower(), (zf, n))

    def get(self, name: str) -> Optional[bytes]:
        key = name.replace("\\", "/").split("/")[-1].lower()
        ent = self.files.get(key)
        if ent is None:
            return None
        zf, n = ent
        return zf.read(n)

    def has(self, name: str) -> bool:
        return name.replace("\\", "/").split("/")[-1].lower() in self.files


class VcmxArchive:
    """Открытый .vcmx: основной rsc + индекс встроенной геометрии."""

    def __init__(self, path: str, search_dirs: Iterable[str] = (), log=print):
        self.path = path
        self.log = log
        self._zips: List[zipfile.ZipFile] = []
        zf = zipfile.ZipFile(path)
        self._zips.append(zf)
        self.root = _Store("<root>")
        self.root.add_zip(zf)
        self.by_vcid: Dict[str, _Store] = {}
        self.packages: List[Tuple[str, str]] = []  # (ecat-имя, папка)
        self.rsc_name, self.rsc_bytes = self._find_main_rsc(zf)
        self.is_layout = self.rsc_name.lower().endswith("layout.rsc")
        own = self.root.get("component.dat")
        if own:
            vid = _read_vcid(own)
            if vid:
                self.by_vcid[vid] = self.root
        self._index_nested(zf)
        self.search_dirs = [d for d in search_dirs if d and os.path.isdir(d)]
        self._external_index: Optional[Dict[str, str]] = None
        self._external_cache: Dict[str, _Store] = {}

    # ------------------------------------------------------------------
    def close(self):
        for z in self._zips:
            try:
                z.close()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ------------------------------------------------------------------
    @staticmethod
    def _find_main_rsc(zf: zipfile.ZipFile) -> Tuple[str, bytes]:
        names = zf.namelist()
        for cand in ("layout.rsc", "component.rsc"):
            for n in names:
                if n.lower() == cand:
                    return n, zf.read(n)
        # model.xml -> ModelUrl
        for n in names:
            if n.lower() == "model.xml":
                m = re.search(rb"<ModelUrl>([^<]+)</ModelUrl>", zf.read(n))
                if m:
                    mu = m.group(1).decode("utf-8", "ignore").strip()
                    for k in names:
                        if k.lower() == mu.lower():
                            return k, zf.read(k)
        rscs = [n for n in names if n.lower().endswith(".rsc") and "/" not in n]
        if rscs:
            return rscs[0], zf.read(rscs[0])
        raise ValueError("В архиве .vcmx не найден layout.rsc / component.rsc — файл повреждён или имеет неизвестный формат")

    def _index_nested(self, zf: zipfile.ZipFile):
        # package.dat
        pd = self.root.get("package.dat")
        if pd:
            for m in re.finditer(rb'<File[^>]*ecat="([^"]*)"[^>]*file="([^"]*)"', pd):
                self.packages.append((m.group(1).decode("utf-8", "ignore"), m.group(2).decode("utf-8", "ignore")))
        folders = set()
        for n in zf.namelist():
            if "/" in n:
                folders.add(n.split("/")[0])
        for folder in sorted(folders):
            prefix = folder + "/"
            store = _Store(folder)
            store.add_zip(zf, prefix)
            vid = None
            dat = store.get("component.dat")
            if dat:
                vid = _read_vcid(dat)
            vcm_name = None
            for k in list(store.files):
                if k.endswith(".vcm") or k.endswith(".vcmx"):
                    vcm_name = k
            if vcm_name:
                try:
                    inner = zipfile.ZipFile(io.BytesIO(store.get(vcm_name)))
                    self._zips.append(inner)
                    store.add_zip(inner)
                    if not vid:
                        d2 = store.get("component.dat")
                        vid = _read_vcid(d2) if d2 else None
                except zipfile.BadZipFile:
                    self.log(f"Предупреждение: вложенный файл {folder}/{vcm_name} повреждён")
            if vid:
                self.by_vcid.setdefault(vid, store)

    # ------------------------------------------------------------------
    def _external(self, vcid: str) -> Optional[_Store]:
        if not self.search_dirs:
            return None
        if self._external_index is None:
            self._external_index = build_external_index(self.search_dirs, self.log)
        p = self._external_index.get(vcid)
        if not p:
            return None
        if p in self._external_cache:
            return self._external_cache[p]
        try:
            zf = zipfile.ZipFile(p)
        except (OSError, zipfile.BadZipFile):
            return None
        self._zips.append(zf)
        st = _Store(p)
        st.add_zip(zf)
        self._external_cache[p] = st
        return st

    def resolve(self, uri: str, context_vcid: Optional[str] = None) -> Tuple[Optional[bytes], str]:
        """Возвращает (данные, описание источника) или (None, причина)."""
        u = (uri or "").strip()
        if not u:
            return None, "пустая ссылка Uri"
        if u.lower().startswith("vcid:"):
            body = u[5:]
            vcid, _, name = body.partition("#")
            vcid = vcid.strip().lower()
            name = name.strip()
            st = self.by_vcid.get(vcid)
            if st is not None and name:
                d = st.get(name)
                if d is not None:
                    return d, f"встроено ({st.label})"
            d = self.root.get(name) if name else None
            if d is not None:
                return d, "встроено (корень архива)"
            for s in self.by_vcid.values():
                d = s.get(name) if name else None
                if d is not None:
                    return d, f"встроено ({s.label})"
            ext = self._external(vcid)
            if ext is not None and name:
                d = ext.get(name)
                if d is not None:
                    return d, f"внешний файл {ext.label}"
            return None, f"компонент VCID {vcid} не встроен в файл"
        if u.lower().startswith("file:"):
            u = re.sub(r"^file:/*", "", u)
        # внешний абсолютный путь
        if re.match(r"^[A-Za-z]:[\\/]", u) or u.startswith("\\\\") or u.startswith("/"):
            if os.path.isfile(u):
                with open(u, "rb") as f:
                    return f.read(), f"внешний файл {u}"
            base = os.path.basename(u.replace("\\", "/"))
            d = self.root.get(base)
            if d is not None:
                return d, "встроено (корень архива)"
            return None, f"внешний файл {u} не найден"
        name = u.split("#")[-1]
        if context_vcid and context_vcid.lower() in self.by_vcid:
            d = self.by_vcid[context_vcid.lower()].get(name)
            if d is not None:
                return d, "встроено"
        d = self.root.get(name)
        if d is not None:
            return d, "встроено (корень архива)"
        for s in self.by_vcid.values():
            d = s.get(name)
            if d is not None:
                return d, f"встроено ({s.label})"
        for st in self._external_cache.values():
            d = st.get(name)
            if d is not None:
                return d, f"внешний файл {st.label}"
        return None, "файл геометрии отсутствует в архиве"


# ---------------------------------------------------------------------------
def _index_cache_path() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    d = os.path.join(base, "VCMX2STP")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "vcid_index.json")


def build_external_index(dirs: Iterable[str], log=print) -> Dict[str, str]:
    """Индекс VCID -> путь к .vcmx в папках поиска (кэшируется по времени изменения файлов)."""
    cache_file = _index_cache_path()
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    files = cache.get("files", {})
    result: Dict[str, str] = {}
    changed = False
    for d in dirs:
        for root, _dirs, fnames in os.walk(d):
            for fn in fnames:
                if not fn.lower().endswith((".vcmx", ".vcm")):
                    continue
                p = os.path.join(root, fn)
                try:
                    mt = os.path.getmtime(p)
                except OSError:
                    continue
                ent = files.get(p)
                if ent and abs(ent.get("mtime", 0) - mt) < 1e-6:
                    vid = ent.get("vcid")
                else:
                    vid = None
                    try:
                        with zipfile.ZipFile(p) as zf:
                            for n in zf.namelist():
                                if n.lower() == "component.dat":
                                    vid = _read_vcid(zf.read(n))
                                    break
                    except (OSError, zipfile.BadZipFile):
                        pass
                    files[p] = {"mtime": mt, "vcid": vid}
                    changed = True
                if vid:
                    result.setdefault(vid, p)
    if changed:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"files": files}, f)
        except OSError:  # pragma: no cover
            pass
    log(f"Папки поиска: найдено {len(result)} компонентов .vcmx")
    return result
