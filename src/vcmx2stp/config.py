"""Настройки приложения (хранятся в %APPDATA%\\VCMX2STP\\settings.json)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List

# Репозиторий GitHub, из которого берутся обновления (Releases).
# Значение подставляется workflow при сборке (переменная GITHUB_REPOSITORY),
# его также можно изменить в настройках программы.
try:
    from ._build_info import GITHUB_REPO as _DEFAULT_REPO  # type: ignore
except ImportError:  # pragma: no cover - локальный запуск из исходников
    _DEFAULT_REPO = "OWNER/vcmx2stp"


@dataclass
class Settings:
    github_repo: str = _DEFAULT_REPO
    check_updates_on_start: bool = True
    search_dirs: List[str] = field(default_factory=list)
    include_hidden: bool = False
    include_envelopes: bool = False
    colors: bool = True
    angle_tol: float = 0.5
    last_dir: str = ""
    output_dir: str = ""


def settings_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~/.config")
    d = os.path.join(base, "VCMX2STP")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "settings.json")


def default_search_dirs() -> List[str]:
    """Типичные расположения eCatalog / My Models KUKA.Sim и Visual Components."""
    out = []
    pub = os.environ.get("PUBLIC", r"C:\Users\Public")
    docs = os.path.join(os.path.expanduser("~"), "Documents")
    for base in (os.path.join(pub, "Documents"), docs):
        for vendor in ("KUKA.Sim", "KUKA", "Visual Components"):
            p = os.path.join(base, vendor)
            if os.path.isdir(p):
                out.append(p)
    return out


def load() -> Settings:
    s = Settings()
    try:
        with open(settings_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(s, k):
                setattr(s, k, v)
    except (OSError, ValueError):
        s.search_dirs = default_search_dirs()
    if not s.github_repo or s.github_repo == "OWNER/vcmx2stp":
        s.github_repo = _DEFAULT_REPO
    return s


def save(s: Settings) -> None:
    try:
        with open(settings_path(), "w", encoding="utf-8") as f:
            json.dump(asdict(s), f, ensure_ascii=False, indent=2)
    except OSError:  # pragma: no cover
        pass
