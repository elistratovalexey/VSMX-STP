"""Проверка и установка обновлений из GitHub Releases.

Workflow GitHub Actions при публикации тега vX.Y.Z создаёт Release и прикладывает
установщик ``VCMX2STP-Setup-X.Y.Z.exe``. Программа сравнивает свою версию с
последним релизом, скачивает установщик и запускает его в тихом режиме —
Inno Setup обновляет программу поверх (AppId неизменен), настройки сохраняются.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from . import __version__

API = "https://api.github.com/repos/{repo}/releases/latest"
UA = f"VCMX2STP/{__version__}"


@dataclass
class UpdateInfo:
    version: str
    tag: str
    url: str           # ссылка на установщик
    size: int
    notes: str
    page: str


def parse_version(v: str) -> Tuple[int, ...]:
    nums = re.findall(r"\d+", v or "")
    return tuple(int(x) for x in nums[:4]) or (0,)


def is_newer(remote: str, local: str = __version__) -> bool:
    return parse_version(remote) > parse_version(local)


def _get(url: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/vnd.github+json"})
    token = os.environ.get("VCMX2STP_GITHUB_TOKEN")  # для приватных репозиториев (необязательно)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def check(repo: str) -> Optional[UpdateInfo]:
    """Возвращает информацию о более новой версии или None."""
    if not repo or "/" not in repo or repo.startswith("OWNER/"):
        raise ValueError("Не задан репозиторий GitHub для обновлений (формат: владелец/репозиторий)")
    data = json.loads(_get(API.format(repo=repo)).decode("utf-8"))
    tag = data.get("tag_name", "")
    ver = tag.lstrip("vV")
    if not is_newer(ver):
        return None
    asset = None
    for a in data.get("assets", []):
        n = a.get("name", "").lower()
        if n.endswith(".exe") and "setup" in n:
            asset = a
            break
    if asset is None:
        raise RuntimeError(f"В релизе {tag} нет файла установщика (*Setup*.exe)")
    return UpdateInfo(version=ver, tag=tag, url=asset["browser_download_url"], size=int(asset.get("size", 0)),
                      notes=data.get("body", "") or "", page=data.get("html_url", ""))


def download(info: UpdateInfo, progress: Optional[Callable[[int, int], None]] = None) -> str:
    path = os.path.join(tempfile.gettempdir(), f"VCMX2STP-Setup-{info.version}.exe")
    req = urllib.request.Request(info.url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r, open(path + ".part", "wb") as f:
        total = int(r.headers.get("Content-Length", info.size or 0))
        done = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    os.replace(path + ".part", path)
    return path


def run_installer(path: str, silent: bool = True) -> None:
    """Запускает установщик и завершает текущий процесс (установщик закроет/заменит файлы)."""
    args = [path]
    if silent:
        args += ["/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS", "/RELAUNCH"]
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(args, close_fds=True, creationflags=flags)
    os._exit(0)
