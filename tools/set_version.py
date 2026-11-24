"""Подготовка версии для сборки в GitHub Actions.

  python tools/set_version.py --ref refs/tags/v1.2.3 --repo user/vcmx2stp --run 15

* тег vX.Y.Z  -> версия X.Y.Z;
* без тега    -> версия из src/vcmx2stp/_version.py + ".devN" не используется (Inno/Windows
  требуют числовую версию), поэтому берётся X.Y.Z.N, где N — номер запуска workflow.
Записывает src/vcmx2stp/_version.py, src/vcmx2stp/_build_info.py, build/version_info.txt
и выводит version=... в GITHUB_OUTPUT.
"""
from __future__ import annotations

import argparse
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG = os.path.join(ROOT, "src", "vcmx2stp")


def current_version() -> str:
    with open(os.path.join(PKG, "_version.py"), encoding="utf-8") as f:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', f.read())
    return m.group(1) if m else "0.0.0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "OWNER/vcmx2stp"))
    ap.add_argument("--run", default=os.environ.get("GITHUB_RUN_NUMBER", "0"))
    a = ap.parse_args()

    m = re.match(r"^refs/tags/v(\d+\.\d+\.\d+)$", a.ref or "")
    if m:
        version, release = m.group(1), True
    else:
        base = re.match(r"(\d+\.\d+\.\d+)", current_version())
        version, release = f"{base.group(1) if base else '0.0.0'}.{int(a.run or 0)}", False

    with open(os.path.join(PKG, "_version.py"), "w", encoding="utf-8") as f:
        f.write(f'"""Версия программы (перезаписывается при сборке)."""\n__version__ = "{version}"\n')
    with open(os.path.join(PKG, "_build_info.py"), "w", encoding="utf-8") as f:
        f.write('"""Создаётся tools/set_version.py при сборке в GitHub Actions."""\n'
                f'GITHUB_REPO = "{a.repo}"\nRELEASE_BUILD = {release}\n')

    nums = [int(x) for x in version.split(".")] + [0, 0, 0, 0]
    tup = ", ".join(str(x) for x in nums[:4])
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    with open(os.path.join(ROOT, "build", "version_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers=({tup}), prodvers=({tup}), mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('041904B0', [
      StringStruct('CompanyName', 'VCMX2STP'),
      StringStruct('FileDescription', 'Конвертер KUKA.Sim VCMX в STEP'),
      StringStruct('FileVersion', '{version}'),
      StringStruct('InternalName', 'VCMX2STP'),
      StringStruct('OriginalFilename', 'VCMX2STP.exe'),
      StringStruct('ProductName', 'VCMX2STP'),
      StringStruct('ProductVersion', '{version}')])]),
    VarFileInfo([VarStruct('Translation', [0x0419, 1200])])
  ]
)
""")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"version={version}\nrelease={'true' if release else 'false'}\n")
    print(f"version={version} release={release} repo={a.repo}")


if __name__ == "__main__":
    main()
