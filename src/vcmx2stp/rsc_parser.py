"""Парсер текстового формата ресурсов Visual Components (*.rsc).

Формат представляет собой дерево вида::

    Node "rSimResource"
    {
    Name "Robot"
    Location 1 0 0 0  0 1 0 0  0 0 1 0  100 200 0 1
    Feature "rGeoFeature"
    {
    Uri "vcid:...#base"
    }
    }

Строка = ключ + аргументы. Если за строкой следует ``{``, элемент получает
список дочерних элементов. Внутри строк ``\\`` + перевод строки означает
перенос, ``\\"`` — экранированная кавычка.
"""
from __future__ import annotations

from typing import Iterator, List, Optional


class El:
    """Элемент дерева RSC."""

    __slots__ = ("key", "args", "children")

    def __init__(self, key: str, args: List[str], children: Optional[List["El"]] = None):
        self.key = key
        self.args = args
        self.children = children

    # ---- удобные методы -------------------------------------------------
    @property
    def arg(self) -> str:
        return self.args[0] if self.args else ""

    def kids(self, key: Optional[str] = None) -> Iterator["El"]:
        for c in self.children or ():
            if key is None or c.key == key:
                yield c

    def first(self, key: str) -> Optional["El"]:
        for c in self.children or ():
            if c.key == key:
                return c
        return None

    def value(self, key: str, default: str = "") -> str:
        e = self.first(key)
        return e.arg if e is not None and e.args else default

    def values(self, key: str) -> List[str]:
        e = self.first(key)
        return list(e.args) if e is not None else []

    def expression(self, key: str) -> Optional[str]:
        """Возвращает выражение поля вида ``Key { Expression "..." }`` или ``Key value``."""
        e = self.first(key)
        if e is None:
            return None
        ex = e.first("Expression")
        if ex is not None:
            return " ".join(ex.args)
        if e.args:
            return " ".join(e.args)
        return ""

    @property
    def name(self) -> str:
        return self.value("Name")

    def __repr__(self) -> str:  # pragma: no cover
        return f"El({self.key!r}, {self.args!r}, {'[...]' if self.children is not None else None})"


def _tokenize(text: str):
    i = 0
    n = len(text)
    out = []
    append = out.append
    ws = " \t\r\f\v"
    stop = " \t\r\n{}\"\f\v"
    while i < n:
        c = text[i]
        if c == "\n":
            append((0, None))
            i += 1
        elif c in ws:
            i += 1
        elif c == '"':
            i += 1
            buf = []
            while i < n:
                c = text[i]
                if c == "\\" and i + 1 < n:
                    nx = text[i + 1]
                    if nx == "\n":
                        i += 2
                        continue
                    if nx == "\r" and i + 2 < n and text[i + 2] == "\n":
                        i += 3
                        continue
                    buf.append(nx)
                    i += 2
                    continue
                if c == '"':
                    i += 1
                    break
                buf.append(c)
                i += 1
            append((1, "".join(buf)))
        elif c == "{":
            append((2, None))
            i += 1
        elif c == "}":
            append((3, None))
            i += 1
        else:
            j = i
            while j < n and text[j] not in stop:
                j += 1
            append((1, text[i:j]))
            i = j
    return out


def parse(text: str) -> List[El]:
    """Разбирает текст RSC и возвращает список элементов верхнего уровня."""
    toks = _tokenize(text)
    root: List[El] = []
    stack: List[List[El]] = [root]
    line: List[str] = []

    def flush():
        if line:
            stack[-1].append(El(line[0], line[1:]))
            line.clear()

    for t, v in toks:
        if t == 1:
            line.append(v)
        elif t == 0:
            if line:
                flush()
        elif t == 2:
            flush()
            cur = stack[-1]
            if cur and cur[-1].children is None:
                cur[-1].children = []
                stack.append(cur[-1].children)
            else:
                e = El("", [], [])
                cur.append(e)
                stack.append(e.children)
        else:
            flush()
            if len(stack) > 1:
                stack.pop()
    flush()
    return root


def parse_bytes(data: bytes) -> List[El]:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover
        text = data.decode("utf-8", errors="replace")
    # Первая строка вида "VCMD0028040100000000COMPONENT" — сигнатура, отбрасываем.
    if text.startswith("VCMD"):
        nl = text.find("\n")
        text = text[nl + 1:] if nl >= 0 else ""
    return parse(text)


def walk(els: List[El]) -> Iterator[El]:
    stack = list(reversed(els))
    while stack:
        e = stack.pop()
        yield e
        if e.children:
            stack.extend(reversed(e.children))
