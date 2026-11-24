"""Вычислитель выражений Visual Components.

Поддерживаются числа с единицами измерения (``100 {mm}``, ``45 {deg}``),
строки, переменные с вкладками (``Advanced::Width``), арифметика,
сравнения, логика, тернарный оператор, матричные функции
``Tx/Ty/Tz/Rx/Ry/Rz/Sx/Sy/Sz`` и их цепочки ``Tx(10).Rz(45)``,
векторы ``Vector(x,y,z)``, тригонометрия и прочие математические функции.
Матрицы 4x4 хранятся как ``numpy.ndarray`` (столбцы N, O, A, P — как в VC).
"""
from __future__ import annotations

import math
import re
from typing import Callable, Dict, Optional

import numpy as np


class ExprError(Exception):
    pass


class Vec(np.ndarray):
    """Вектор (4 компоненты)."""


def vec(x=0.0, y=0.0, z=0.0, w=1.0) -> np.ndarray:
    return np.array([float(x), float(y), float(z), float(w)]).view(Vec)


def is_mat(v) -> bool:
    return isinstance(v, np.ndarray) and v.shape == (4, 4)


def is_vec(v) -> bool:
    return isinstance(v, np.ndarray) and v.shape == (4,)


# ------------------------------------------------------------------ матрицы
def ident() -> np.ndarray:
    return np.eye(4)


def T(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    m = np.eye(4)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def R(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    if axis == "x":
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    elif axis == "y":
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    else:
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def S(x=1.0, y=1.0, z=1.0) -> np.ndarray:
    return np.diag([x, y, z, 1.0])


def axis_angle(i, j, k, deg) -> np.ndarray:
    v = np.array([i, j, k], float)
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.eye(4)
    x, y, z = v / n
    a = math.radians(deg)
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    m = np.eye(4)
    m[:3, :3] = [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]
    return m


def ijk(i, j, k) -> np.ndarray:
    z = np.array([i, j, k], float)
    n = np.linalg.norm(z)
    if n < 1e-15:
        return np.eye(4)
    z /= n
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = x, y, z
    return m


def quaternion(q1, q2, q3, q4) -> np.ndarray:
    w, x, y, z = q1, q2, q3, q4
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return m


def matrix_from_vc(vals) -> np.ndarray:
    """16 чисел VC (N, O, A, P по 4) -> матрица 4x4."""
    a = np.array([float(v) for v in vals][:16], float)
    if a.size != 16:
        return np.eye(4)
    return a.reshape(4, 4).T.copy()


# ------------------------------------------------------------------ лексер
_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<unit>\{[^}]*\})
  | (?P<id>[A-Za-z_\u0400-\u04FF][A-Za-z0-9_\u0400-\u04FF]*(?:::[A-Za-z0-9_\u0400-\u04FF]+)*)
  | (?P<op>==|!=|<=|>=|&&|\|\||[-+*/%^()<>!?:,.\[\]=])
    """,
    re.VERBOSE,
)

_UNITS = {
    "mm": 1.0, "m": 1000.0, "cm": 10.0, "dm": 100.0, "km": 1e6, "in": 25.4, "inch": 25.4,
    "ft": 304.8, "um": 1e-3, "deg": 1.0, "rad": 180.0 / math.pi, "s": 1.0, "ms": 1e-3,
    "min": 60.0, "h": 3600.0, "kg": 1.0, "g": 1e-3, "%": 0.01, "mm/s": 1.0, "deg/s": 1.0,
}


def _tokenize(src: str):
    pos = 0
    out = []
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise ExprError(f"Недопустимый символ {src[pos]!r} в выражении {src!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        out.append((kind, m.group()))
    out.append(("end", ""))
    return out


# ------------------------------------------------------------------ функции
def _num(v) -> float:
    if isinstance(v, (bool, np.bool_)):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return 0.0
    if is_vec(v):
        return float(np.linalg.norm(v[:3]))
    raise ExprError(f"Ожидалось число, получено {type(v).__name__}")


def _truth(v) -> bool:
    if isinstance(v, str):
        return v not in ("", "0", "false", "False")
    if isinstance(v, np.ndarray):
        return True
    return abs(_num(v)) > 1e-12


def _wpr(w, p, r):
    return R("z", r) @ R("y", p) @ R("x", w)


def _euler(a, b, c):
    return R("z", a) @ R("y", b) @ R("z", c)


def _set16(*v):
    return matrix_from_vc(v)


def _mean_dist(*a):
    # распределения VC: uniform(stream,min,max), normal(stream,mean,dev) ...
    nums = [_num(x) for x in a]
    if len(nums) >= 3:
        return (nums[1] + nums[2]) / 2.0
    if len(nums) == 2:
        return nums[1]
    return nums[0] if nums else 0.0


def _first_arg_dist(*a):
    nums = [_num(x) for x in a]
    return nums[1] if len(nums) >= 2 else (nums[0] if nums else 0.0)


_FUNCS: Dict[str, Callable] = {
    # матрицы
    "tx": lambda d: T(_num(d), 0, 0), "ty": lambda d: T(0, _num(d), 0), "tz": lambda d: T(0, 0, _num(d)),
    "rx": lambda a: R("x", _num(a)), "ry": lambda a: R("y", _num(a)), "rz": lambda a: R("z", _num(a)),
    "sx": lambda f: S(_num(f), 1, 1), "sy": lambda f: S(1, _num(f), 1), "sz": lambda f: S(1, 1, _num(f)),
    "scale": lambda *f: S(*([_num(x) for x in f] * 3)[:3]) if len(f) == 1 else S(*[_num(x) for x in f][:3]),
    "pos": lambda x, y, z: T(_num(x), _num(y), _num(z)),
    "identity": lambda: ident(),
    "euler": lambda a, b, c: _euler(_num(a), _num(b), _num(c)),
    "wpr": lambda w, p, r: _wpr(_num(w), _num(p), _num(r)),
    "ijk": lambda i, j, k: ijk(_num(i), _num(j), _num(k)),
    "axisangle": lambda i, j, k, a: axis_angle(_num(i), _num(j), _num(k), _num(a)),
    "quaternion": lambda a, b, c, d: quaternion(_num(a), _num(b), _num(c), _num(d)),
    "set": _set16,
    "matrix": lambda *v: _set16(*v) if len(v) == 16 else ident(),
    "vector": lambda x=0, y=0, z=0, w=1: vec(_num(x), _num(y), _num(z), _num(w)),
    # тригонометрия (радианы по умолчанию, суффикс d — градусы)
    "sin": lambda a: math.sin(_num(a)), "cos": lambda a: math.cos(_num(a)), "tan": lambda a: math.tan(_num(a)),
    "asin": lambda a: math.asin(max(-1, min(1, _num(a)))), "acos": lambda a: math.acos(max(-1, min(1, _num(a)))),
    "atan": lambda a: math.atan(_num(a)), "atan2": lambda a, b: math.atan2(_num(a), _num(b)),
    "sind": lambda a: math.sin(math.radians(_num(a))), "cosd": lambda a: math.cos(math.radians(_num(a))),
    "tand": lambda a: math.tan(math.radians(_num(a))),
    "asind": lambda a: math.degrees(math.asin(max(-1, min(1, _num(a))))),
    "acosd": lambda a: math.degrees(math.acos(max(-1, min(1, _num(a))))),
    "atand": lambda a: math.degrees(math.atan(_num(a))),
    "atan2d": lambda a, b: math.degrees(math.atan2(_num(a), _num(b))),
    "sinh": lambda a: math.sinh(_num(a)), "cosh": lambda a: math.cosh(_num(a)), "tanh": lambda a: math.tanh(_num(a)),
    "degree": lambda a: math.degrees(_num(a)), "radian": lambda a: math.radians(_num(a)),
    "degrees": lambda a: math.degrees(_num(a)), "radians": lambda a: math.radians(_num(a)),
    # математика
    "sqrt": lambda a: math.sqrt(max(0.0, _num(a))), "abs": lambda a: abs(_num(a)), "fabs": lambda a: abs(_num(a)),
    "pow": lambda a, b: _num(a) ** _num(b), "exp": lambda a: math.exp(_num(a)),
    "log": lambda a: math.log(_num(a)) if _num(a) > 0 else 0.0,
    "log10": lambda a: math.log10(_num(a)) if _num(a) > 0 else 0.0,
    "ln": lambda a: math.log(_num(a)) if _num(a) > 0 else 0.0,
    "floor": lambda a: float(math.floor(_num(a))), "ceil": lambda a: float(math.ceil(_num(a))),
    "round": lambda a, n=0: float(round(_num(a), int(_num(n)))), "int": lambda a: float(int(_num(a))),
    "trunc": lambda a: float(int(_num(a))), "fmod": lambda a, b: math.fmod(_num(a), _num(b)) if _num(b) else 0.0,
    "mod": lambda a, b: math.fmod(_num(a), _num(b)) if _num(b) else 0.0,
    "min": lambda *a: min(_num(x) for x in a), "max": lambda *a: max(_num(x) for x in a),
    "sign": lambda a: math.copysign(1.0, _num(a)) if _num(a) else 0.0,
    "if": lambda c, a, b=0.0: a if _truth(c) else b,
    "clamp": lambda v, lo, hi: max(_num(lo), min(_num(hi), _num(v))),
    # статистические распределения — берётся среднее значение
    "uniform": _mean_dist, "normal": _first_arg_dist, "exponential": _first_arg_dist,
    "triangular": lambda *a: (sum(_num(x) for x in a[1:4]) / 3.0) if len(a) >= 4 else _mean_dist(*a),
    "gamma": _first_arg_dist, "erlang": _first_arg_dist, "lognormal": _first_arg_dist,
    "weibull": _first_arg_dist, "beta": _first_arg_dist, "constant": lambda *a: _num(a[-1]) if a else 0.0,
}

_CONSTS = {"pi": math.pi, "e": math.e, "true": 1.0, "false": 0.0, "inf": float("inf")}


class Evaluator:
    """Вычисляет выражения в контексте переменных компонента.

    ``resolver(name)`` возвращает значение переменной или ``None``.
    """

    def __init__(self, resolver: Callable[[str], object], warn: Optional[Callable[[str], None]] = None):
        self.resolver = resolver
        self.warn = warn or (lambda m: None)
        self._unknown = set()

    # -- API ----------------------------------------------------------------
    def eval(self, src: Optional[str], default=0.0):
        if src is None:
            return default
        s = src.strip()
        if not s:
            return default
        saved = (getattr(self, "_toks", None), getattr(self, "_i", 0))  # реентерабельность
        try:
            self._toks = _tokenize(s)
            self._i = 0
            v = self._ternary()
            if self._peek()[0] != "end":
                raise ExprError(f"Лишние символы в выражении {s!r}")
            return v
        except (ExprError, ZeroDivisionError, OverflowError, ValueError, TypeError, IndexError) as ex:
            self.warn(f"Не удалось вычислить выражение «{s}»: {ex}")
            return default
        finally:
            self._toks, self._i = saved

    def num(self, src, default=0.0) -> float:
        v = self.eval(src, default)
        try:
            return _num(v)
        except ExprError:
            return default

    def matrix(self, src) -> np.ndarray:
        v = self.eval(src, None)
        if v is None:
            return np.eye(4)
        if is_mat(v):
            return v
        if is_vec(v):
            return T(v[0], v[1], v[2])
        self.warn(f"Выражение «{src}» не является матрицей")
        return np.eye(4)

    def vector(self, src, default=(0.0, 0.0, 1.0)) -> np.ndarray:
        v = self.eval(src, None)
        if is_vec(v):
            return np.array(v[:3], float)
        if is_mat(v):
            return v[:3, 3].copy()
        return np.array(default, float)

    # -- парсер ------------------------------------------------------------
    def _peek(self):
        return self._toks[self._i]

    def _next(self):
        t = self._toks[self._i]
        self._i += 1
        return t

    def _accept(self, op):
        t = self._peek()
        if t[0] == "op" and t[1] == op:
            self._i += 1
            return True
        return False

    def _expect(self, op):
        if not self._accept(op):
            raise ExprError(f"Ожидалось «{op}»")

    def _ternary(self):
        c = self._or()
        if self._accept("?"):
            a = self._ternary()
            self._expect(":")
            b = self._ternary()
            return a if _truth(c) else b
        return c

    def _or(self):
        v = self._and()
        while self._accept("||"):
            r = self._and()
            v = 1.0 if (_truth(v) or _truth(r)) else 0.0
        return v

    def _and(self):
        v = self._cmp()
        while self._accept("&&"):
            r = self._cmp()
            v = 1.0 if (_truth(v) and _truth(r)) else 0.0
        return v

    def _cmp(self):
        v = self._add()
        while True:
            t = self._peek()
            if t[0] == "op" and t[1] in ("==", "!=", "<", ">", "<=", ">=", "="):
                self._i += 1
                r = self._add()
                op = t[1]
                if isinstance(v, str) or isinstance(r, str):
                    a, b = str(v), str(r)
                else:
                    a, b = _num(v), _num(r)
                if op in ("==", "="):
                    v = 1.0 if (a == b if isinstance(a, str) else abs(a - b) < 1e-9) else 0.0
                elif op == "!=":
                    v = 1.0 if (a != b if isinstance(a, str) else abs(a - b) >= 1e-9) else 0.0
                elif op == "<":
                    v = 1.0 if a < b else 0.0
                elif op == ">":
                    v = 1.0 if a > b else 0.0
                elif op == "<=":
                    v = 1.0 if a <= b else 0.0
                else:
                    v = 1.0 if a >= b else 0.0
            else:
                return v

    def _add(self):
        v = self._mul()
        while True:
            if self._accept("+"):
                r = self._mul()
                if isinstance(v, str) or isinstance(r, str):
                    v = f"{v}{r}"
                elif isinstance(v, np.ndarray) or isinstance(r, np.ndarray):
                    v = _vadd(v, r, 1)
                else:
                    v = _num(v) + _num(r)
            elif self._accept("-"):
                r = self._mul()
                if isinstance(v, np.ndarray) or isinstance(r, np.ndarray):
                    v = _vadd(v, r, -1)
                else:
                    v = _num(v) - _num(r)
            else:
                return v

    def _mul(self):
        v = self._unary()
        while True:
            if self._accept("*"):
                r = self._unary()
                v = _mul(v, r)
            elif self._accept("/"):
                r = self._unary()
                d = _num(r)
                if isinstance(v, np.ndarray):
                    v = (v / d) if d else v
                else:
                    v = _num(v) / d if d else 0.0
            elif self._accept("%"):
                r = self._unary()
                d = _num(r)
                v = math.fmod(_num(v), d) if d else 0.0
            else:
                return v

    def _unary(self):
        if self._accept("-"):
            v = self._unary()
            return -v if isinstance(v, np.ndarray) else -_num(v)
        if self._accept("+"):
            return self._unary()
        if self._accept("!"):
            return 0.0 if _truth(self._unary()) else 1.0
        return self._power()

    def _power(self):
        v = self._postfix()
        if self._accept("^"):
            r = self._unary()
            return _num(v) ** _num(r)
        return v

    def _postfix(self):
        v = self._primary()
        while True:
            t = self._peek()
            if t[0] == "unit":
                self._i += 1
                u = t[1][1:-1].strip().lower()
                k = _UNITS.get(u, 1.0)
                if not isinstance(v, (str, np.ndarray)):
                    v = _num(v) * k
                continue
            if t[0] == "op" and t[1] == ".":
                self._i += 1
                nt = self._next()
                if nt[0] != "id":
                    raise ExprError("Ожидалось имя после «.»")
                name = nt[1]
                if self._accept("("):
                    args = self._args()
                    f = _FUNCS.get(name.lower())
                    if f is None:
                        raise ExprError(f"Неизвестная функция {name}")
                    r = f(*args)
                    if is_mat(v) and is_mat(r):
                        v = v @ r
                    elif is_mat(r):
                        v = r
                    else:
                        v = r
                else:
                    low = name.lower()
                    if is_vec(v) and low in ("x", "y", "z", "w"):
                        v = float(v["xyzw".index(low)])
                    elif is_mat(v) and low in ("n", "o", "a", "p"):
                        v = vec(*v[:3, "noap".index(low)], 1.0)
                    elif is_vec(v) and low in ("length", "len"):
                        v = float(np.linalg.norm(v[:3]))
                    else:
                        raise ExprError(f"Неизвестный член .{name}")
                continue
            if t[0] == "op" and t[1] == "[":
                self._i += 1
                idx = int(_num(self._ternary()))
                self._expect("]")
                if isinstance(v, np.ndarray):
                    v = float(v.flat[idx])
                continue
            return v

    def _args(self):
        args = []
        if self._accept(")"):
            return args
        while True:
            args.append(self._ternary())
            if self._accept(")"):
                return args
            self._expect(",")

    def _primary(self):
        t = self._next()
        k, s = t
        if k == "num":
            return float(s)
        if k == "str":
            return bytes(s[1:-1], "utf-8").decode("unicode_escape") if "\\" in s else s[1:-1]
        if k == "op" and s == "(":
            v = self._ternary()
            self._expect(")")
            return v
        if k == "id":
            if self._accept("("):
                args = self._args()
                f = _FUNCS.get(s.lower())
                if f is None:
                    raise ExprError(f"Неизвестная функция {s}")
                return f(*args)
            val = self.resolver(s)
            if val is None:
                low = s.lower()
                if low in _CONSTS:
                    return _CONSTS[low]
                if s not in self._unknown:
                    self._unknown.add(s)
                    self.warn(f"Неизвестная переменная «{s}» — принято значение 0")
                return 0.0
            return val
        raise ExprError(f"Неожиданный элемент «{s}»")


def _vadd(a, b, sign):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
        r = a + sign * b
        if a.shape == (4,):
            r[3] = 1.0
        return r
    return a if isinstance(a, np.ndarray) else b


def _mul(a, b):
    if is_mat(a) and is_mat(b):
        return a @ b
    if is_mat(a) and is_vec(b):
        r = a @ np.array(b, float)
        return r.view(Vec)
    if isinstance(a, np.ndarray) and not isinstance(b, np.ndarray):
        r = a * _num(b)
        if a.shape == (4,):
            r[3] = 1.0
        return r
    if isinstance(b, np.ndarray) and not isinstance(a, np.ndarray):
        r = b * _num(a)
        if b.shape == (4,):
            r[3] = 1.0
        return r
    if is_vec(a) and is_vec(b):
        return float(np.dot(a[:3], b[:3]))
    return _num(a) * _num(b)
