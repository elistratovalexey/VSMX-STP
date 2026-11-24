import math

from vcmx2stp.expr import Evaluator
from vcmx2stp.rsc_parser import parse, parse_bytes, walk
from vcmx2stp.updater import is_newer, parse_version


def test_parse_tree():
    els = parse('Node "rSimResource"\n{\nName "A b"\nLocation 1 0 0\nFeature "rGeoFeature"\n{\nUri "x#y"\n}\n}\n')
    assert len(els) == 1
    n = els[0]
    assert n.key == "Node" and n.arg == "rSimResource"
    assert n.name == "A b"
    assert n.values("Location") == ["1", "0", "0"]
    assert [e.value("Uri") for e in walk(els) if e.key == "Feature"] == ["x#y"]


def test_parse_line_continuation_and_header():
    els = parse_bytes(b'VCMD0028040100000000COMPONENT\nUri "file:///C:/a\\\nb.vcmx#p"\n')
    assert els[0].arg == "file:///C:/a" + "b.vcmx#p"


def test_expression_basic():
    ev = Evaluator(lambda name: {"L": 100.0}.get(name))
    assert ev.eval("2*L + 10") == 210
    assert abs(ev.eval("sin(90)") - 1.0) < 1e-9 or abs(ev.eval("sin(pi/2)") - 1.0) < 1e-9
    assert ev.eval("100.00 {mm}") == 100
    assert math.isclose(ev.eval("L > 50 ? 1 : 2"), 1)


def test_versions():
    assert parse_version("v1.2.3") == (1, 2, 3)
    assert is_newer("1.2.4", "1.2.3")
    assert not is_newer("1.2.3", "1.2.3")
    assert is_newer("2.0.0", "1.9.9.120")
