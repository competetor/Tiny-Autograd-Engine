# tests/test_graphviz.py
from __future__ import annotations
import pytest
from autograd.value import Value
from autograd.utils import draw_graph

graphviz = pytest.importorskip("graphviz")  # skip if the python package isn't installed

def test_draw_graph_builds_dot_without_rendering():
    # small graph
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    out = (a * b + c).tanh() + a
    out.backward()  # populate grads for labels

    # do not render (no system 'dot' binary required)
    dot = draw_graph(out, render=False)
    src = dot.source

    # sanity: nodes & edges exist, some ops appear in labels
    print(dot)
    assert "->" in src
    assert "tanh" in src
    # depending on label formatting, '*' or 'op' may appear—just ensure we labeled something
    assert "Value" in src or "*" in src or "relu" in src or "sigmoid" in src
