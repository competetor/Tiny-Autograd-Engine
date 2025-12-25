from __future__ import annotations
import re

from autograd.value import Value


def test_pre_hook_scales_upstream_grad_before_parent_update():
    # y = a * b  =>  dy/da = b, dy/db = a
    a, b = Value(2.0), Value(3.0)
    y = a * b

    # double upstream grad at y BEFORE it propagates to a,b
    def scale2(v: Value):
        v.grad *= 2.0

    y.register_hook(scale2, when="pre")

    a.grad = b.grad = 0.0
    y.backward()
    assert abs(a.grad - 2.0 * b.data) < 1e-12
    assert abs(b.grad - 2.0 * a.data) < 1e-12


def test_post_hook_runs_after_parent_update_and_cant_affect_parents():
    # z = a * b + a  => dz/da = b + 1, dz/db = a
    a, b = Value(1.5), Value(-4.0)
    z = a * b + a

    calls = {"post": 0}

    def post(v: Value):
        calls["post"] += 1
        # zeroing v.grad here must NOT change grads already written to parents
        v.grad = 0.0

    z.register_hook(post, when="post")
    a.grad = b.grad = 0.0
    z.backward()

    assert calls["post"] >= 1
    assert abs(a.grad - (b.data + 1.0)) < 1e-12
    assert abs(b.grad - a.data) < 1e-12


def test_backward_profile_prints_summary(capsys):
    x, y = Value(2.0), Value(3.0)
    out = (x * y + x).tanh()
    out.backward(profile=True)
    s = capsys.readouterr().out
    assert "[autograd]" in s
    assert re.search(r"nodes=\d+", s) is not None
    assert "backward=" in s and "total=" in s and "ops=[" in s
