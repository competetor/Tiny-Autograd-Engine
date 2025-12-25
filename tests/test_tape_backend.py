from __future__ import annotations
import math
from autograd.value import Value
from autograd.tape import tape_backward


def make_safe_expr():
    # Build a moderately rich scalar expression with safe log input
    a = Value(0.7)
    b = Value(-1.3)
    c = Value(2.2)

    # |a+b| + 1 to keep log domain > 0
    abs_ab = (a + b).relu() + (-(a + b)).relu()
    safe = abs_ab + 1.0

    out = ((a * b + c).tanh()
           + (a ** 2) * 0.3
           + (b ** 3) * -0.1
           + safe.log()
           + (a * 0.2).exp()
           + (b * 0.5).sin()
           + (c * -0.3).cos())
    return a, b, c, out


def test_tape_backward_matches_value_backward():
    a, b, c, out = make_safe_expr()

    # grads via regular engine
    a.grad = b.grad = c.grad = 0.0
    out.backward(retain_graph=True)  # retain for second pass
    g_a, g_b, g_c = a.grad, b.grad, c.grad

    # reset grads and run tape
    a.grad = b.grad = c.grad = 0.0
    tape_backward(out, retain_graph=False)
    g2_a, g2_b, g2_c = a.grad, b.grad, c.grad

    assert abs(g_a - g2_a) < 1e-10
    assert abs(g_b - g2_b) < 1e-10
    assert abs(g_c - g2_c) < 1e-10

    # graph should be freed now
    assert out._prev == ()


def test_tape_on_simple_pow_and_relu():
    x = Value(1.5)
    y = ((x ** 3) * 2.0 + x.relu())
    tape_backward(y)
    # dy/dx = 2*3*x^2 + 1 = 6*x^2 + 1
    expected = 6 * (1.5 ** 2) + 1.0
    assert abs(x.grad - expected) < 1e-12
