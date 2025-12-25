# tests/test_finite_difference.py
from __future__ import annotations

import math
import random

from autograd.value import Value


def finite_diff(f, x: Value, eps: float = 1e-6) -> float:
    """Central difference ∂f/∂x ≈ (f(x+ε) - f(x-ε)) / (2ε)."""
    x.data += eps
    f1 = f().data
    x.data -= 2 * eps
    f2 = f().data
    x.data += eps  # restore
    return (f1 - f2) / (2 * eps)


def test_random_scalar_expression_gradients_agree_with_fd():
    random.seed(0)

    # base parameters
    a = Value(0.5)
    b = Value(-1.25)
    c = Value(2.0)

    # construct a safe/randomish scalar expression using supported ops
    # keep domains safe for log by shifting with +1 via relu
    def make_expr() -> Value:
        x = (a * b + c).tanh()
        y = (a ** 2) * 0.3 + (b ** 3) * -0.1
        z = (a - b).relu() + (c - a).relu()
        # safe positive input for log: |(a+b)| + 1
        safe_pos = ( (a + b).relu() + (-(a + b)).relu() ) + 1.0
        w = safe_pos.log() + (a * 0.2).exp()
        return (x + y + z + w)

    # analytic grads
    out = make_expr()
    a.grad = b.grad = c.grad = 0.0
    out.backward()

    # finite-diff grads
    ga = finite_diff(make_expr, a)
    gb = finite_diff(make_expr, b)
    gc = finite_diff(make_expr, c)

    assert abs(a.grad - ga) < 1e-4
    assert abs(b.grad - gb) < 1e-4
    assert abs(c.grad - gc) < 1e-4
