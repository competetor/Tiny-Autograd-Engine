from __future__ import annotations
import math
from autograd import Value
from autograd.higher import hvp_fd

def test_hvp_quadratic_exact():
    # f(x) = 0.5 * x^T A x with A = [[3,1],[1,2]]  => H = A
    def f(x0: Value, x1: Value) -> Value:
        return 0.5 * (3*x0*x0 + 2*x1*x1 + 2*x0*x1)  # symmetric cross term
    x = [1.0, -2.0]
    v = [0.3, 0.4]
    hv = hvp_fd(f, x, v, eps=1e-6)  # should be A @ v
    A = [[3.0, 1.0],[1.0, 2.0]]
    hv_true = [A[0][0]*v[0] + A[0][1]*v[1], A[1][0]*v[0] + A[1][1]*v[1]]
    assert abs(hv[0] - hv_true[0]) < 1e-6
    assert abs(hv[1] - hv_true[1]) < 1e-6

def test_hvp_fd_reasonable_on_nonlinear():
    # Nonlinear scalar: f = sin(x0*x1) + exp(x0 - x1)
    def f(x0: Value, x1: Value) -> Value:
        return (x0 * x1).sin() + (x0 - x1).exp()
    x = [0.7, -0.9]
    v = [0.2, -0.5]
    hv = hvp_fd(f, x, v, eps=1e-5)

    # coarse self-consistency check: ||H v|| not absurdly large and finite
    norm = (hv[0]**2 + hv[1]**2) ** 0.5
    assert math.isfinite(norm)
    assert norm < 1e6
