# tests/test_jvp_dual.py
from __future__ import annotations
import math
from autograd.dual import jvp

def test_jvp_matches_directional_fd():
    # scalar function using Dual ops
    def f(x, y):
        return ( (x * y + x**2).tanh() + (x.sigmoid() * y.cos()) - (x - y).exp().log() )
        # last term simplifies to (x - y) for domain >0; keep domain positive below

    x = [1.2, 0.5]  # ensure x - y > 0
    v = [0.7, -1.1]
    y, jv = jvp(f, x, v)

    # numerical directional derivative (central difference)
    eps = 1e-6
    def f_num(xx, yy):
        return math.tanh(xx*yy + xx**2) + (1/(1+math.exp(-xx)))*math.cos(yy) - (xx - yy)
    y_plus  = f_num(x[0] + eps*v[0], x[1] + eps*v[1])
    y_minus = f_num(x[0] - eps*v[0], x[1] - eps*v[1])
    jv_fd = (y_plus - y_minus) / (2.0 * eps)

    assert abs(jv - jv_fd) < 1e-6
