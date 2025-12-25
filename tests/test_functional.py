# tests/test_functional.py
from __future__ import annotations
from autograd.functional import grad, value_and_grad, jacobian

def test_grad_and_value_and_grad():
    f = lambda x, y: (x * x + 3.0 * x * y + y ** 2)
    g = grad(f)
    vals = g(2.0, -1.0)  # ∂f/∂x = 2x + 3y, ∂f/∂y = 3x + 2y
    assert abs(vals[0] - (2*2.0 + 3*(-1.0))) < 1e-12
    assert abs(vals[1] - (3*2.0 + 2*(-1.0))) < 1e-12

    vg = value_and_grad(f)
    v, grads = vg(1.0, 2.0)
    assert abs(v - (1.0*1.0 + 3*1.0*2.0 + 4.0)) < 1e-12
    assert len(grads) == 2

def test_jacobian_vector_output():
    def F(x, y):
        return [x*y, x + y, (x - y)]
    J = jacobian(F)(2.0, 3.0)
    # rows: ∂f_i/∂[x,y]
    assert J[0] == [3.0, 2.0]
    assert J[1] == [1.0, 1.0]
    assert J[2] == [1.0, -1.0]
