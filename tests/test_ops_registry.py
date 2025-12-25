# tests/test_ops_registry.py
from __future__ import annotations
from autograd.value import Value
from autograd.tape import tape_backward, Tape

def make_expr():
    a = Value(0.7); b = Value(-1.3); c = Value(2.2)
    # includes many ops supported by the registry
    out = ((a * b + c).tanh()
           + (a ** 2) * 0.3
           + (b ** 3) * -0.1
           + (a * 0.2).exp()
           + (b * 0.5).sin()
           + (c * -0.3).cos())
    return a, b, c, out

def test_registry_tape_matches_classic():
    a, b, c, out = make_expr()

    # classic grads
    a.grad = b.grad = c.grad = 0.0
    out.backward(retain_graph=True)
    g = (a.grad, b.grad, c.grad)

    # registry-powered tape grads
    a.grad = b.grad = c.grad = 0.0
    assert Tape.supports_graph(out)
    tape_backward(out)
    g2 = (a.grad, b.grad, c.grad)

    for x, y in zip(g, g2):
        assert abs(x - y) < 1e-12
