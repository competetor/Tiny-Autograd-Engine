# autograd/higher.py
from __future__ import annotations
from typing import Callable, Iterable, List
from .value import Value

def _as_values(xs: Iterable[float]) -> List[Value]:
    return [Value(float(x), requires_grad=True) for x in xs]

def grad_of(f: Callable[..., Value], xs: List[float]) -> List[float]:
    vals = _as_values(xs)
    out = f(*vals)
    if not isinstance(out, Value):
        raise TypeError("f must return a Value")
    for v in vals:
        v.grad = 0.0
    out.backward()
    return [v.grad for v in vals]

def hvp_fd(
    f: Callable[..., Value],
    x: Iterable[float],
    v: Iterable[float],
    eps: float = 1e-5,
) -> List[float]:
    """
    Hessian-vector product H(x) @ v via central finite differences of the gradient:
        H v ≈ (∇f(x+εv) - ∇f(x-εv)) / (2ε)
    Robust and works for any function expressed with Value ops.
    """
    x = list(x)
    v = list(v)
    if len(x) != len(v):
        raise ValueError("x and v must have same length")
    x_plus  = [xi + eps * vi for xi, vi in zip(x, v)]
    x_minus = [xi - eps * vi for xi, vi in zip(x, v)]
    g_plus  = grad_of(f, x_plus)
    g_minus = grad_of(f, x_minus)
    inv = 1.0 / (2.0 * eps)
    return [(gp - gm) * inv for gp, gm in zip(g_plus, g_minus)]
