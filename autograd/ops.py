# autograd/ops.py
from __future__ import annotations
from typing import Tuple, List, Callable, Dict, Optional
import math

ForwardFn = Callable[[Tuple[float, ...]], float]
GradFn = Callable[[int, Tuple[int, ...], List[float], List[float], List[bool]], None]

class Op:
    """
    Small operation object holding both forward and backward rules.

    In a "real" engine this would also carry:
      - dtype / device dispatch
      - categories (elementwise, reduction, etc.)
      - maybe multiple kernels
    Here we keep it tiny but structured.
    """

    __slots__ = ("name", "forward_fn", "grad_fn", "arity", "pretty_name")

    def __init__(
        self,
        name: str,
        *,
        forward_fn: Optional[ForwardFn] = None,
        grad_fn: Optional[GradFn] = None,
        arity: Optional[int] = None,
        pretty_name: Optional[str] = None,
    ) -> None:
        self.name = name
        self.forward_fn = forward_fn
        self.grad_fn = grad_fn
        self.arity = arity
        self.pretty_name = pretty_name or name

    # These helpers keep tape / Value code simple & uniform.
    def apply_forward(self, parents_data: Tuple[float, ...]) -> float:
        if self.forward_fn is None:
            raise NotImplementedError(
                f"Op {self.name!r} has no forward_fn registered"
            )
        # optional: cheap arity check in debug mode
        if self.arity is not None and len(parents_data) != self.arity:
            raise ValueError(
                f"Op {self.name!r} expected {self.arity} parents, "
                f"got {len(parents_data)}"
            )
        return self.forward_fn(parents_data)

    def apply_grad(
        self,
        i: int,
        parents: Tuple[int, ...],
        data: List[float],
        grads: List[float],
        requires: List[bool],
    ) -> None:
        if self.grad_fn is None:
            # Some ops might be forward-only, but for now we treat this as an error.
            raise NotImplementedError(
                f"Op {self.name!r} has no grad_fn registered"
            )
        self.grad_fn(i, parents, data, grads, requires)

# Global registry: string op-code -> Op object.
_OPS: Dict[str, Op] = {}

def _register(
    name: str,
    *,
    forward_fn: Optional[ForwardFn] = None,
    grad_fn: Optional[GradFn] = None,
    arity: Optional[int] = None,
    pretty_name: Optional[str] = None,
) -> None:
    if name in _OPS:
        raise ValueError(f"duplicate op registration for {name!r}")
    _OPS[name] = Op(
        name,
        forward_fn=forward_fn,
        grad_fn=grad_fn,
        arity=arity,
        pretty_name=pretty_name,
    )

# ---------------------------------------------------------------------------
# Primitive grad rules
# ---------------------------------------------------------------------------

def _add_forward(xs: Tuple[float, ...]) -> float:
    a, b = xs
    return a + b
    
def _add_grad(
    i: int,
    parents: Tuple[int, ...],
    data: List[float],
    grads: List[float],
    requires: List[bool],
) -> None:
    a, b = parents
    g = grads[i]
    if g == 0.0:
        return
    if requires[a]:
        grads[a] += g
    if requires[b]:
        grads[b] += g

def _mul_forward(xs: Tuple[float, ...]) -> float:
    a, b = xs
    return a * b

def _mul_grad(
    i: int,
    parents: Tuple[int, ...],
    data: List[float],
    grads: List[float],
    requires: List[bool],
) -> None:
    a, b = parents
    g = grads[i]
    if g == 0.0:
        return
    if requires[a]:
        grads[a] += data[b] * g
    if requires[b]:
        grads[b] += data[a] * g

def _neg_forward(xs: Tuple[float, ...]) -> float:
    (a,) = xs
    return -a

def _neg_grad(
    i: int,
    parents: Tuple[int, ...],
    data: List[float],
    grads: List[float],
    requires: List[bool],
) -> None:
    (a,) = parents
    g = grads[i]
    if g == 0.0:
        return
    if requires[a]:
        grads[a] -= g

def _relu_forward(xs: Tuple[float, ...]) -> float:
    (a,) = xs
    return a if a > 0.0 else 0.0

def _relu_grad(
    i: int,
    parents: Tuple[int, ...],
    data: List[float],
    grads: List[float],
    requires: List[bool],
) -> None:
    (a,) = parents
    g = grads[i]
    if g == 0.0:
        return
    if requires[a]:
        grads[a] += (1.0 if data[i] > 0.0 else 0.0) * g

def _tanh_forward(xs: Tuple[float, ...]) -> float:
    (a,) = xs
    return math.tanh(a)

def _tanh_grad(
    i: int,
    parents: Tuple[int, ...],
    data: List[float],
    grads: List[float],
    requires: List[bool],
) -> None:
    (a,) = parents
    g = grads[i]
    if g == 0.0:
        return
    if requires[a]:
        t = data[i]  # tanh output
        grads[a] += (1.0 - t * t) * g

def _sigmoid_forward(xs: Tuple[float, ...]) -> float:
    (a,) = xs
    # standard numerically-stable-ish logistic
    if a >= 0:
        e = math.exp(-a)
        return 1.0 / (1.0 + e)
    else:
        e = math.exp(a)
        return e / (1.0 + e)

def _sigmoid_grad(
    i: int,
    parents: Tuple[int, ...],
    data: List[float],
    grads: List[float],
    requires: List[bool],
) -> None:
    (a,) = parents
    g = grads[i]
    if g == 0.0:
        return
    if requires[a]:
        s = data[i]
        grads[a] += (s * (1.0 - s)) * g

def _exp_forward(xs: Tuple[float, ...]) -> float:
    (a,) = xs
    return math.exp(a)

def _exp_grad(
    i: int,
    parents: Tuple[int, ...],
    data: List[float],
    grads: List[float],
    requires: List[bool],
) -> None:
    (a,) = parents
    g = grads[i]
    if g == 0.0:
        return
    if requires[a]:
        grads[a] += data[i] * g  # out == exp(a)

def _log_forward(xs: Tuple[float, ...]) -> float:
    (a,) = xs
    return math.log(a)

def _log_grad(
    i: int,
    parents: Tuple[int, ...],
    data: List[float],
    grads: List[float],
    requires: List[bool],
) -> None:
    (a,) = parents
    g = grads[i]
    if g == 0.0:
        return
    if requires[a]:
        grads[a] += (1.0 / data[a]) * g

def _sin_forward(xs: Tuple[float, ...]) -> float:
    (a,) = xs
    return math.sin(a)

def _sin_grad(
    i: int,
    parents: Tuple[int, ...],
    data: List[float],
    grads: List[float],
    requires: List[bool],
) -> None:
    (a,) = parents
    g = grads[i]
    if g == 0.0:
        return
    if requires[a]:
        grads[a] += math.cos(data[a]) * g

def _cos_forward(xs: Tuple[float, ...]) -> float:
    (a,) = xs
    return math.cos(a)

def _cos_grad(
    i: int,
    parents: Tuple[int, ...],
    data: List[float],
    grads: List[float],
    requires: List[bool],
) -> None:
    (a,) = parents
    g = grads[i]
    if g == 0.0:
        return
    if requires[a]:
        grads[a] += -math.sin(data[a]) * g

def _pow_forward(power: float, xs: Tuple[float, ...]) -> float:
    (a,) = xs
    return a ** power

def _pow_grad(
    power: float,
    i: int,
    parents: Tuple[int, ...],
    data: List[float],
    grads: List[float],
    requires: List[bool],
) -> None:
    (a,) = parents
    g = grads[i]
    if g == 0.0:
        return
    if requires[a]:
        grads[a] += (power * (data[a] ** (power - 1.0))) * g
        
# Register primitive ops (fixed names).
_register(
    "+",
    forward_fn=_add_forward,
    grad_fn=_add_grad,
    arity=2,
    pretty_name="add",
)
_register(
    "*",
    forward_fn=_mul_forward,
    grad_fn=_mul_grad,
    arity=2,
    pretty_name="mul",
)
_register(
    "neg",
    forward_fn=_neg_forward,
    grad_fn=_neg_grad,
    arity=1,
    pretty_name="neg",
)
_register(
    "relu",
    forward_fn=_relu_forward,
    grad_fn=_relu_grad,
    arity=1,
    pretty_name="ReLU",
)
_register(
    "tanh",
    forward_fn=_tanh_forward,
    grad_fn=_tanh_grad,
    arity=1,
    pretty_name="tanh",
)
_register(
    "sigmoid",
    forward_fn=_sigmoid_forward,
    grad_fn=_sigmoid_grad,
    arity=1,
    pretty_name="sigmoid",
)
_register(
    "exp",
    forward_fn=_exp_forward,
    grad_fn=_exp_grad,
    arity=1,
    pretty_name="exp",
)
_register(
    "log",
    forward_fn=_log_forward,
    grad_fn=_log_grad,
    arity=1,
    pretty_name="log",
)
_register(
    "sin",
    forward_fn=_sin_forward,
    grad_fn=_sin_grad,
    arity=1,
    pretty_name="sin",
)
_register(
    "cos",
    forward_fn=_cos_forward,
    grad_fn=_cos_grad,
    arity=1,
    pretty_name="cos",
)

def register_custom_op(
    name: str,
    forward_fn: ForwardFn,
    grad_fn: GradFn,
    *,
    arity: Optional[int] = None,
    pretty_name: Optional[str] = None,
) -> None:
    """
    Public API for registering a new scalar op in the global registry.

    Examples
    --------
    >>> def cube_forward(xs):
    ...     (x,) = xs
    ...     return x ** 3
    ...
    >>> def cube_grad(i, parents, data, grads, requires):
    ...     (a,) = parents
    ...     g = grads[i]
    ...     if g == 0.0:
    ...         return
    ...     if requires[a]:
    ...         grads[a] += 3.0 * (data[a] ** 2) * g
    ...
    >>> register_custom_op("cube", cube_forward, cube_grad, arity=1, pretty_name="x^3")
    """
    _register(
        name,
        forward_fn=forward_fn,
        grad_fn=grad_fn,
        arity=arity,
        pretty_name=pretty_name,
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_supported(op: str) -> bool:
    """
    Return True if op is understood by the tape backend / registry.

    Notes:
      - "" is treated as leaf / no-op.
      - f"**{p}" (power) is handled specially and always supported
        as long as p parses as float.
    """
    if op == "":
        return True
    if op in _OPS:
        return True
    if op.startswith("**"):
        try:
            float(op[2:])
        except ValueError:
            return False
        return True
    return False

def _get_pow_exponent(op: str) -> float:
    try:
        return float(op[2:])
    except ValueError as exc:
        raise NotImplementedError(
            f"ops: malformed power op {op!r}"
        ) from exc

def _get_op(op: str) -> Op:
    try:
        return _OPS[op]
    except KeyError as exc:
        raise NotImplementedError(
            f"ops: unsupported op {op!r}"
        ) from exc
    
def get_pretty_name(op: str) -> str:
    """
    Human-friendly name for an op code, used in profiling / graphviz.

    - ""         -> "leaf"
    - "**2.0"    -> "pow(2.0)"
    - registered -> op.pretty_name
    - unknown    -> fall back to the raw op string
    """
    if op == "":
        return "leaf"
    if op.startswith("**"):
        p = _get_pow_exponent(op)
        return f"pow({p})"
    try:
        return _get_op(op).pretty_name
    except NotImplementedError:
        return op

def apply_forward(op: str, parents_data: Tuple[float, ...]) -> float:
    """
    Compute forward value for op given parents' scalar data.

    Used by Value.__add__ / __mul__ / relu / etc. so that
    forward and backward share a single source of truth.
    """
    if op == "":
        # Leaf / passthrough – forward is just parents_data[0]
        if len(parents_data) != 1:
            raise ValueError(
                "apply_forward('', ...) expects exactly one parent"
            )
        return parents_data[0]

    if op.startswith("**"):
        pwr = _get_pow_exponent(op)
        return _pow_forward(pwr, parents_data)

    op_obj = _get_op(op)
    return op_obj.apply_forward(parents_data)

def apply_grad(
    op: str,
    i: int,
    parents: Tuple[int, ...],
    data: List[float],
    grads: List[float],
    requires: List[bool],
) -> None:
    """
    Accumulate gradients for parents of node i given op and parents[].

    This mirrors micrograd-style derivatives; used by the tape backend
    and now fully routed through a registry of Op objects.
    """
    g = grads[i]
    if g == 0.0 or op == "":
        return

    if op.startswith("**"):
        pwr = _get_pow_exponent(op)
        _pow_grad(pwr, i, parents, data, grads, requires)
        return

    op_obj = _get_op(op)
    op_obj.apply_grad(i, parents, data, grads, requires)
