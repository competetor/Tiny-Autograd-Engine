# autograd/value.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Tuple, List, Set, Optional, Dict, Any
import time

from . import ops  # NEW: use the op registry for forward


class Value:
    """
    Scalar reverse-mode autodiff node.

    Features:
    - Deterministic graph edges: _prev is a tuple (not a set).
    - Optional graph freeing after backward (retain_graph=False by default).
    - Global grad switch (no_grad) + per-node requires_grad.
    - All ops accumulate grads (never overwrite).
    - Gradient hooks (pre/post) per node.
    - Mini profiler (timings, node/edge counts, op histogram).
    - NEW: In-place mutation guard on non-leaf nodes.
    - NEW: stop_gradient() to detach-and-leafify any node.
    """

    __slots__ = (
        "data",
        "grad",
        "_backward",
        "_prev",
        "_op",
        "requires_grad",
        "_pre_hooks",
        "_post_hooks",
        "_viz_prev",
    )

    _grad_enabled: bool = True  # global switch for grad tracking on new nodes

    def __init__(
        self,
        data: float,
        _children: Tuple["Value", ...] | List["Value"] | None = None,
        _op: str = "",
        requires_grad: bool = True,
    ):
        # NOTE: data is set before _prev exists; guard in __setattr__ allows this.
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Tuple["Value", ...] = tuple(_children) if _children else ()
        self._op = _op
        self.requires_grad = bool(requires_grad)
        self._viz_prev = ()
        self._pre_hooks: List[Callable[[Value], None]] = []
        self._post_hooks: List[Callable[[Value], None]] = []

    # -------------------- safety: in-place guards --------------------

    def __setattr__(self, name, value):
        # block mutating .data on non-leaf nodes (nodes with parents)
        if name == "data" and hasattr(self, "_prev"):
            # allow during __init__ (before _prev exists) and leaf updates (optimizer)
            if self._prev:
                raise RuntimeError(
                    "Cannot assign to .data of a non-leaf Value (result of an operation). "
                    "Call stop_gradient() to cut the graph, or mutate leaf parameters only."
                )
        object.__setattr__(self, name, value)

    def stop_gradient(self) -> "Value":
        """
        Cut the graph at this node: make it a leaf that does not require grad.
        After this, it's safe to mutate .data on this node.
        """
        self._prev = ()
        self._backward = lambda: None
        self.requires_grad = False
        return self

    # -------------------- hooks API --------------------

    def register_hook(
        self, fn: Callable[["Value"], None], when: str = "pre"
    ) -> Callable[["Value"], None]:
        """
        Register a hook on this node. The hook receives the node (self) and can
        read/modify `self.grad`.

        `when`:
          - "pre"  : called right BEFORE this node propagates grad to parents
          - "post" : called right AFTER this node propagates grad to parents
        """
        if when not in ("pre", "post"):
            raise ValueError("when must be 'pre' or 'post'")
        (self._pre_hooks if when == "pre" else self._post_hooks).append(fn)
        return fn

    def clear_hooks(self) -> None:
        self._pre_hooks.clear()
        self._post_hooks.clear()

    # -------------------- utils --------------------

    def __repr__(self) -> str:
        label = self._op if self._op else "leaf"
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op={label})"

    def detach(self) -> "Value":
        """Return a brand-new leaf copy that does not track gradients."""
        return Value(self.data, requires_grad=False)

    def zero_grad(self) -> None:
        """Zero out grads on this node and (if the graph still exists) its parents."""
        self.grad = 0.0
        for p in self._prev:
            p.zero_grad()

    # -------------------- autodiff core --------------------

    def backward(self, retain_graph: bool = False, profile: bool = False) -> None:
        """
        Backprop from this scalar output.
        - Seeds self.grad = 1.0.
        - Topologically sorts reachable graph once.
        - Frees graph refs unless retain_graph=True.
        - If profile=True, prints a tiny timing/size/op summary.

        Hooks:
        - Pre-hooks are called on each node *before* propagating its grad to parents.
        - Post-hooks are called *after* the node has propagated grad to parents.
        """
        t0 = time.perf_counter()

        topo: List[Value] = []
        visited: Set[Value] = set()

        def build(v: "Value") -> None:
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)

        build(self)
        t_build = time.perf_counter()

        # seed
        self.grad = 1.0

        # backprop
        edges_count = sum(len(v._prev) for v in topo)
        pre_calls = 0
        post_calls = 0

        for v in reversed(topo):
            if v._pre_hooks:
                for h in v._pre_hooks:
                    h(v)
                pre_calls += len(v._pre_hooks)

            v._backward()

            if v._post_hooks:
                for h in v._post_hooks:
                    h(v)
                post_calls += len(v._post_hooks)

            if not retain_graph:
                if getattr(v, "_prev", ()):
                    v._viz_prev = tuple(v._prev)
                v._prev = ()
                v._backward = lambda: None

        t_bw = time.perf_counter()

        if profile:
            ops_hist: Dict[str, int] = {}
            for v in topo:
                key = ops.get_pretty_name(v._op)
                ops_hist[key] = ops_hist.get(key, 0) + 1

            build_ms = (t_build - t0) * 1000.0
            bw_ms = (t_bw - t_build) * 1000.0
            total_ms = (t_bw - t0) * 1000.0
            top_ops = sorted(ops_hist.items(), key=lambda kv: kv[1], reverse=True)
            top_ops_str = ", ".join(f"{k}:{v}" for k, v in top_ops[:6])

            print(
                f"[autograd] nodes={len(topo)}, edges={edges_count}, "
                f"build={build_ms:.2f}ms, backward={bw_ms:.2f}ms, total={total_ms:.2f}ms, "
                f"hooks(pre={pre_calls}, post={post_calls}), ops=[{top_ops_str}]"
            )

    # -------------------- helpers --------------------

    @staticmethod
    def _wrap(other: float | "Value") -> "Value":
        """Auto-wrap raw numbers as non-grad leaves."""
        return other if isinstance(other, Value) else Value(other, requires_grad=False)

    @staticmethod
    def _propagate_requires_grad(*parents: "Value") -> bool:
        """Node requires grad only if any parent requires grad AND grad is globally enabled."""
        return Value._grad_enabled and any(p.requires_grad for p in parents)

    @staticmethod
    def _from_op(
        op: str,
        parents: Tuple["Value", ...],
        need_grad: bool,
    ) -> "Value":
        """
        Centralized creation of a new Value via the op registry.

        - Uses ops.apply_forward(op, parents_data) for the scalar output.
        - Wires parents only if need_grad is True.
        """
        data = ops.apply_forward(op, tuple(p.data for p in parents))
        return Value(data, parents if need_grad else (), op, need_grad)

    # -------------------- binary ops --------------------

    def __add__(self, other: float | "Value") -> "Value":
        other = Value._wrap(other)
        need_grad = self._propagate_requires_grad(self, other)
        out = Value._from_op("+", (self, other), need_grad)

        def _backward():
            if not need_grad:
                return
            if self.requires_grad:
                self.grad += 1.0 * out.grad
            if other.requires_grad:
                other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self) -> "Value":
        need_grad = self._propagate_requires_grad(self)
        out = Value._from_op("neg", (self,), need_grad)

        def _backward():
            if not need_grad:
                return
            if self.requires_grad:
                self.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: float | "Value") -> "Value":
        return self + (-Value._wrap(other))

    def __rsub__(self, other: float | "Value") -> "Value":
        return Value._wrap(other) + (-self)

    def __mul__(self, other: float | "Value") -> "Value":
        other = Value._wrap(other)
        need_grad = self._propagate_requires_grad(self, other)
        out = Value._from_op("*", (self, other), need_grad)

        def _backward():
            if not need_grad:
                return
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other: float | "Value") -> "Value":
        other = Value._wrap(other)
        if other.data == 0.0:
            raise ZeroDivisionError(
                "division by zero in Value.__truediv__ (denominator.data == 0)"
            )
        return self * (other ** -1)

    def __rtruediv__(self, other: float | "Value") -> "Value":
        other = Value._wrap(other)
        if self.data == 0.0:
            raise ZeroDivisionError(
                "division by zero in Value.__rtruediv__ (self.data == 0)"
            )
        return other * (self ** -1)

    def __pow__(self, power: float) -> "Value":
        if not isinstance(power, (int, float)):
            raise TypeError("power must be a number")
        need_grad = self._propagate_requires_grad(self)
        op = f"**{float(power)}"
        out = Value._from_op(op, (self,), need_grad)

        def _backward():
            if not need_grad:
                return
            if self.requires_grad:
                self.grad += (power * (self.data ** (power - 1))) * out.grad

        out._backward = _backward
        return out

    # -------------------- unary funcs --------------------

    def relu(self) -> "Value":
        need_grad = self._propagate_requires_grad(self)
        out = Value._from_op("relu", (self,), need_grad)

        def _backward():
            if not need_grad:
                return
            if self.requires_grad:
                # out.data is already max(self.data, 0), but we check self.data for clarity
                self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        need_grad = self._propagate_requires_grad(self)
        out = Value._from_op("tanh", (self,), need_grad)
        t = out.data  # tanh(self.data) from ops

        def _backward():
            if not need_grad:
                return
            if self.requires_grad:
                self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        need_grad = self._propagate_requires_grad(self)
        out = Value._from_op("sigmoid", (self,), need_grad)
        s = out.data  # sigmoid(self.data) from ops

        def _backward():
            if not need_grad:
                return
            if self.requires_grad:
                self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        need_grad = self._propagate_requires_grad(self)
        out = Value._from_op("exp", (self,), need_grad)
        e = out.data  # exp(self.data) from ops

        def _backward():
            if not need_grad:
                return
            if self.requires_grad:
                self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        if self.data <= 0.0:
            raise ValueError(
                f"log domain error: input must be > 0, got {self.data}"
            )
        need_grad = self._propagate_requires_grad(self)
        out = Value._from_op("log", (self,), need_grad)

        def _backward():
            if not need_grad:
                return
            if self.requires_grad:
                self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sin(self) -> "Value":
        need_grad = self._propagate_requires_grad(self)
        out = Value._from_op("sin", (self,), need_grad)

        def _backward():
            if not need_grad:
                return
            if self.requires_grad:
                # derivative of sin is cos, evaluated at the input
                self.grad += ops.apply_forward("cos", (self.data,)) * out.grad

        out._backward = _backward
        return out

    def cos(self) -> "Value":
        need_grad = self._propagate_requires_grad(self)
        out = Value._from_op("cos", (self,), need_grad)

        def _backward():
            if not need_grad:
                return
            if self.requires_grad:
                # derivative of cos is -sin, evaluated at the input
                sin_x = ops.apply_forward("sin", (self.data,))
                self.grad += -sin_x * out.grad

        out._backward = _backward
        return out


@contextmanager
def no_grad():
    """
    Temporarily disable gradient tracking for newly created nodes.
    Existing nodes keep their current requires_grad values.
    """
    old = Value._grad_enabled
    Value._grad_enabled = False
    try:
        yield
    finally:
        Value._grad_enabled = old
