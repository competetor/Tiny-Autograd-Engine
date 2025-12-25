# autograd/tape.py
from __future__ import annotations
from typing import List, Tuple, Dict
import time

from .value import Value
from . import ops as _ops


class Tape:
    """
    Compact computation tape built from Value nodes.
    Backward uses centralized derivatives from autograd.ops.
    """

    __slots__ = ("nodes", "index", "op", "parents", "data", "requires")

    def __init__(self):
        self.nodes: List[Value] = []
        self.index: Dict[Value, int] = {}
        self.op: List[str] = []
        self.parents: List[Tuple[int, ...]] = []
        self.data: List[float] = []
        self.requires: List[bool] = []

    @staticmethod
    def _topo(root: Value) -> List[Value]:
        topo: List[Value] = []
        seen = set()

        def build(v: Value):
            if v in seen:
                return
            seen.add(v)
            for p in getattr(v, "_prev", ()) or getattr(v, "_viz_prev", ()) or ():
                build(p)
            topo.append(v)

        build(root)
        return topo

    @classmethod
    def from_root(cls, root: Value) -> "Tape":
        t = cls()
        t.nodes = cls._topo(root)
        t.index = {v: i for i, v in enumerate(t.nodes)}
        t.op = [v._op for v in t.nodes]
        t.parents = [tuple(t.index[p] for p in ((v._prev or getattr(v, "_viz_prev", ()))) ) for v in t.nodes]
        t.data = [v.data for v in t.nodes]
        t.requires = [v.requires_grad for v in t.nodes]
        return t

    @staticmethod
    def _collect_ops(root: Value) -> set[str]:
        ops = set()
        seen = set()

        def walk(v: Value):
            if v in seen:
                return
            seen.add(v)
            ops.add(v._op)
            for p in getattr(v, "_prev", ()) or getattr(v, "_viz_prev", ()) or ():
                walk(p)

        walk(root)
        return ops

    @classmethod
    def supports_graph(cls, root: Value) -> bool:
        return all(_ops.is_supported(op) for op in cls._collect_ops(root))

    def backward(self, retain_graph: bool = False, profile: bool = False) -> None:
        """
        Reverse pass using derivative registry.
        Accumulates into original Value.grads.
        """
        t0 = time.perf_counter()

        N = len(self.nodes)
        grads = [0.0] * N
        grads[-1] = 1.0  # seed

        for i in range(N - 1, -1, -1):
            _ops.apply_grad(self.op[i], i, self.parents[i], self.data, grads, self.requires)

        # write grads back
        for i, v in enumerate(self.nodes):
            v.grad += grads[i]

        if not retain_graph:
            for v in self.nodes:
                # keep viz snapshot if present; value.py already handles this after classic backward
                v._prev = ()
                v._backward = lambda: None

        if profile:
            edges = sum(len(p) for p in self.parents)
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"[tape] nodes={N}, edges={edges}, backward={dt:.2f}ms")


def tape_backward(root: Value, *, retain_graph: bool = False, profile: bool = False) -> None:
    Tape.from_root(root).backward(retain_graph=retain_graph, profile=profile)
