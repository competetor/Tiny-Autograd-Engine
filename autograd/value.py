from __future__ import annotations
import math
from typing import Callable, List, Optional, Set

class Value:
    """A scalar value supporting automatic differentiation."""
    def __init__(self, data: float, _children: Set['Value'] | None = None, _op: str = ""):
        self.data = data
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = _children if _children else set()
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: 'Value' | float) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, {self, other}, '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other: float) -> 'Value':
        return self + other

    def __neg__(self) -> 'Value':
        return self * -1

    def __sub__(self, other: 'Value' | float) -> 'Value':
        return self + (-other)

    def __rsub__(self, other: float) -> 'Value':
        return other + (-self)

    def __mul__(self, other: 'Value' | float) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, {self, other}, '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> 'Value':
        return self * other

    def __pow__(self, power: float) -> 'Value':
        assert isinstance(power, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** power, {self}, f'**{power}')

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> 'Value':
        t = math.tanh(self.data)
        out = Value(t, {self}, 'tanh')

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> 'Value':
        e = math.exp(self.data)
        out = Value(e, {self}, 'exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def backward(self) -> None:
        topo: List[Value] = []
        visited: Set[Value] = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
