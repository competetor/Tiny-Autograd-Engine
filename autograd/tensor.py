from __future__ import annotations
import numpy as np
from functools import reduce
from typing import Any, Iterable
from .value import Value

try:  # optional GPU support
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cp = None  # type: ignore


class Tensor:
    """A lightweight container for values supporting elementwise ops."""

    def __init__(self, data: Iterable[Any]):
        arr = np.array(data, dtype=object)
        it = np.nditer(arr, flags=['multi_index', 'refs_ok'], op_flags=['readwrite'])
        for x in it:
            if not isinstance(x.item(), Value):
                arr[it.multi_index] = Value(float(x))
        self.data = arr

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def _op(self, other: Tensor | float, fn) -> 'Tensor':
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = np.vectorize(lambda a, b: fn(a, b), otypes=[object])(self.data, other_t.data)
        return Tensor(out)

    # elementwise operations
    def __add__(self, other: Tensor | float) -> 'Tensor':
        return self._op(other, lambda a, b: a + b)

    def __radd__(self, other: float) -> 'Tensor':
        return self + other

    def __mul__(self, other: Tensor | float) -> 'Tensor':
        return self._op(other, lambda a, b: a * b)

    def __rmul__(self, other: float) -> 'Tensor':
        return self * other

    def __neg__(self) -> 'Tensor':
        return self * -1

    def __sub__(self, other: Tensor | float) -> 'Tensor':
        return self + (-other)

    def __rsub__(self, other: float) -> 'Tensor':
        return (-self) + other

    def sum(self) -> Value:
        return reduce(lambda a, b: a + b, self.data.ravel(), Value(0.0))

    # simplistic GPU conversion helper
    def to_gpu(self) -> 'Tensor':
        if cp is None:
            raise RuntimeError('cupy is required for GPU tensors')
        arr = cp.array(self.data, dtype=object)
        return Tensor(arr)
