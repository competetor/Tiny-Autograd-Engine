# autograd/functional.py
from __future__ import annotations
from typing import Callable, Iterable, List, Tuple, Union
from .value import Value

Number = Union[int, float]

def _as_value(x: Union[Value, Number]) -> Value:
    return x if isinstance(x, Value) else Value(float(x), requires_grad=True)

def grad(f: Callable[..., Value]) -> Callable[..., List[float]]:
    def wrapper(*args):
        xs = [_as_value(a) for a in args]
        for x in xs: x.grad = 0.0
        out = f(*xs)
        if not isinstance(out, Value):
            raise TypeError("f must return a Value")
        out.backward()
        return [x.grad for x in xs]
    return wrapper

def value_and_grad(f: Callable[..., Value]) -> Callable[..., Tuple[float, List[float]]]:
    def wrapper(*args):
        xs = [_as_value(a) for a in args]
        for x in xs: x.grad = 0.0
        out = f(*xs)
        if not isinstance(out, Value):
            raise TypeError("f must return a Value")
        out.backward()
        return out.data, [x.grad for x in xs]
    return wrapper

def jacobian(f: Callable[..., Union[Value, Iterable[Value]]]) -> Callable[..., List[List[float]]]:
    def wrapper(*args):
        xs = [_as_value(a) for a in args]
        for x in xs: x.grad = 0.0
        ys = f(*xs)
        ys_list = list(ys) if not isinstance(ys, Value) else [ys]
        J: List[List[float]] = []
        for i, yi in enumerate(ys_list):
            # retain graph until the last backward
            for x in xs: x.grad = 0.0
            yi.backward(retain_graph=(i != len(ys_list) - 1))
            J.append([x.grad for x in xs])
        return J
    return wrapper
