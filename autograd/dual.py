# autograd/dual.py
from __future__ import annotations
import math
from typing import Union, Iterable, Tuple, Callable

Number = Union[int, float]

class Dual:
    """
    Dual number: stores (primal, tangent).
    Supports scalar forward-mode AD for common ops used in this repo.
    """
    __slots__ = ("p", "t")

    def __init__(self, primal: Number, tangent: Number = 0.0):
        self.p = float(primal)
        self.t = float(tangent)

    # ---------- arithmetic ----------
    @staticmethod
    def _coerce(x: Union["Dual", Number]) -> "Dual":
        return x if isinstance(x, Dual) else Dual(x, 0.0)

    def __add__(self, other: Union["Dual", Number]) -> "Dual":
        o = Dual._coerce(other)
        return Dual(self.p + o.p, self.t + o.t)

    __radd__ = __add__

    def __sub__(self, other: Union["Dual", Number]) -> "Dual":
        o = Dual._coerce(other)
        return Dual(self.p - o.p, self.t - o.t)

    def __rsub__(self, other: Union["Dual", Number]) -> "Dual":
        o = Dual._coerce(other)
        return Dual(o.p - self.p, o.t - self.t)

    def __mul__(self, other: Union["Dual", Number]) -> "Dual":
        o = Dual._coerce(other)
        return Dual(self.p * o.p, self.t * o.p + self.p * o.t)

    def __rmul__(self, other: Union["Dual", Number]) -> "Dual":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Dual", Number]) -> "Dual":
        o = Dual._coerce(other)
        if o.p == 0.0:
            raise ZeroDivisionError("Dual division by zero")
        inv = 1.0 / o.p
        return Dual(self.p * inv, (self.t * o.p - self.p * o.t) * (inv * inv))

    def __rtruediv__(self, other: Union["Dual", Number]) -> "Dual":
        o = Dual._coerce(other)
        return o.__truediv__(self)

    def __neg__(self) -> "Dual":
        return Dual(-self.p, -self.t)

    def __pow__(self, power: Number) -> "Dual":
        if not isinstance(power, (int, float)):
            raise TypeError("power must be numeric")
        if self.p == 0.0 and power < 1:
            # let math raise domain error if needed
            pass
        val = self.p ** power
        return Dual(val, power * (self.p ** (power - 1.0)) * self.t)

    # ---------- unary ----------
    def tanh(self) -> "Dual":
        tp = math.tanh(self.p)
        return Dual(tp, (1.0 - tp * tp) * self.t)

    def relu(self) -> "Dual":
        return Dual(self.p if self.p > 0 else 0.0, (self.t if self.p > 0 else 0.0))

    def exp(self) -> "Dual":
        e = math.exp(self.p)
        return Dual(e, e * self.t)

    def log(self) -> "Dual":
        if self.p <= 0.0:
            raise ValueError(f"log domain error: input must be > 0, got {self.p}")
        return Dual(math.log(self.p), (1.0 / self.p) * self.t)

    def sigmoid(self) -> "Dual":
        # stable enough for duals
        t = math.tanh(self.p * 0.5)
        s = 0.5 * (t + 1.0)
        return Dual(s, (s * (1.0 - s)) * self.t)

    def sin(self) -> "Dual":
        return Dual(math.sin(self.p), math.cos(self.p) * self.t)

    def cos(self) -> "Dual":
        return Dual(math.cos(self.p), -math.sin(self.p) * self.t)


def jvp(f: Callable[..., Dual], x: Iterable[Number], v: Iterable[Number]) -> Tuple[float, float]:
    xs = list(x); vs = list(v)
    if len(xs) != len(vs):
        raise ValueError("x and v must have same length")
    dx = [Dual(xi, vi) for xi, vi in zip(xs, vs)]
    out = f(*dx)
    if not isinstance(out, Dual):
        raise TypeError("f must return a Dual (scalar)")
    return out.p, out.t
