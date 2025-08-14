from __future__ import annotations
import math
from typing import Iterable, List
from .value import Value

class SGD:
    """Stochastic gradient descent with optional momentum."""
    def __init__(self, params: Iterable[Value], lr: float = 0.01, momentum: float = 0.0):
        self.params: List[Value] = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [0.0 for _ in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            v = self.momentum * self.velocity[i] + (1 - self.momentum) * p.grad
            p.data -= self.lr * v
            self.velocity[i] = v

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0


class Adam:
    """Adam optimizer."""
    def __init__(self, params: Iterable[Value], lr: float = 0.001,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.params: List[Value] = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [0.0 for _ in self.params]
        self.v = [0.0 for _ in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0
