# autograd/schedulers.py
from __future__ import annotations
from math import cos, pi
from typing import List

class _BaseLRScheduler:
    """
    Minimal, PyTorch-like LR scheduler operating on optimizer.param_groups.

    Usage:
        opt = SGD([...], lr=0.1)
        sch = StepLR(opt, step_size=10, gamma=0.1)
        for epoch in range(E):
            train(...)
            sch.step()   # updates group["lr"]
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        # snapshot base LRs
        self.base_lrs: List[float] = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
        self.last_epoch: int = -1  # PyTorch convention

    def get_lrs(self, epoch: int) -> List[float]:
        raise NotImplementedError

    def step(self) -> None:
        self.last_epoch += 1
        lrs = self.get_lrs(self.last_epoch)
        for g, lr in zip(self.optimizer.param_groups, lrs):
            g["lr"] = float(lr)

class StepLR(_BaseLRScheduler):
    """
    Decay LR by gamma every `step_size` steps.
    """
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def get_lrs(self, epoch: int) -> List[float]:
        factor = self.gamma ** (epoch // self.step_size)
        return [base * factor for base in self.base_lrs]

class CosineAnnealingLR(_BaseLRScheduler):
    """
    Cosine annealing from base_lr (at t=0) to eta_min (at t=T_max).
    After T_max, it keeps eta_min (no restarts).
    """
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        if T_max <= 0:
            raise ValueError("T_max must be > 0")
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)

    def _alpha(self, t: int) -> float:
        # clamp t ∈ [0, T_max]
        tt = min(max(t, 0), self.T_max)
        return 0.5 * (1.0 + cos(pi * tt / self.T_max))  # 1→0

    def get_lrs(self, epoch: int) -> List[float]:
        a = self._alpha(epoch)
        return [self.eta_min + (base - self.eta_min) * a for base in self.base_lrs]