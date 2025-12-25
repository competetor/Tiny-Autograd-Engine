# autograd/early_stopping.py
from __future__ import annotations
from typing import Optional

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best: Optional[float] = None
        self.bad_steps: int = 0

    def _is_improvement(self, val: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return (self.best - val) > self.min_delta
        else:
            return (val - self.best) > self.min_delta

    def step(self, current: float) -> bool:
        if self._is_improvement(current):
            self.best = current
            self.bad_steps = 0
        else:
            self.bad_steps += 1
        return self.bad_steps >= self.patience
