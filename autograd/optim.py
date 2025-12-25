# autograd/optim.py
from __future__ import annotations
from typing import Iterable, List, Dict, Any, Optional, Tuple, Union
from .value import Value
import math

ParamsLike = Union[Iterable[Value], List[Dict[str, Any]]]

def _flatten_params(param_groups: List[Dict[str, Any]]) -> List[Value]:
    return [p for g in param_groups for p in g["params"]]

class _BaseOpt:
    def __init__(self, params: ParamsLike):
        # normalize to param_groups: List[Dict]
        if isinstance(params, list) and params and isinstance(params[0], dict):
            self.param_groups: List[Dict[str, Any]] = []
            for g in params:
                group = {
                    "params": list(g["params"]),
                    # defaults filled by subclass
                }
                for k, v in g.items():
                    if k != "params":
                        group[k] = v
                self.param_groups.append(group)
        else:
            self.param_groups = [{"params": list(params)}]

    def parameters(self) -> List[Value]:
        return _flatten_params(self.param_groups)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

class SGD(_BaseOpt):
    """
    SGD with per-group lr, momentum, weight_decay (L2), and global (per-group) grad clip.
    Backward compatible with: SGD([params], lr=..., momentum=..., weight_decay=..., clip_norm=...)
    New style:
      SGD([
        {"params": group1, "lr": 0.1},
        {"params": group2, "lr": 1e-3, "weight_decay": 1e-2, "momentum": 0.9, "clip_norm": 1.0},
      ])
    """
    def __init__(
        self,
        params: ParamsLike,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        clip_norm: Optional[float] = None,
    ):
        super().__init__(params)
        # fill defaults
        for g in self.param_groups:
            g.setdefault("lr", float(lr))
            g.setdefault("momentum", float(momentum))
            g.setdefault("weight_decay", float(weight_decay))
            g.setdefault("clip_norm", float(clip_norm) if clip_norm is not None else None)

        # per-parameter velocity
        self._vel: Dict[int, float] = {id(p): 0.0 for p in self.parameters()}

    def step(self) -> None:
        # update state for new params (in case user rebuilt groups)
        for p in self.parameters():
            self._vel.setdefault(id(p), 0.0)

        for g in self.param_groups:
            lr = float(g["lr"])
            mom = float(g["momentum"])
            wd = float(g["weight_decay"])
            clip = g["clip_norm"]

            params: List[Value] = g["params"]

            # optional per-group clip by global L2 norm
            if clip is not None:
                total = math.sqrt(sum((p.grad * p.grad) for p in params))
                if total > 1e-12 and total > clip:
                    scale = clip / total
                    for p in params:
                        p.grad *= scale

            for p in params:
                g_tot = p.grad + (wd * p.data if wd != 0.0 else 0.0)
                vid = id(p)
                if mom != 0.0:
                    self._vel[vid] = mom * self._vel[vid] + g_tot
                    p.data -= lr * self._vel[vid]
                else:
                    p.data -= lr * g_tot

class Adam(_BaseOpt):
    """
    Adam with per-group hyperparams and decoupled weight decay (AdamW style) + optional clip.
    Backward compatible with Adam([params], lr=..., betas=(0.9,0.999), eps=1e-8, weight_decay=0.0, clip_norm=None)
    """
    def __init__(
        self,
        params: ParamsLike,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,   # decoupled
        clip_norm: Optional[float] = None,
    ):
        super().__init__(params)
        for g in self.param_groups:
            g.setdefault("lr", float(lr))
            g.setdefault("betas", tuple(betas))
            g.setdefault("eps", float(eps))
            g.setdefault("weight_decay", float(weight_decay))
            g.setdefault("clip_norm", float(clip_norm) if clip_norm is not None else None)

        self._m: Dict[int, float] = {}
        self._v: Dict[int, float] = {}
        self._t: int = 0

    def step(self) -> None:
        self._t += 1
        for g in self.param_groups:
            lr = float(g["lr"])
            b1, b2 = g["betas"]
            eps = float(g["eps"])
            wd = float(g["weight_decay"])
            clip = g["clip_norm"]
            params: List[Value] = g["params"]

            if clip is not None:
                total = math.sqrt(sum((p.grad * p.grad) for p in params))
                if total > 1e-12 and total > clip:
                    scale = clip / total
                    for p in params:
                        p.grad *= scale

            for p in params:
                pid = id(p)
                m = self._m[pid] = b1 * self._m.get(pid, 0.0) + (1.0 - b1) * p.grad
                v = self._v[pid] = b2 * self._v.get(pid, 0.0) + (1.0 - b2) * (p.grad * p.grad)

                mhat = m / (1.0 - (b1 ** self._t))
                vhat = v / (1.0 - (b2 ** self._t))

                # adam update
                p.data -= lr * (mhat / (math.sqrt(vhat) + eps))
                # decoupled weight decay
                if wd != 0.0:
                    p.data -= lr * wd * p.data
