# autograd/nn.py
from __future__ import annotations

import pickle
import random
from typing import List, Union

from .value import Value
from .tensor import Tensor

Number = Union[int, float]


class Module:
    def parameters(self) -> List[Value]:
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    # ---- simple serialization API ----

    def state_dict(self) -> List[float]:
        """Flat list of parameter .data in parameters() order."""
        return [p.data for p in self.parameters()]

    def load_state_dict(self, weights: List[float]) -> None:
        """Assign parameter .data from a flat list in parameters() order."""
        params = self.parameters()
        if len(weights) != len(params):
            raise ValueError(f"load_state_dict: size mismatch (got {len(weights)} values, "
                             f"need {len(params)})")
        for p, w in zip(params, weights):
            # leaf params are safe to mutate
            p.data = float(w)

    def save(self, path: str) -> None:
        """Save just the flat weight list (keeps file format simple for tests)."""
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)


class Neuron(Module):
    def __init__(self, nin: int, nonlin: bool = True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: List[Number] | List[Value]):
        act = self.b
        for wi, xi in zip(self.w, x):
            xi = xi if isinstance(xi, Value) else Value(float(xi), requires_grad=False)
            act = act + wi * xi
        return act.tanh() if self.nonlin else act

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin: int, nout: int, **kw):
        self.neurons = [Neuron(nin, **kw) for _ in range(nout)]

    def __call__(self, x: List[Number] | List[Value]):
        return [n(x) for n in self.neurons]

    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
                       for i in range(len(nouts))]

    def __call__(self, x: List[Number] | List[Value]):
        for layer in self.layers:
            x = layer(x)
        # final layer returns a single scalar Value
        return x[0]

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    # class-level loader used by tests
    @classmethod
    def load(cls, path: str, nin: int, nouts: List[int]) -> "MLP":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        # support either raw list or {'weights': list}
        weights = blob.get("weights") if isinstance(blob, dict) else blob
        model = cls(nin, nouts)
        model.load_state_dict(weights)
        return model


class Linear(Module):
    """
    Linear(in_features, out_features, bias=True)

    Accepts:
      - Tensor 1-D: shape (in_features,) -> returns Tensor (out_features,)
      - Tensor 2-D: shape (batch, in_features) -> returns Tensor (batch, out_features)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        # weights: (out_features, in_features)
        self.W: List[List[Value]] = [
            [Value(random.uniform(-1, 1)) for _ in range(self.in_features)]
            for _ in range(self.out_features)
        ]
        self.b: List[Value] = [Value(0.0) for _ in range(self.out_features)] if bias else []

    def __call__(self, x: Tensor | List[Number] | List[Value]) -> Tensor:
        if isinstance(x, Tensor):
            shape = x.shape
            if len(shape) == 1:
                if shape[0] != self.in_features:
                    raise ValueError(f"expected input dim {self.in_features}, got {shape[0]}")
                out: List[Value] = []
                for o in range(self.out_features):
                    acc = self.W[o][0] * x.data[0]  # type: ignore[index]
                    for j in range(1, self.in_features):
                        acc = acc + self.W[o][j] * x.data[j]  # type: ignore[index]
                    if self.b:
                        acc = acc + self.b[o]
                    out.append(acc)
                return Tensor(out)

            elif len(shape) == 2:
                m, n = shape
                if n != self.in_features:
                    raise ValueError(f"expected input dim {self.in_features}, got {n}")
                rows: List[List[Value]] = []
                for i in range(m):
                    row: List[Value] = []
                    for o in range(self.out_features):
                        acc = self.W[o][0] * x.data[i][0]  # type: ignore[index]
                        for j in range(1, self.in_features):
                            acc = acc + self.W[o][j] * x.data[i][j]  # type: ignore[index]
                        if self.b:
                            acc = acc + self.b[o]
                        row.append(acc)
                    rows.append(row)
                return Tensor(rows)
            else:
                raise ValueError("Input tensor must be 1-D or 2-D")
        else:
            # treat as 1-D vector list
            if len(x) != self.in_features:  # type: ignore[arg-type]
                raise ValueError(f"expected input dim {self.in_features}, got {len(x)}")
            vals: List[Value] = [
                xi if isinstance(xi, Value) else Value(float(xi), requires_grad=False)
                for xi in x  # type: ignore[arg-type]
            ]
            return self(Tensor(vals))

    def parameters(self) -> List[Value]:
        ps = [w for row in self.W for w in row]
        ps += self.b
        return ps

    def __repr__(self) -> str:
        return f"Linear({self.in_features}, {self.out_features}, bias={bool(self.b)})"
