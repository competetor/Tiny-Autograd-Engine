from __future__ import annotations
import random
from typing import Iterable, List
from .value import Value

class Neuron:
    """A single fully connected neuron with tanh activation."""
    def __init__(self, n_inputs: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x: List[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

class Layer:
    """A layer of neurons."""
    def __init__(self, n_inputs: int, n_outputs: int):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x: List[Value]) -> List[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    """A simple multilayer perceptron."""
    def __init__(self, n_inputs: int, sizes: Iterable[int]):
        sz = [n_inputs] + list(sizes)
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz) - 1)]

    def __call__(self, x: List[Value]) -> Value:
        for layer in self.layers:
            x = layer(x)
        assert len(x) == 1
        return x[0]

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    # serialization utilities
    def save(self, path: str) -> None:
        import pickle
        with open(path, 'wb') as f:
            pickle.dump([p.data for p in self.parameters()], f)

    @classmethod
    def load(cls, path: str, n_inputs: int, sizes: Iterable[int]) -> 'MLP':
        import pickle
        model = cls(n_inputs, sizes)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for p, d in zip(model.parameters(), data):
            p.data = d
        return model