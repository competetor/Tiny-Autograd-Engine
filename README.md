# Tiny Autograd Engine

This repository implements a minimal automatic differentiation engine inspired by micrograd. It provides a `Value` class supporting basic operations and backpropagation, along with simple neural network building blocks.

## Features
- Scalar operations with gradients (addition, multiplication, power, `tanh`, `exp`).
- Topological backpropagation.
- `Neuron`, `Layer` and `MLP` classes for quickly assembling networks.
- Example scripts training networks on XOR and a two-moons dataset.

## Usage
Run an example:

```bash
python examples/xor.py       # XOR classification
python examples/moons.py     # two-moons classification
```

Run tests with:

```bash
python tests/test_value.py
python tests/test_mlp.py
```
