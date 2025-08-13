# Tiny Autograd Engine

This repository implements a minimal automatic differentiation engine inspired by micrograd. It provides a `Value` class supporting basic operations and backpropagation, along with simple neural network building blocks.

## Features
- Scalar operations with gradients (addition, multiplication, power, `tanh`, `exp`, `relu`, `sigmoid`, `log`).
- Topological backpropagation.
- `Tensor` container for vectorized elementwise operations.
- `Neuron`, `Layer` and `MLP` classes for quickly assembling networks.
- Optimizers (`SGD`, `Adam`) to streamline training loops.
- Save and load utilities for `MLP` parameters.
- Example scripts training networks on XOR and a two-moons dataset.
- Packaging via `pyproject.toml` and CI with GitHub Actions.
- Optional GPU acceleration for tensors via `cupy` when available.

## Usage
Run an example:

```bash
python examples/xor.py --epochs 300       # XOR classification (override epochs if desired)
python examples/moons.py --epochs 500     # two-moons classification
```

Both scripts accept a `--epochs` flag to control training duration; defaults are 300 for XOR and 500 for the two-moons task.

Run tests with:

```bash
python tests/test_value.py
python tests/test_mlp.py
python tests/test_primitives.py
python tests/test_optim.py
python tests/test_tensor.py
python tests/test_serialization.py
```

