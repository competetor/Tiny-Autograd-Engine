# Tiny Autograd Engine

This repository implements a minimal automatic differentiation engine inspired by micrograd. It provides a `Value` class supporting basic operations and backpropagation, along with simple neural network building blocks.

## Features
- Scalar operations with gradients (addition, subtraction, multiplication, division, power, `tanh`, `exp`, `relu`, `sigmoid`, `log`, `sin`, `cos`).
- Topological backpropagation.
- `Tensor` container for vectorized elementwise operations.
- `Neuron`, `Layer` and `MLP` classes with configurable activations for quickly assembling networks.
- Optimizers (`SGD`, `Adam`) to streamline training loops.
- Save and load utilities for `MLP` parameters.
- Example scripts training networks on XOR and a two-moons dataset.
- Packaging via `pyproject.toml` and CI with GitHub Actions.
- Optional GPU acceleration for tensors via `cupy` when available.

## Usage
Run an example:

```bash
python examples/xor.py --epochs 300 --activation tanh       # XOR classification
python examples/moons.py --epochs 500 --activation tanh     # two-moons classification
```

Both scripts accept `--epochs` to control training duration and `--activation` to pick a primitive (`tanh`, `relu`, `sigmoid`, `exp`, `log`, `sin`, `cos`). The `log` option applies `log(|x| + 1)` to avoid invalid inputs. Defaults are 300 epochs and `tanh` for XOR, 500 epochs and `tanh` for two-moons.

Run tests with:

```bash
python tests/test_value.py
python tests/test_mlp.py
python tests/test_primitives.py
python tests/test_optim.py
python tests/test_tensor.py
python tests/test_serialization.py
```

