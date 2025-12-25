![CI](https://github.com/competetor/Tiny-Autograd-Engine/actions/workflows/ci.yml/badge.svg?branch=main)

# Tiny-Autograd-Engine

Tiny-Autograd-Engine is a lightweight automatic differentiation library written entirely in Python.  
It is inspired by micrograd, but extends the concept into a compact deep-learning framework, providing both scalar and tensor-based autodiff, neural-network layers, optimizers, scheduling utilities, dual-number forward-mode differentiation, tape-based backpropagation, and support for custom operators.

The implementation is intentionally simple, readable, and educational — while remaining fully functional and numerically correct.

---

## Key Features

• Reverse-mode autodiff via a scalar `Value` class  
• Tensor abstraction supporting n-dimensional operations  
• Central operation registry defining forward + backward rules  
• Functional API including `grad`, `value_and_grad`, and `jacobian`  
• Forward-mode JVP via dual numbers  
• Finite-difference Hessian–vector products  
• Neural network utilities (`Linear`, `MLP`)  
• Optimizers (`SGD`, `Adam`) and learning-rate schedulers  
• Early stopping utilities  
• NumPy bridge (`Tensor.from_numpy`, `Tensor.to_numpy`)  
• Runtime extensibility via user-defined custom ops  
• Tape-based autodiff backend  
• Graph visualization helper  

All implemented in pure Python, making the system easy to understand and extend.

---

## Installation

**Optional:** You may want to run the install inside a virtual environment (e.g., `venv` or Conda) to keep dependencies isolated.

```bash
git clone https://github.com/competetor/Tiny-Autograd-Engine.git
cd Tiny-Autograd-Engine
pip install -e .

## Usage Examples

### Scalar Autodiff (`Value`)

    from autograd import Value

    x = Value(2.0, requires_grad=True)
    y = (x * 3 + 1).tanh()
    y.backward()
    print(x.grad)

---

### Working with Tensors

    from autograd import Tensor

    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = (x * 2).sum()
    y.backward()
    print(x.grad)

---

### NumPy Bridge

    import numpy as np
    from autograd import Tensor

    t = Tensor.from_numpy(np.array([[1., 2.], [3., 4.]]), requires_grad=True)
    (t * 2).sum().backward()
    print(t.to_numpy())

---

### Neural Network Example

    from autograd import MLP, mse, SGD, Tensor

    model = MLP(2, [16, 16, 1])
    optim = SGD(model.parameters(), lr=0.01)

    x = Tensor([[0.0, 1.0]])
    y = Tensor([[1.0]])

    pred = model(x)
    loss = mse(pred, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

---

### Functional Gradient API

    from autograd import grad, Value

    def f(x):
        return (x * x).tanh()

    df = grad(f)
    print(df(Value(2.0)))

---

### Forward-Mode JVP

    from autograd import jvp, Value

    def f(x):
        return x * x * x

    y, dy = jvp(f, Value(3.0), 1.0)
    print(y.data, dy)

---

### Tape Backend

    from autograd import Value, tape_backward

    x = Value(3)
    y = x * x
    tape_backward(y)

---

### Defining a Custom Operation

    from autograd import register_custom_op

    def cube_forward(xs):
        (x,) = xs
        return x ** 3

    def cube_grad(i, parents, data, grads, requires):
        (a,) = parents
        g = grads[i]
        if g and requires[a]:
            grads[a] += 3 * (data[a] ** 2) * g

    register_custom_op("cube", cube_forward, cube_grad, arity=1, pretty_name="x^3")

---

## Testing

A full test suite is included.

Run all tests with:

    python -m pytest

Expected output:

    47 passed, 1 skipped

---

## Project Goals

This project is designed to:

• Demonstrate how autodiff engines work internally  
• Provide a bridge between theory and practical deep-learning frameworks  
• Offer a clean codebase suitable for study, teaching, and experimentation  

It is **not intended as a production deep-learning framework**, but rather as a compact, understandable prototype.

---

## License

MIT — free to use, modify, and learn from.

---

If you find this project useful, consider starring the repository or contributing improvements.
