"""Train a tiny neural network to learn XOR using the autograd engine."""
import argparse
import random
from autograd import Value, SGD

random.seed(0)

# dataset: inputs and expected outputs
inputs = [
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
]

# initialize weights and biases
w1 = [Value(random.uniform(-1, 1)) for _ in range(6)]  # 2x3 hidden layer weights
b1 = [Value(0.0) for _ in range(3)]
w2 = [Value(random.uniform(-1, 1)) for _ in range(3)]  # 3x1 output layer weights
b2 = Value(0.0)

parser = argparse.ArgumentParser(description="XOR training example")
parser.add_argument(
    "--epochs", type=int, default=300, help="number of training epochs (default: 300)"
)
parser.add_argument(
    "--activation",
    choices=["tanh", "relu", "sigmoid", "exp", "log", "sin", "cos"],
    default="tanh",
    help="activation function to apply",
)
args = parser.parse_args()


def act(v: Value) -> Value:
    if args.activation == "log":
        return ((v.relu() + (-v).relu()) + 1.0).log()
    return getattr(v, args.activation)()

params = w1 + b1 + w2 + [b2]
optimizer = SGD(params, lr=0.1)
for epoch in range(args.epochs):
    total_loss = Value(0.0)
    for x1, x2, y in inputs:
        # forward pass
        h = []
        for i in range(3):
            v = w1[2 * i] * x1 + w1[2 * i + 1] * x2 + b1[i]
            h.append(act(v))
        out = sum((h[i] * w2[i] for i in range(3)), b2)
        y_pred = act(out)
        loss = (y_pred - y) ** 2
        total_loss = total_loss + loss
    total_loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if epoch % 20 == 0:
        print(epoch, total_loss.data)