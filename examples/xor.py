"""Train a tiny neural network to learn XOR using the autograd engine."""
import argparse
import random
from autograd import Value, SGD

random.seed(0)

# dataset: inputs and expected outputs
data = [
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
]
random.shuffle(data)
train_data, test_data = data[:3], data[3:]

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

def forward(x1: float, x2: float) -> Value:
    h = []
    for i in range(3):
        v = w1[2 * i] * x1 + w1[2 * i + 1] * x2 + b1[i]
        h.append(act(v))
    out = sum((h[i] * w2[i] for i in range(3)), b2)
    return act(out)


params = w1 + b1 + w2 + [b2]
optimizer = SGD(params, lr=0.1)
for epoch in range(args.epochs):
    total_loss = Value(0.0)
    for x1, x2, y in train_data:
        y_pred = forward(x1, x2)
        loss = (y_pred - y) ** 2
        total_loss = total_loss + loss
    total_loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if epoch % 20 == 0:
        print(epoch, total_loss.data)

correct = 0
for x1, x2, y in test_data:
    pred = forward(x1, x2)
    pred_label = 1.0 if pred.data > 0.5 else 0.0
    if pred_label == y:
        correct += 1
accuracy = correct / len(test_data)
print(f"accuracy: {accuracy:.2f}")
