"""Train an MLP on a two-moons dataset."""
import argparse
import math
import random
from autograd import Value, MLP, Adam

random.seed(0)


def make_moons(n_samples: int = 100, noise: float = 0.1):
    data = []
    for _ in range(n_samples):
        angle = random.uniform(0, math.pi)
        x1 = math.cos(angle)
        y1 = math.sin(angle)
        x1 += noise * (random.random() * 2 - 1)
        y1 += noise * (random.random() * 2 - 1)
        data.append((x1, y1, 0.0))

        angle = random.uniform(0, math.pi)
        x2 = 1 - math.cos(angle)
        y2 = -math.sin(angle) + 0.5
        x2 += noise * (random.random() * 2 - 1)
        y2 += noise * (random.random() * 2 - 1)
        data.append((x2, y2, 1.0))
    return data


data = make_moons(20, noise=0.1)

parser = argparse.ArgumentParser(description="two-moons training example")
parser.add_argument(
    "--epochs", type=int, default=500, help="number of training epochs (default: 500)"
)
parser.add_argument(
    "--activation",
    choices=["tanh", "relu", "sigmoid", "exp", "log", "sin", "cos"],
    default="tanh",
    help="activation function for MLP",
)
args = parser.parse_args()

model = MLP(2, [8, 8, 1], activation=args.activation)
optimizer = Adam(model.parameters(), lr=0.05)

for epoch in range(args.epochs):
    total_loss = Value(0.0)
    for x1, x2, label in data:
        y_pred = model([Value(x1), Value(x2)])
        total_loss = total_loss + (y_pred - label) ** 2
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(epoch, total_loss.data)

# report accuracy
correct = 0
for x1, x2, label in data:
    y_pred = model([Value(x1), Value(x2)]).data
    pred = 1.0 if y_pred > 0.5 else 0.0
    correct += pred == label
print("accuracy", correct / len(data))
