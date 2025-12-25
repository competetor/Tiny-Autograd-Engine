# examples/08_xor_mlp.py
"""
XOR with a tiny MLP + optional decision boundary plot.
Run:
  python examples/08_xor_mlp.py --epochs 2000 --lr 0.1 --plot
"""
import argparse
from autograd import MLP, Value, SGD, bce_with_logits

def xor_data():
    # (x1, x2) -> y
    return [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ]

def train(epochs: int, lr: float, hidden: int = 8):
    model = MLP(2, [hidden, 1])           # 2→H→1
    opt = SGD(model.parameters(), lr=lr, momentum=0.9)
    data = xor_data()

    for e in range(1, epochs + 1):
        opt.zero_grad()
        loss = Value(0.0, requires_grad=True)
        correct = 0

        for x1, x2, y in data:
            out_logit = model([Value(x1, requires_grad=False), Value(x2, requires_grad=False)])
            loss = loss + bce_with_logits(out_logit, Value(y, requires_grad=False))
            pred = 1.0 if out_logit.sigmoid().data >= 0.5 else 0.0
            correct += int(pred == y)

        loss = loss * (1.0 / len(data))
        loss.backward()
        opt.step()

        if e % max(1, epochs // 10) == 0:
            print(f"epoch {e:4d} | loss {loss.data:.6f} | acc {correct/4:.3f}")
    return model

def _maybe_plot(model):
    try:
        import numpy as np  # optional
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib not installed; skipping plot.")
        return

    # grid over [−0.5, 1.5]²
    xs = np.linspace(-0.5, 1.5, 200)
    ys = np.linspace(-0.5, 1.5, 200)
    Z = np.zeros((len(xs), len(ys)))
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            out = model([Value(float(xv), requires_grad=False),
                         Value(float(yv), requires_grad=False)])
            Z[j, i] = out.sigmoid().data  # note transpose on display

    # data points
    pts = xor_data()
    X0 = [(x1, x2) for x1, x2, y in pts if y == 0.0]
    X1 = [(x1, x2) for x1, x2, y in pts if y == 1.0]

    plt.figure(figsize=(5, 5))
    plt.contourf(xs, ys, Z, levels=50, alpha=0.7)
    if X0:
        plt.scatter([p[0] for p in X0], [p[1] for p in X0], c="black", s=80, label="class 0")
    if X1:
        plt.scatter([p[0] for p in X1], [p[1] for p in X1], c="white", edgecolors="black", s=80, label="class 1")
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.legend(loc="best")
    plt.title("XOR decision boundary")
    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    model = train(args.epochs, args.lr, args.hidden)
    if args.plot:
        _maybe_plot(model)

if __name__ == "__main__":
    main()
