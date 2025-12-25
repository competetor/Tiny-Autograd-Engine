# examples/09_moons_mlp.py
"""
Two-moons classification with train/test split, optional plotting, and a label-shuffle sanity check.

Run:
  python examples/09_moons_mlp.py --epochs 600 --hidden 16 --lr 0.05 --noise 0.15 --test-split 0.3 --plot
  python examples/09_moons_mlp.py --sanity  # shuffle labels to confirm chance accuracy
"""
import argparse
import math
import random
from typing import List, Tuple
from autograd import MLP, Value, Adam, bce_with_logits, StepLR

def make_moons(n_samples=400, noise=0.1, seed=0) -> Tuple[List[Tuple[float,float]], List[float]]:
    random.seed(seed)
    X, Y = [], []
    n = n_samples // 2
    for _ in range(n):
        t = random.uniform(0.0, math.pi)
        r = 1.0 + random.gauss(0, noise)
        X.append((r * math.cos(t), r * math.sin(t))); Y.append(0.0)
    for _ in range(n):
        t = random.uniform(0.0, math.pi)
        r = 1.0 + random.gauss(0, noise)
        X.append((1.0 - r * math.cos(t), -r * math.sin(t) - 0.5)); Y.append(1.0)
    return X, Y

def split(X, Y, test_frac=0.3, seed=0):
    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    cut = int(len(X) * (1 - test_frac))
    tr = idx[:cut]; te = idx[cut:]
    Xtr = [X[i] for i in tr]; Ytr = [Y[i] for i in tr]
    Xte = [X[i] for i in te]; Yte = [Y[i] for i in te]
    return Xtr, Ytr, Xte, Yte

def acc_on(model, X, Y) -> float:
    correct = 0
    for (x1, x2), y in zip(X, Y):
        logit = model([Value(x1, requires_grad=False), Value(x2, requires_grad=False)])
        pred = 1.0 if logit.sigmoid().data >= 0.5 else 0.0
        correct += int(pred == y)
    return correct / len(X)

def maybe_plot(model, X, Y):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib not installed; skipping.")
        return
    xs = [p[0] for p in X]; ys = [p[1] for p in X]
    pad = 0.5
    gx = np.linspace(min(xs)-pad, max(xs)+pad, 300)
    gy = np.linspace(min(ys)-pad, max(ys)+pad, 300)
    Z = np.zeros((len(gy), len(gx)))
    for i, xv in enumerate(gx):
        for j, yv in enumerate(gy):
            out = model([Value(float(xv), requires_grad=False),
                         Value(float(yv), requires_grad=False)])
            Z[j, i] = out.sigmoid().data
    X0 = [p for p, y in zip(X, Y) if y == 0.0]
    X1 = [p for p, y in zip(X, Y) if y == 1.0]
    plt.figure(figsize=(6,5))
    cs = plt.contourf(gx, gy, Z, levels=50, alpha=0.7)
    if X0: plt.scatter([p[0] for p in X0], [p[1] for p in X0], c="black", s=12, label="class 0")
    if X1: plt.scatter([p[0] for p in X1], [p[1] for p in X1], c="white", edgecolors="black", s=12, label="class 1")
    plt.legend(); plt.title("Two Moons decision boundary"); plt.tight_layout(); plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--test-split", type=float, default=0.0, help="fraction held out for test (0..0.9)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--sanity", action="store_true", help="shuffle labels to confirm chance performance")
    args = ap.parse_args()

    X, Y = make_moons(n_samples=400, noise=args.noise, seed=args.seed)
    if args.sanity:        
        Yr = Y[:]
        random.Random(args.seed + 1).shuffle(Yr)
        Y = Yr

    if 0.0 < args.test_split < 0.9:
        Xtr, Ytr, Xte, Yte = split(X, Y, test_frac=args.test_split, seed=args.seed)
    else:
        Xtr, Ytr, Xte, Yte = X, Y, [], []

    model = MLP(2, [args.hidden, args.hidden, 1])
    opt = Adam(model.parameters(), lr=args.lr)
    sch = StepLR(opt, step_size=max(1, args.epochs // 4), gamma=0.5)

    for e in range(1, args.epochs + 1):
        opt.zero_grad()
        loss = Value(0.0, requires_grad=True)
        for (x1, x2), y in zip(Xtr, Ytr):
            logit = model([Value(x1, requires_grad=False), Value(x2, requires_grad=False)])
            loss = loss + bce_with_logits(logit, Value(y, requires_grad=False))
        loss = loss * (1.0 / len(Xtr))
        loss.backward()
        opt.step()
        sch.step()

        if e % max(1, args.epochs // 10) == 0:
            train_acc = acc_on(model, Xtr, Ytr)
            line = f"epoch {e:4d} | loss {loss.data:.5f} | train {train_acc:.3f}"
            if Xte:
                test_acc = acc_on(model, Xte, Yte)
                line += f" | test {test_acc:.3f}"
            print(line)

    if args.plot:        
        maybe_plot(model, X, Y)

if __name__ == "__main__":
    main()
