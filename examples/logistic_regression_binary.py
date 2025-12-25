# examples/03_logistic_regression_binary.py
import argparse
import random
import math
from autograd import Value, SGD, bce_with_logits

def blobs(n_per=100, seed=0):
    random.seed(seed)
    X, Y = [], []
    for _ in range(n_per):
        r, t = random.uniform(0.0, 1.0), random.uniform(0, math.pi)
        x1, x2 = r * math.cos(t), r * math.sin(t)
        X.append((Value(x1), Value(x2)))
        Y.append(Value(0.0))
    for _ in range(n_per):
        r, t = random.uniform(0.0, 1.0), random.uniform(math.pi, 2*math.pi)
        x1, x2 = r * math.cos(t) + 0.5, r * math.sin(t) + 0.2
        X.append((Value(x1), Value(x2)))
        Y.append(Value(1.0))
    return X, Y

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.2)
    args = p.parse_args()

    # model: logit = w1*x1 + w2*x2 + b
    w1, w2, b = Value(0.0), Value(0.0), Value(0.0)
    opt = SGD([w1, w2, b], lr=args.lr)

    X, Y = blobs()
    for e in range(args.epochs):
        opt.zero_grad()
        loss = Value(0.0, requires_grad=True)
        for (x1, x2), y in zip(X, Y):
            logit = w1 * x1 + w2 * x2 + b
            loss = loss + bce_with_logits(logit, y)
        loss = loss * (1.0 / len(X))
        loss.backward()
        opt.step()
        if (e+1) % max(1, args.epochs // 10) == 0:
            # train accuracy
            correct = 0
            for (x1, x2), y in zip(X, Y):
                logit = w1 * x1 + w2 * x2 + b
                pred = 1.0 if logit.sigmoid().data >= 0.5 else 0.0
                correct += int(pred == y.data)
            print(f"epoch {e+1:3d} | loss {loss.data:.4f} | acc {correct/len(X):.3f}")

if __name__ == "__main__":
    main()
