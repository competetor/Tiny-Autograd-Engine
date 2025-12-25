# examples/linear_regression.py
import argparse
import random
from autograd import Value, SGD, mse, StepLR

def make_data(n=200, seed=0):
    random.seed(seed)
    X, Y = [], []
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = 2.5 * x - 0.7 + random.gauss(0, 0.1)
        X.append(Value(x, requires_grad=False))
        Y.append(Value(y, requires_grad=False))
    return X, Y

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    args = p.parse_args()

    # model: y = w*x + b
    w, b = Value(0.0), Value(0.0)
    opt = SGD([w, b], lr=args.lr)
    sch = StepLR(opt, step_size=max(1, args.epochs // 4), gamma=0.5)

    X, Y = make_data()

    for e in range(args.epochs):
        opt.zero_grad()
        loss = Value(0.0, requires_grad=True)
        for x, y in zip(X, Y):
            yhat = w * x + b
            loss = loss + mse(yhat, y)
        loss = loss * (1.0 / len(X))
        loss.backward()
        opt.step()
        sch.step()
        if (e + 1) % max(1, args.epochs // 10) == 0:
            print(f"epoch {e+1:3d} | loss {loss.data:.6f} | w {w.data:.3f} b {b.data:.3f} | lr {opt.param_groups[0]['lr']:.3g}")

    print(f"final: w≈{w.data:.3f} (true 2.5), b≈{b.data:.3f} (true -0.7)")

if __name__ == "__main__":
    main()
