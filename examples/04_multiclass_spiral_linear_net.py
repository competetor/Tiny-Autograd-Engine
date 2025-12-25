# examples/04_multiclass_spiral_linear_net.py
import argparse, math, random
from autograd import Linear, Tensor, SGD, cross_entropy_with_logits

def make_spiral(n_per_class=100, k=3, noise=0.2, seed=0):
    random.seed(seed)
    X, Y = [], []
    for j in range(k):
        for i in range(n_per_class):
            r = i / n_per_class
            t = 1.75 * j + 4.0 * r + random.gauss(0, noise)
            x1 = r * math.sin(t)
            x2 = r * math.cos(t)
            X.append([x1, x2]); Y.append(j)
    return X, Y

def tanh_tensor(t: Tensor) -> Tensor:
    # elementwise tanh on a Tensor (keeps graph via Value.tanh())
    if len(t.shape) == 1:
        return Tensor([v.tanh() for v in t.data])  # type: ignore[index]
    return Tensor([[v.tanh() for v in row] for row in t.data])  # type: ignore[index]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()

    X, Y = make_spiral(k=args.k)
    # model: 2 -> hidden -> k (tanh in between)
    l1 = Linear(2, args.hidden, bias=True)
    l2 = Linear(args.hidden, args.k, bias=True)
    params = l1.parameters() + l2.parameters()
    opt = SGD(params, lr=args.lr, momentum=0.9)

    for e in range(args.epochs):
        opt.zero_grad()
        total = None
        correct = 0
        for (x1, x2), y in zip(X, Y):
            x = Tensor([x1, x2])
            h = l1(x)
            h = tanh_tensor(h)
            logits = l2(h)  # Tensor length k
            loss = cross_entropy_with_logits([v for v in logits.data], y)  # type: ignore[index]
            total = loss if total is None else (total + loss)

            # prediction
            logits_vals = [v.data for v in logits.data]  # type: ignore[index]
            pred = max(range(args.k), key=lambda i: logits_vals[i])
            correct += int(pred == y)

        total = total * (1.0 / len(X))
        total.backward()
        opt.step()
        if (e+1) % max(1, args.epochs // 10) == 0:
            print(f"epoch {e+1:3d} | loss {total.data:.4f} | acc {correct/len(X):.3f}")

if __name__ == "__main__":
    main()
