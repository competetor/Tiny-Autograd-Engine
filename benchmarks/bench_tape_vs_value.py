# benchmarks/bench_tape_vs_value.py
from __future__ import annotations
import argparse, time, random
import numpy as np
from autograd import Value, MLP
from autograd.tape import tape_backward


def seed_all(seed: int = 0):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def timeit(fn, warmup: int = 1, reps: int = 5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def build_long_chain(n: int = 20_000):
    # single-scalar chain: plenty of ops to stress the engine
    x = Value(0.1)
    y = x
    for i in range(n):
        # mix of ops; keep log domain safe
        y = (y * 1.0001 + 0.0003).tanh() + (y * 0.5).exp() * 0.0001
    return y  # Value


def bench_chain(n: int):
    def classic():
        out = build_long_chain(n)
        out.grad = 0.0
        out.backward()
    def tape():
        out = build_long_chain(n)
        out.grad = 0.0
        tape_backward(out)
    return timeit(classic), timeit(tape)


def build_mlp_task(nin=32, widths=(64, 64, 64, 1), batch=128, steps=3):
    seed_all(0)
    model = MLP(nin, list(widths))
    xs = [[Value(random.uniform(-1, 1), requires_grad=False) for _ in range(nin)] for _ in range(batch)]

    def classic():
        total = Value(0.0, requires_grad=True)
        for x in xs:
            total = total + model(x)
        for p in model.parameters(): p.grad = 0.0
        total.backward()
    def tape():
        total = Value(0.0, requires_grad=True)
        for x in xs:
            total = total + model(x)
        for p in model.parameters(): p.grad = 0.0
        tape_backward(total)

    return model, xs, timeit(classic, reps=steps), timeit(tape, reps=steps)


def main():
    p = argparse.ArgumentParser(description="Tiny Autograd: tape vs classic backward")
    p.add_argument("--chain-len", type=int, default=20000, help="length of scalar op chain")
    p.add_argument("--mlp-nin", type=int, default=32)
    p.add_argument("--mlp-widths", type=int, nargs="+", default=[64, 64, 64, 1])
    p.add_argument("--mlp-batch", type=int, default=128)
    p.add_argument("--mlp-steps", type=int, default=5)
    args = p.parse_args()

    seed_all(0)

    print("== Scalar op chain ==")
    t_classic, t_tape = bench_chain(args.chain_len)
    print(f"classic backward: {t_classic*1000:.2f} ms/iter")
    print(f"tape backward   : {t_tape*1000:.2f} ms/iter")
    if t_tape > 0:
        print(f"speedup         : {t_classic / t_tape:.2f}×")

    print("\n== MLP batch sum ==")
    _, _, t_classic_mlp, t_tape_mlp = build_mlp_task(
        nin=args.mlp_nin,
        widths=tuple(args.mlp_widths),
        batch=args.mlp_batch,
        steps=args.mlp_steps,
    )
    print(f"classic backward: {t_classic_mlp*1000:.2f} ms/iter")
    print(f"tape backward   : {t_tape_mlp*1000:.2f} ms/iter")
    if t_tape_mlp > 0:
        print(f"speedup         : {t_classic_mlp / t_tape_mlp:.2f}×")


if __name__ == "__main__":
    main()
