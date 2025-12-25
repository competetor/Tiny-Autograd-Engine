# examples/scalar_autodiff.py
from autograd import Value
from autograd.utils import draw_graph

def f(x: Value) -> Value:
    # a small nonlinear scalar function
    return ((x * x + 3.0 * x + 1.0).tanh() + (x * 0.7).sin() + (x + 2.0).log())

def main():
    x = Value(2.0)
    y = f(x)
    y.backward(profile=True)
    print(f"x= {x.data:.4f}, f(x)= {y.data:.6f}, df/dx= {x.grad:.6f}")

    # visualize (no render by default; won't require Graphviz system 'dot')
    dot = draw_graph(y, render=False)
    print("DOT graph preview:\n", dot.source.splitlines()[0], "...", sep="")

if __name__ == "__main__":
    main()
