# examples/hooks_graphviz_profiler.py
from autograd import Value
from autograd.utils import draw_graph

def main():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    out = ((a * b + c).tanh() + a).sigmoid()

    # pre-hook: clip huge upstream grads at this node
    def clip(v: Value, th=5.0):
        if abs(v.grad) > th:
            v.grad = th if v.grad > 0 else -th
    out.register_hook(clip, when="pre")

    out.backward(profile=True)
    print("a.grad, b.grad, c.grad =", a.grad, b.grad, c.grad)

    dot = draw_graph(out, render=False)
    # dot = draw_graph(out, render=True, filename="autograd_graph", directory=".", format="png") # if you have graphviz installed
    print("graph DOT header:", dot.source.splitlines()[0])

if __name__ == "__main__":
    main()
