# examples/07_jvp_hvp_demo.py
from autograd import Value
from autograd.dual import jvp, Dual
from autograd.higher import hvp_fd

def f_val(x0: Value, x1: Value) -> Value:
    return (x0 * x1).sin() + (x0 - x1).exp() + (x0 * x0 + 1.0).log()

def f_dual(x: Dual, y: Dual) -> Dual:
    return (x * y).sin() + (x - y).exp() + (x * x + 1.0).log()

def main():
    x = [0.7, -0.3]
    v = [0.2, -0.4]

    y, jv = jvp(f_dual, x, v)
    print("JVP:", jv)

    hv = hvp_fd(f_val, x, v)
    print("HVP (finite diff):", hv)

if __name__ == "__main__":
    main()
