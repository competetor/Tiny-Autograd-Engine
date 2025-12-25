# examples/tensor_ops_quickstart.py
from autograd import Tensor

def main():
    A = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])          # (2,3)
    b = Tensor([10.0, 20.0, 30.0])         # (3,)
    v = Tensor([0.5, -1.0, 2.0])           # (3,)

    out = (A + b) * 0.5                    # broadcast row, then scale
    s = out.sum()                          # scalar Value
    s.backward()
    print("sum(A+b)/2 =", s.data)
    print("d/dA is ones ->", [[v.grad for v in row] for row in A.data])  # grads of A

    A.zero_grad()
    b.zero_grad()
    y = A @ v                              # (2,)
    z = y.sum()
    z.backward()
    print("A@v =", [vv.data for vv in y.data])
    print("d/db (unused here) =", [vv.grad for vv in b.data])

if __name__ == "__main__":
    main()
