from __future__ import annotations
from autograd import Tensor, Value, Linear

def test_matmul_matrix_matrix_and_grads():
    A = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])       # (2,3)
    B = Tensor([[10.0, 20.0],
                [30.0, 40.0],
                [50.0, 60.0]])          # (3,2)

    C = A @ B                           # (2,2)
    # numeric check
    assert [v.data for v in C.data[0]] == [1*10+2*30+3*50, 1*20+2*40+3*60]
    assert [v.data for v in C.data[1]] == [4*10+5*30+6*50, 4*20+5*40+6*60]

    s = C.sum()
    # grads: d/dA_{ik} = sum_j B_{kj}; d/dB_{kj} = sum_i A_{ik}
    s.backward()

    dA_expected = [[sum(B.data[k][j].data for j in range(2)) for k in range(3)] for _ in range(2)]
    for i in range(2):
        for k in range(3):
            assert A.data[i][k].grad == dA_expected[i][k]

    dB_expected = [[sum(A.data[i][k].data for i in range(2)) for j in range(2)] for k in range(3)]
    for k in range(3):
        for j in range(2):
            assert B.data[k][j].grad == dB_expected[k][j]

def test_matmul_matrix_vector_and_vector_matrix():
    A = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])  # (2,3)
    v = Tensor([7.0, 8.0, 9.0])    # (3,)
    y = A @ v                      # (2,)
    assert [y.data[i].data for i in range(2)] == [
        1*7 + 2*8 + 3*9,
        4*7 + 5*8 + 6*9
    ]

    M = Tensor([[1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0]])       # (3,2)
    u = v @ M                      # (2,)
    assert [u.data[i].data for i in range(2)] == [
        7*1 + 8*3 + 9*5,
        7*2 + 8*4 + 9*6
    ]

def test_dot_grad():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    s = a.dot(b)           # 1*4 + 2*5 + 3*6 = 32
    s.backward()
    assert s.data == 32.0
    # grads: ∂/∂a = b, ∂/∂b = a
    assert [v.grad for v in a.data] == [4.0, 5.0, 6.0]
    assert [v.grad for v in b.data] == [1.0, 2.0, 3.0]

def test_linear_single_forward_and_grads():
    lin = Linear(3, 2, bias=True)
    x = Tensor([1.0, 2.0, 3.0])       # (3,)
    y = lin(x)                        # (2,)
    # make a simple scalar loss
    s = y.sum()
    s.backward()

    # dL/dW[o,j] = x_j ; dL/db[o] = 1
    for o in range(2):
        for j in range(3):
            assert abs(lin.W[o][j].grad - x.data[j].data) < 1e-12
        assert abs(lin.b[o].grad - 1.0) < 1e-12

def test_linear_batch_forward_and_grads():
    lin = Linear(3, 2, bias=True)
    X = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])     # (2,3)
    Y = lin(X)                         # (2,2)
    s = Y.sum()
    s.backward()
    # dL/dW[o,j] = sum_i X[i,j] ; dL/db[o] = batch_size
    sums = [X.data[0][j].data + X.data[1][j].data for j in range(3)]
    for o in range(2):
        for j in range(3):
            assert abs(lin.W[o][j].grad - sums[j]) < 1e-12
        assert abs(lin.b[o].grad - 2.0) < 1e-12
