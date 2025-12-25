# tests/test_broadcast.py
from __future__ import annotations
import math
import pytest

from autograd import Tensor, Value


def test_scalar_broadcast_add_mul_sum():
    t = Tensor([1.0, 2.0, 3.0])
    out = t + 10  # broadcast scalar
    s = (out * 2).sum()  # -> Value
    s.backward()
    # out = [11,12,13], (out*2) grads wrt t are 2
    grads = [v.grad for v in t.data]
    assert grads == [2.0, 2.0, 2.0]
    assert s.data == (11 + 12 + 13) * 2


def test_row_vector_broadcast_to_matrix_and_backward():
    A = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])         # shape (2,3)
    b = Tensor([10.0, 20.0, 30.0])        # shape (3,)
    out = A + b                           # broadcast to (2,3)
    s = out.sum()
    s.backward()

    # d/dA of sum is 1 everywhere
    for row in A.data:
        for v in row:
            assert v.grad == 1.0
    # d/db is number of rows for each element (summed over broadcast axis)
    grads_b = [v.grad for v in b.data]
    assert grads_b == [2.0, 2.0, 2.0]


def test_col_vector_broadcast_to_matrix_and_backward():
    A = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])         # (2,3)
    c = Tensor([[100.0], [200.0]])        # (2,1)
    out = A + c                           # (2,3)
    s = out.sum()
    s.backward()

    # d/dA of sum is 1 everywhere
    for row in A.data:
        for v in row:
            assert v.grad == 1.0
    # d/dc is number of columns for each element
    grads_c = [row[0].grad for row in c.data]  # column vector
    assert grads_c == [3.0, 3.0]


def test_sum_axis_and_mean_axis():
    A = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])
    # sum over axis 0 -> [5,7,9]
    s0 = A.sum(axis=0)
    assert isinstance(s0, Tensor)
    assert [v.data for v in s0.data] == [5.0, 7.0, 9.0]
    # sum over axis 1 -> [6,15]
    s1 = A.sum(axis=1)
    assert isinstance(s1, Tensor)
    assert [v.data for v in s1.data] == [6.0, 15.0]
    # mean over all -> scalar
    m = A.mean()
    assert isinstance(m, Value)
    assert abs(m.data - (1+2+3+4+5+6)/6) < 1e-12


def test_broadcast_shape_mismatch_raises():
    A = Tensor([[1.0, 2.0, 3.0]])
    b = Tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        _ = A + b
