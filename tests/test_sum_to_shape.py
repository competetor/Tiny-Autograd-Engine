from __future__ import annotations
import numpy as np
from autograd import Tensor, Value

def test_sum_to_shape_scalar_and_row():
    # Upstream grad shaped like (2, 3)
    G = Tensor([[Value(1.0), Value(2.0), Value(3.0)],
                [Value(4.0), Value(5.0), Value(6.0)]])  # shape (2,3)

    # Sum to scalar () -> Value(1+2+3+4+5+6)
    g_scalar = G.sum_to_shape(())
    assert isinstance(g_scalar, Value)
    assert abs(g_scalar.data - 21.0) < 1e-12

    # Sum to row (1,3): sum over axis 0
    g_row = G.sum_to_shape((1,3))
    assert isinstance(g_row, Tensor)
    assert g_row.shape == (1,3)

def test_sum_to_shape_column_and_vector():
    G = Tensor([[Value(1.0), Value(2.0), Value(3.0)],
                [Value(4.0), Value(5.0), Value(6.0)]])  # (2,3)

    # Sum to (2,1): sum over axis 1
    g_col = G.sum_to_shape((2,1))
    assert g_col.shape == (2,1)
    vals = [g_col.data[0][0].data, g_col.data[1][0].data]
    assert vals == [1+2+3, 4+5+6]

    # Sum to vector (3,) by summing axis 0
    g_vec = G.sum_to_shape((3,))
    assert isinstance(g_vec, Tensor)
    assert g_vec.shape == (3,)
    assert [v.data for v in g_vec.data] == [1+4, 2+5, 3+6]

def test_sum_to_shape_matches_numpy_broadcast_rules():
    # Simulate upstream grad from broadcasting (1,3) + (2,1) -> (2,3)
    a = Tensor([[Value(10.0), Value(20.0), Value(30.0)]])   # (1,3)
    b = Tensor([[Value(1.0)], [Value(2.0)]])                # (2,1)
    out = a + b                                             # (2,3)
    s = out.sum()                                           # scalar
    s.backward()  # grads: da broadcast along axis 0; db along axis 1

    # "Upstream grad" is all ones at (2,3)
    G = Tensor(np.ones((2,3), dtype=object))
    # Parameter shapes T_a = (1,3), T_b = (2,1)
    ga = G.sum_to_shape((1,3))
    gb = G.sum_to_shape((2,1))

    # Gradients accumulated on actual Value leaves should match these sums
    assert [v.grad for v in a.data[0]] == [2.0, 2.0, 2.0]
    assert [b.data[0][0].grad, b.data[1][0].grad] == [3.0, 3.0]
    # and the helper reductions match that logic numerically
    assert [v.data for v in ga.data[0]] == [2.0, 2.0, 2.0]
    assert [gb.data[0][0].data, gb.data[1][0].data] == [3.0, 3.0]
