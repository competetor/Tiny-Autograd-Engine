from __future__ import annotations
from autograd import Tensor, Value

def test_reshape_preserves_grads():
    A = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])      # (2,3)
    B = A.reshape(3, 2)                # (3,2)
    s = B.sum()
    s.backward()
    # every original element should see grad 1
    grads = [v.grad for row in A.data for v in row]
    assert grads == [1.0]*6

def test_transpose_2d_and_backward():
    A = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])      # (2,3)
    AT = A.transpose()                 # default (1,0) -> (3,2)
    # check values moved where expected
    assert [v.data for v in AT.data[0]] == [1.0, 4.0]
    assert [v.data for v in AT.data[1]] == [2.0, 5.0]
    assert [v.data for v in AT.data[2]] == [3.0, 6.0]
    # grads flow back correctly
    s = AT.sum()
    s.backward()
    grads = [v.grad for row in A.data for v in row]
    assert grads == [1.0]*6

def test_transpose_nd_axes():
    A = Tensor([[[1.0, 2.0],[3.0, 4.0]],
                [[5.0, 6.0],[7.0, 8.0]]])   # (2,2,2)
    B = A.transpose((2,1,0))                # (2,2,2)
    # sanity check: positions map; also grads should be 1s on sum
    s = B.sum()
    s.backward()
    grads = [v.grad for v in A.flat]
    assert grads == [1.0]*8
