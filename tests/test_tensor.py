from autograd import Tensor
import numpy as np

def test_tensor_ops():
    t1 = Tensor([1, 2])
    t2 = Tensor([3, 4])
    t3 = t1 * t2 + 1
    s = t3.sum()
    s.backward()
    assert s.data == 13
    assert t1.data[0].grad == 3
    assert t1.data[1].grad == 4

def test_numpy_bridge_roundtrip():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = Tensor.from_numpy(arr, requires_grad=True)

    # Do a simple differentiable operation
    out = (t * 2.0).sum()  # should be a scalar Value
    out.backward()

    # to_numpy should give us back the same numerical values
    out_arr = t.to_numpy()
    assert isinstance(out_arr, np.ndarray)
    assert out_arr.shape == arr.shape
    assert np.allclose(out_arr, arr)
