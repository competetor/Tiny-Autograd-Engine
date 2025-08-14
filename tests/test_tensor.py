from autograd import Tensor

def test_tensor_ops():
    t1 = Tensor([1, 2])
    t2 = Tensor([3, 4])
    t3 = t1 * t2 + 1
    s = t3.sum()
    s.backward()
    assert s.data == 13
    assert t1.data[0].grad == 3
    assert t1.data[1].grad == 4

if __name__ == "__main__":
    test_tensor_ops()
    print("All tests passed.")
