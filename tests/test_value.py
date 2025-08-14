from autograd import Value

def test_basic_grad():
    x = Value(3.0)
    y = (x * x) + (2 * x) + 1
    y.backward()
    assert abs(x.grad - 8.0) < 1e-6

if __name__ == "__main__":
    test_basic_grad()
    print("Basic gradient test passed.")
