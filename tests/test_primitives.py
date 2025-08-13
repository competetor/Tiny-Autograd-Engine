import math
from autograd import Value

def test_relu_sigmoid_log():
    # relu
    x = Value(-1.0)
    y = x.relu()
    y.backward()
    assert y.data == 0.0
    assert x.grad == 0.0

    x = Value(1.0)
    y = x.relu()
    y.backward()
    assert y.data == 1.0
    assert x.grad == 1.0

    # sigmoid
    x = Value(0.0)
    y = x.sigmoid()
    y.backward()
    assert abs(y.data - 0.5) < 1e-6
    assert abs(x.grad - 0.25) < 1e-6

    # log
    x = Value(2.0)
    y = x.log()
    y.backward()
    assert abs(y.data - math.log(2.0)) < 1e-6
    assert abs(x.grad - 0.5) < 1e-6
