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
    assert abs(x.grad - (0.5 * (1 - 0.5))) < 1e-6

    # safe log via positive input
    x = Value(2.0)
    y = x.log()
    y.backward()
    assert abs(y.data - math.log(2.0)) < 1e-6
    assert abs(x.grad - 0.5) < 1e-6

def test_div_sin_cos():
    # division
    a = Value(4.0)
    b = Value(2.0)
    y = a / b
    y.backward()
    assert abs(a.grad - 1 / b.data) < 1e-6
    assert abs(b.grad + a.data / (b.data ** 2)) < 1e-6

    # sin
    x = Value(0.5)
    y = x.sin()
    y.backward()
    assert abs(y.data - math.sin(0.5)) < 1e-6
    assert abs(x.grad - math.cos(0.5)) < 1e-6

    # cos
    x = Value(0.5)
    y = x.cos()
    y.backward()
    assert abs(y.data - math.cos(0.5)) < 1e-6
    assert abs(x.grad + math.sin(0.5)) < 1e-6
