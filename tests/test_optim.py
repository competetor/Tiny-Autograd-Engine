from autograd import Value, SGD, Adam

def test_sgd_converges():
    w = Value(0.0)
    opt = SGD([w], lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        loss = (w - 3) ** 2
        loss.backward()
        opt.step()
    assert abs(w.data - 3.0) < 1e-2


def test_adam_converges():
    w = Value(0.0)
    opt = Adam([w], lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        loss = (w + 2) ** 2
        loss.backward()
        opt.step()
    assert abs(w.data + 2.0) < 1e-2

if __name__ == "__main__":
    test_sgd_converges()
    test_adam_converges()
    print("All tests passed.")