# tests/test_param_groups.py
from autograd import Value, SGD

def test_sgd_param_groups_different_lrs_one_step():
    w1, w2 = Value(1.0), Value(1.0)
    opt = SGD([
        {"params": [w1], "lr": 0.1},
        {"params": [w2], "lr": 0.001},
    ])

    opt.zero_grad()
    loss = (w1**2) + (w2**2)
    loss.backward()
    opt.step()

    assert abs(w1.data - 0.8) < 1e-9  # 1 - 0.1*(2*1) = 0.8
    assert abs(w2.data - 0.998) < 1e-9  # 1 - 0.001*(2*1) = 0.998

def test_sgd_weight_decay_effect():
    w = Value(1.0)
    opt = SGD([{"params": [w], "lr": 0.1, "weight_decay": 1.0}])

    opt.zero_grad()
    loss = (w**2)
    loss.backward()
    opt.step()

    # g=2, wd*param=1 => total grad=3 -> w=1 - 0.1*3 = 0.7
    assert abs(w.data - 0.7) < 1e-9
