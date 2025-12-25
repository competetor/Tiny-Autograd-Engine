# tests/test_mlp.py
from autograd import Value, MLP

def test_mlp_xor():
    model = MLP(2, [3, 1])
    data = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ]
    for _ in range(100):
        loss = Value(0.0)
        for x1, x2, y in data:
            y_pred = model([Value(x1), Value(x2)])
            loss = loss + (y_pred - y) ** 2
        loss.backward()
        for p in model.parameters():
            p.data -= 0.1 * p.grad
            p.grad = 0.0
    preds = [1.0 if model([Value(x1), Value(x2)]).data > 0.5 else 0.0 for x1, x2, _ in data]
    assert preds == [0.0, 1.0, 1.0, 0.0]
