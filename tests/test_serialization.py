import os
import tempfile
from autograd import MLP


def test_model_save_load():
    model = MLP(2, [3, 1])
    params_before = [p.data for p in model.parameters()]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'model.pkl')
        model.save(path)
        loaded = MLP.load(path, 2, [3, 1])
    params_after = [p.data for p in loaded.parameters()]
    assert params_before == params_after

if __name__ == "__main__":
    test_model_save_load()
    print("Model save/load test passed.")
