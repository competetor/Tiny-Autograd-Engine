# tests/conftest.py
import random
import os

try:
    import numpy as np  # optional
except Exception:  # pragma: no cover
    np = None

def pytest_configure(config):
    # deterministic runs across the suite
    random.seed(0)
    if np is not None:
        np.random.seed(0)
    # make sure we don't accidentally spam stdout in CI
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
