from __future__ import annotations
from autograd.early_stopping import EarlyStopping

def test_early_stopping_min_mode_triggers_after_patience():
    es = EarlyStopping(patience=3, min_delta=0.0, mode="min")
    vals = [1.0, 0.9, 0.9, 0.91, 0.915, 0.915, 0.916]  # improvements stop after idx=1
    stops = []
    for v in vals:
        stops.append(es.step(v))
    # After best at 0.9, we get 3 "bad" steps at 0.9, 0.91, 0.915 -> stop on the 4th element after best
    assert stops[-1] is True

def test_early_stopping_max_mode():
    es = EarlyStopping(patience=2, min_delta=0.05, mode="max")
    vals = [0.1, 0.12, 0.16, 0.17, 0.171, 0.171, 0.171]  # improvements slow down, then stall
    stop = False
    for v in vals:
        stop = es.step(v)
    assert stop is True
