# tests/test_schedulers.py
from __future__ import annotations
from autograd import Value, SGD
from autograd.schedulers import StepLR, CosineAnnealingLR

def test_step_lr_updates_lrs_on_boundaries():
    w = Value(0.0)
    opt = SGD([w], lr=0.1)
    sch = StepLR(opt, step_size=2, gamma=0.5)

    lrs = []
    for _ in range(5):
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    # epochs: 0,1 -> 0.1 ; 2,3 -> 0.05 ; 4 -> 0.025
    assert lrs == [0.1, 0.1, 0.05, 0.05, 0.025]

def test_cosine_annealing_lr_goes_from_base_to_min():
    w = Value(0.0)
    opt = SGD([w], lr=0.2)
    sch = CosineAnnealingLR(opt, T_max=4, eta_min=0.02)

    lrs = []
    for _ in range(5):
        sch.step()
        lrs.append(round(opt.param_groups[0]["lr"], 6))
    # t=0: base, t=4: eta_min, then stays
    assert lrs[0] == 0.2
    assert lrs[4] == 0.02
    # monotonic non-increasing over the window
    assert all(lrs[i] >= lrs[i+1] for i in range(4))
