# tests/test_safety.py
import pytest
from autograd import Value

def test_leaf_data_mutation_is_allowed():
    w = Value(1.0)
    w.data = 2.5  # should not raise
    assert w.data == 2.5

def test_nonleaf_data_mutation_is_blocked():
    x = Value(2.0)
    y = x * 3.0  # non-leaf (has parents)
    with pytest.raises(RuntimeError):
        y.data = 7.0

def test_stop_gradient_makes_node_mutable_and_breaks_chain():
    a = Value(2.0)
    z = (a + 1.0).tanh()
    z.stop_gradient()      # cut graph here
    z.data = 9.0           # now allowed

    out = z * 2.0          # this node won't require grad (parents had requires_grad=False)
    a.grad = 0.0
    out.backward()         # backprop shouldn't touch 'a'
    assert a.grad == 0.0
