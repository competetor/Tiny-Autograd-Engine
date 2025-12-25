# tests/test_losses.py
import math
from autograd import Value, cross_entropy_with_logits

def softmax_probs(logits):
    m = max(logit.data for logit in logits)
    exps = [math.exp(logit.data - m) for logit in logits]
    s = sum(exps)
    return [e / s for e in exps]

def test_cross_entropy_with_logits_forward_and_grad():
    # three-class example
    logits = [Value(2.0), Value(-1.0), Value(0.5)]
    target = 0  # class 0
    loss = cross_entropy_with_logits(logits, target)
    loss.backward()

    # forward check against numeric LSE
    m = max(logit.data for logit in logits)
    lse_num = math.log(sum(math.exp(logit.data - m) for logit in logits)) + m
    assert abs(loss.data - (lse_num - logits[target].data)) < 1e-10

    # grad check: ∂L/∂logit_i = softmax_i - 1[i==target]
    ps = softmax_probs(logits)
    for i, logit in enumerate(logits):
        expected = ps[i] - (1.0 if i == target else 0.0)
        assert abs(logit.grad - expected) < 1e-6
