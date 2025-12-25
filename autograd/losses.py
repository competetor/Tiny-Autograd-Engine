# autograd/losses.py
from __future__ import annotations
from typing import Union, List
from .value import Value
import math

Number = Union[int, float]

def _as_value(x: Union[Value, Number]) -> Value:
    return x if isinstance(x, Value) else Value(float(x), requires_grad=False)

def mse(pred: Union[Value, Number], target: Union[Value, Number]) -> Value:
    p = _as_value(pred)
    t = _as_value(target)
    diff = p - t
    return diff * diff

def bce_with_logits(logit: Union[Value, Number], target: Union[Value, Number]) -> Value:
    x = _as_value(logit)
    y = _as_value(target)
    absx = x.relu() + (-x).relu()
    lse = ((-absx).exp() + 1.0).log()
    max0 = x.relu()
    return max0 - x * y + lse

def cross_entropy_with_logits(logits: List[Value], target_ix: int) -> Value:
    """
    Multiclass CE on logits (stable). target_ix in [0, C-1].
      CE = logsumexp(logits) - logits[target]
      where logsumexp uses the 'max trick'.
    """
    if not logits:
        raise ValueError("logits must be non-empty")
    if not (0 <= target_ix < len(logits)):
        raise IndexError("target_ix out of range")

    # float max for stability
    m = max(l.data for l in logits)
    # sum exp(logit - m) with Value ops
    sum_exp = None
    for l in logits:
        term = (l - m).exp()
        sum_exp = term if sum_exp is None else (sum_exp + term)
    lse = sum_exp.log() + m
    return lse - logits[target_ix]

__all__ = ["mse", "bce_with_logits", "cross_entropy_with_logits"]
