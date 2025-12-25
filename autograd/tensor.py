# autograd/tensor.py
from __future__ import annotations
from typing import List, Union, Tuple, Any, Sequence, Callable
from .value import Value

Number = Union[int, float]
Leaf = Union[Value, Number]
Nested = Union[Leaf, List["Nested"]]

# --- leaf vs constant wrappers ---
def _as_leaf(x: Leaf) -> Value:
    return x if isinstance(x, Value) else Value(float(x), requires_grad=True)

def _as_const(x: Leaf) -> Value:
    return x if isinstance(x, Value) else Value(float(x), requires_grad=False)

# --- shape & structure helpers ---
def _is_value_or_number(x: Any) -> bool:
    return isinstance(x, (Value, int, float))

def _infer_shape(x: Nested) -> Tuple[int, ...]:
    if _is_value_or_number(x):
        return ()
    if not isinstance(x, list):
        raise TypeError("Tensor data must be a Value/number or nested list of them")
    if len(x) == 0:
        raise ValueError("Empty lists are not supported")
    child_shape = _infer_shape(x[0])
    for el in x[1:]:
        if _infer_shape(el) != child_shape:
            raise ValueError("Ragged (non-rectangular) lists are not supported")
    return (len(x),) + child_shape

def _map_nested(x: Nested, fn: Callable[[Leaf], Leaf]) -> Nested:
    if _is_value_or_number(x):
        return fn(x)
    return [_map_nested(el, fn) for el in x]  # type: ignore[index]

def _zip_map_nested(a: Nested, b: Nested, fn):
    if _is_value_or_number(a) and _is_value_or_number(b):
        return fn(a, b)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [_zip_map_nested(aa, bb, fn) for aa, bb in zip(a, b)]
    raise ValueError("Shape mismatch in _zip_map_nested")

def _broadcast_shape(sa: Tuple[int, ...], sb: Tuple[int, ...]) -> Tuple[int, ...]:
    ra, rb = sa[::-1], sb[::-1]
    out_rev: List[int] = []
    for i in range(max(len(ra), len(rb))):
        da = ra[i] if i < len(ra) else 1
        db = rb[i] if i < len(rb) else 1
        if da == db or da == 1 or db == 1:
            out_rev.append(max(da, db))
        else:
            raise ValueError(f"Shapes {sa} and {sb} not broadcastable")
    return tuple(out_rev[::-1])

def _broadcast_to_data(data: Nested, from_shape: Tuple[int, ...], to_shape: Tuple[int, ...]) -> Nested:
    if len(from_shape) < len(to_shape):
        pad = (1,) * (len(to_shape) - len(from_shape))
        from_shape = pad + from_shape
        for _ in range(len(pad)):
            data = [data]  # type: ignore[list-item]
    if from_shape == to_shape:
        return data
    if not isinstance(data, list):
        return [_broadcast_to_data(data, (), to_shape[1:]) for _ in range(to_shape[0])]
    out: List[Nested] = []
    for i in range(to_shape[0]):
        src_idx = 0 if (from_shape[0] == 1) else i
        child = _broadcast_to_data(data[src_idx], from_shape[1:], to_shape[1:])
        out.append(child)
    return out

def _flatten_refs(x: Nested) -> List[Value]:
    if _is_value_or_number(x):
        if isinstance(x, Value):
            return [x]
        return [Value(float(x))]
    out: List[Value] = []
    for el in x:
        out.extend(_flatten_refs(el))  # type: ignore[arg-type]
    return out

def _build_from_flat(flat: List[Value], shape: Tuple[int, ...]):
    it = iter(flat)
    def build(sh: Tuple[int, ...]):
        if not sh:
            return next(it)
        return [build(sh[1:]) for _ in range(sh[0])]
    return build(shape)

def _index_get(nested: Nested, idx: Tuple[int, ...]) -> Value:
    cur = nested
    for i in idx:
        cur = cur[i]  # type: ignore[index]
    return cur  # type: ignore[return-value]

#def _iter_indices(shape: Tuple[int, ...]):
#    if not shape:
#        yield ()
#    else:
#        from itertools import product
#        for idx in product(*[range(d) for d in shape]):
#            yield idx

class Tensor:
    __slots__ = ("data",)

    def __init__(self, data):
        """
        Accepts:
          - a Tensor (copies its nested-list data),
          - a Value or Python number,
          - nested lists/tuples/ranges/generators of Values/numbers,
          - NumPy ndarrays/scalars at ANY nesting depth.
        Normalizes to a nested Python list of `Value`s and stores it in self.data.
        (Do NOT assign to self.shape; your property computes it from self.data.)
        """
        # Local imports to avoid circulars / hard deps at module import
        try:
            import numpy as _np
        except Exception:
            _np = None

        from autograd.value import Value
        from collections.abc import Iterable, Sequence

        # Copy a nested list structure (keeping Value objects)
        def _deepcopy_list(x):
            if isinstance(x, list):
                return [_deepcopy_list(t) for t in x]
            return x

        # Convert arbitrary iterables to lists (but not str/bytes)
        def _to_list(x):
            if _np is not None and isinstance(x, _np.ndarray):
                return x.tolist()
            if isinstance(x, Tensor):                 # Tensor → copy its nested list
                return _deepcopy_list(x.data)
            if isinstance(x, (list, tuple, range)):
                return list(x)
            if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
                return list(x)
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                return list(x)
            return x

        def _normalize(x):
            x = _to_list(x)
            # Value stays as-is
            if isinstance(x, Value):
                return x
            # Python numeric → Value  (requires_grad=True so grads flow)
            if isinstance(x, (int, float)):
                return Value(float(x), requires_grad=True)
            # NumPy scalar → Value  (requires_grad=True so grads flow)
            if _np is not None and isinstance(x, _np.generic):
                return Value(float(x), requires_grad=True)
            # Nested sequence → recurse
            if isinstance(x, list):
                return [_normalize(t) for t in x]
            raise TypeError("Tensor data must be a Value/number or nested list of them")

        nested = _normalize(data)

        # Validate rectangularity (your shape/ops rely on this)
        def _infer_shape_local(x):
            if isinstance(x, Value):
                return ()
            if not isinstance(x, list):
                raise TypeError("Tensor data must be a Value/number or nested list of them")
            if len(x) == 0:
                return (0,)
            child = _infer_shape_local(x[0])
            for y in x[1:]:
                if _infer_shape_local(y) != child:
                    raise ValueError("Jagged nested lists are not allowed")
            return (len(x),) + child

        _ = _infer_shape_local(nested)

        # Store as nested Python lists (NOT a NumPy array)
        self.data = nested

    @property
    def shape(self) -> Tuple[int, ...]:
        return _infer_shape(self.data)

    def __len__(self) -> int:
        s = self.shape
        return s[0] if s else 1

    def __repr__(self) -> str:
        s = self.shape
        if len(s) == 1:
            arr = ", ".join(f"{v.data:.6g}" for v in self.data)  # type: ignore[index]
            return f"Tensor([{arr}])"
        return f"Tensor(shape={s})"

    def __getitem__(self, idx):
        return self.data[idx]  # type: ignore[index]

    @property
    def flat(self) -> List[Value]:
        return _flatten_refs(self.data)

    # --- elementwise ops (with broadcasting) ---
    def _binary_ew(self, other: Union["Tensor", Value, Number], fn):
        a_data, sa = self.data, self.shape
        if isinstance(other, Tensor):
            b_data, sb = other.data, other.shape
        else:
            b_data, sb = _as_const(other), ()
        out_shape = _broadcast_shape(sa, sb)
        a_b = _broadcast_to_data(a_data, sa, out_shape)
        b_b = _broadcast_to_data(b_data, sb, out_shape)
        def cast(v: Leaf, const: bool) -> Value:
            if isinstance(v, Value):
                return v
            return _as_const(v) if const else _as_leaf(v)
        out = _zip_map_nested(a_b, b_b, lambda aa, bb: fn(cast(aa, False), cast(bb, True)))
        return Tensor(out)

    def __add__(self, other):  return self._binary_ew(other, lambda a, b: a + b)
    def __radd__(self, other): return self._binary_ew(other, lambda a, b: a + b)
    def __sub__(self, other):  return self._binary_ew(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._binary_ew(other, lambda a, b: b - a)
    def __mul__(self, other):  return self._binary_ew(other, lambda a, b: a * b)
    def __rmul__(self, other): return self._binary_ew(other, lambda a, b: a * b)
    def __truediv__(self, other):  return self._binary_ew(other, lambda a, b: a / b)
    def __rtruediv__(self, other): return self._binary_ew(other, lambda a, b: b / a)

    # --- matrix multiply & dot ---
    def matmul(self, other: "Tensor") -> "Tensor | Value":
        sa, sb = self.shape, other.shape

        # vector @ vector -> scalar (dot)
        if len(sa) == 1 and len(sb) == 1:
            if sa[0] != sb[0]:
                raise ValueError("dot product requires same length vectors")
            n = sa[0]
            acc = _as_const(0.0)
            for k in range(n):
                acc = acc + _index_get(self.data, (k,)) * _index_get(other.data, (k,))
            return acc

        # matrix @ vector -> vector
        if len(sa) == 2 and len(sb) == 1:
            m, n = sa
            if n != sb[0]:
                raise ValueError(f"Incompatible shapes {sa} and {sb} for matmul")
            out_row: List[Value] = []
            for i in range(m):
                acc = _index_get(self.data, (i, 0)) * _index_get(other.data, (0,))
                for k in range(1, n):
                    acc = acc + _index_get(self.data, (i, k)) * _index_get(other.data, (k,))
                out_row.append(acc)
            return Tensor(out_row)

        # vector @ matrix -> vector
        if len(sa) == 1 and len(sb) == 2:
            n, = sa
            n2, p = sb
            if n != n2:
                raise ValueError(f"Incompatible shapes {sa} and {sb} for matmul")
            out: List[Value] = []
            for j in range(p):
                acc = _index_get(self.data, (0,)) * _index_get(other.data, (0, j))
                for k in range(1, n):
                    acc = acc + _index_get(self.data, (k,)) * _index_get(other.data, (k, j))
                out.append(acc)
            return Tensor(out)

        # matrix @ matrix -> matrix
        if len(sa) == 2 and len(sb) == 2:
            m, n = sa
            n2, p = sb
            if n != n2:
                raise ValueError(f"Incompatible shapes {sa} and {sb} for matmul")
            out: List[List[Value]] = []
            for i in range(m):
                row: List[Value] = []
                for j in range(p):
                    acc = _index_get(self.data, (i, 0)) * _index_get(other.data, (0, j))
                    for k in range(1, n):
                        acc = acc + _index_get(self.data, (i, k)) * _index_get(other.data, (k, j))
                    row.append(acc)
                out.append(row)
            return Tensor(out)

        raise ValueError(f"matmul not implemented for shapes {sa} and {sb}")

    def __matmul__(self, other: "Tensor"):  return self.matmul(other)
    def __rmatmul__(self, other: "Tensor"): return Tensor(other).matmul(self)  # type: ignore[arg-type]

    def dot(self, other: "Tensor") -> Value:
        out = self.matmul(other)
        if not isinstance(out, Value):
            raise ValueError("dot expects 1-D · 1-D -> Value")
        return out

    # --- reductions ---
    def sum(self, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False):
        data = self.data
        shape = self.shape
        if axis is None:
            flat = _flatten_refs(data)
            if not flat:
                return Value(0.0, requires_grad=False)
            acc = flat[0]
            for v in flat[1:]:
                acc = acc + v
            return acc
        if isinstance(axis, int):
            axes = (axis,)
        else:
            axes = tuple(axis)
        axes = tuple(a if a >= 0 else a + len(shape) for a in axes)
        if any(a < 0 or a >= len(shape) for a in axes):
            raise ValueError("axis out of range")
        def reduce_axis0(lst: List[Nested]) -> Nested:
            if len(lst) == 1:
                return lst[0]
            acc = lst[0]
            for el in lst[1:]:
                acc = _zip_map_nested(acc, el, lambda x, y: _as_leaf(x) + _as_leaf(y))
            return acc
        def sum_along(d: Nested, ax: int) -> Nested:
            if ax == 0:
                if not isinstance(d, list):
                    return d
                return reduce_axis0(d)
            if not isinstance(d, list):
                raise ValueError("axis too deep for data")
            return [sum_along(el, ax - 1) for el in d]  # type: ignore[list-item]
        axes_sorted = sorted(set(axes))
        out = data
        for i, a in enumerate(axes_sorted):
            out = sum_along(out, a - i)
        if keepdims:
            for _ in axes_sorted:
                out = [out]  # type: ignore[list-item]
        if _is_value_or_number(out):
            return _as_leaf(out)
        return Tensor(out)

    def mean(self, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False):
        if axis is None:
            total = self.sum()
            count = 1
            for d in self.shape:
                count *= d
            return total / count
        if isinstance(axis, int):
            axes = (axis,)
        else:
            axes = tuple(axis)
        axes = tuple(a if a >= 0 else a + len(self.shape) for a in axes)
        count = 1
        for a in set(axes):
            count *= self.shape[a]
        s = self.sum(axis=axes, keepdims=keepdims)
        if isinstance(s, Value):
            return s / count
        return s * (1.0 / count)

    # --- reshape & transpose ---
    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])  # type: ignore[assignment]
        flat = _flatten_refs(self.data)
        total = 1
        for d in self.shape:
            total *= d
        total_new = 1
        for d in shape:
            total_new *= d
        if total != total_new:
            raise ValueError(f"cannot reshape {self.shape} into {shape} (different sizes)")
        new_nested = _build_from_flat(flat, tuple(shape))
        return Tensor(new_nested)

    def transpose(self, axes: Union[None, Tuple[int, ...], List[int]] = None) -> "Tensor":
        old_shape = self.shape
        ndim = len(old_shape)
        if axes is None:
            if ndim != 2:
                raise ValueError("transpose default only defined for 2-D tensors; pass axes for N-D")
            axes = (1, 0)
        axes = tuple(axes)  # type: ignore[assignment]
        if sorted(axes) != list(range(ndim)):
            raise ValueError("axes must be a permutation of range(ndim)")
        new_shape = tuple(old_shape[a] for a in axes)
        def getter(new_idx: Tuple[int, ...]) -> Value:
            old_idx = [0] * ndim
            for new_d, old_d in enumerate(axes):
                old_idx[old_d] = new_idx[new_d]
            return _index_get(self.data, tuple(old_idx))
        def build(shape: Tuple[int, ...], prefix: Tuple[int, ...] = ()) -> Nested:
            if not shape:
                return getter(prefix)
            return [build(shape[1:], prefix + (i,)) for i in range(shape[0])]
        new_nested = build(new_shape)
        return Tensor(new_nested)
    
    def sum_to_shape(self, target_shape: tuple) -> "Tensor | Value":
        """
        Reduce this tensor by summing broadcasted axes to match `target_shape`.

        NumPy-style, right-aligned rules:
          - Sum any leading axes in S that are not in T.
          - For aligned axes, if T[d] == 1 and S[d] > 1, sum over that axis.
          - Otherwise, keep the axis.

        Returns:
          - Tensor with shape == target_shape, or a scalar Value if target_shape == ().
        """
        import numpy as np

        # normalize list-backed storage to an object ndarray for reductions
        arr = np.array(self.data, dtype=object)
        T = tuple(target_shape)

        # Scalar target: sum all axes → Value
        if len(T) == 0:
            if arr.ndim == 0:
                return arr.item()  # already a scalar Value
            out = arr
            for ax in range(arr.ndim - 1, -1, -1):
                out = out.sum(axis=ax)
            return out.item() if hasattr(out, "item") else out

        # Compute axes to reduce according to broadcasting rules
        S = arr.shape
        axes_to_sum = []
        lead = len(S) - len(T)
        axes_to_sum.extend(range(lead))  # leading dims reduce
        for i in range(len(T)):          # aligned dims: reduce if T==1 and S>1
            Sd = S[lead + i]
            Td = T[i]
            if Td == 1 and Sd > 1:
                axes_to_sum.append(lead + i)

        out = arr
        for ax in sorted(axes_to_sum, reverse=True):
            out = out.sum(axis=ax)

        # reshape to target (np.sum collapses axes)
        out = out.reshape(T) if hasattr(out, "reshape") else out

        # Convert to nested lists of Value for Tensor()
        nested = out.tolist() if hasattr(out, "tolist") else out
        return Tensor(nested)
    
    # --- NumPy bridge -------------------------------------------------
    @staticmethod
    def from_numpy(arr, requires_grad: bool = True) -> "Tensor":
        """
        Create a Tensor from a NumPy array (or array-like).

        Each scalar becomes a Value:
          - requires_grad=True  -> leaf that participates in backprop
          - requires_grad=False -> constant (no gradients)

        This does not share storage: gradients do NOT flow back into
        the original numpy array; it's just a convenient constructor.
        """
        try:
            import numpy as np  # type: ignore[import]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Tensor.from_numpy requires NumPy to be installed"
            ) from exc

        arr = np.asarray(arr, dtype=float)

        def build(x):
            # x is a numpy scalar or ndarray
            if not hasattr(x, "shape") or x.shape == ():  # scalar
                # Use the existing helpers so semantics match rest of Tensor
                return _as_leaf(float(x)) if requires_grad else _as_const(float(x))
            # vector/matrix/higher-dim: recurse along first axis
            return [build(sub) for sub in x]

        nested = build(arr)
        return Tensor(nested)

    def to_numpy(self):
        """
        Convert this Tensor to a NumPy array of floats (no grad info).

        This walks the nested structure and pulls out Value.data,
        returning a regular ndarray.
        """
        try:
            import numpy as np  # type: ignore[import]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Tensor.to_numpy requires NumPy to be installed"
            ) from exc

        def extract(node):
            if isinstance(node, Value):
                return node.data
            if isinstance(node, list):
                return [extract(x) for x in node]
            # fall back to plain numbers if something slipped through
            return float(node)

        nested = extract(self.data)
        return np.array(nested, dtype=float)


    # alias
    T = property(lambda self: self.transpose())

    # --- convenience ---
    def zero_grad(self) -> None:
        for v in _flatten_refs(self.data):
            v.grad = 0.0
