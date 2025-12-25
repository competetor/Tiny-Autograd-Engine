# autograd/utils.py
from __future__ import annotations

from typing import Set, Tuple
from types import SimpleNamespace

# graphviz is optional; tests only need to read .source
try:
    import graphviz as _gv  # type: ignore
except Exception:
    _gv = None


def draw_graph(
    root,
    *,
    render: bool = False,
    filename: str = "graph",
    directory: str | None = None,
    format: str = "png",
):
    """
    Produce Graphviz DOT for the computation graph (micrograd style) and return an
    object with a `.source` attribute containing the DOT text.

    - If the `graphviz` Python package is available, returns a `graphviz.Source`
      (so you can call `.render()` if you like).
    - If it's not available, returns a tiny stub with `.source` and a no-op
      `render()` so tests can still assert on the DOT.
    - Layout:
        * Value nodes are 'record' boxes: { label | data ... | grad ... }
        * Op nodes are small circles labeled with the op string (e.g., "tanh")
        * Edges route child -> op_node -> value
    - Works even after `backward()` if your Value keeps `_viz_prev` snapshots
      (see Value.backward minimal tweak).
    """

    # --- helpers -------------------------------------------------------------

    def parents_of(v):
        # Prefer live edges; if freed by backward(), fall back to viz snapshot.
        return getattr(v, "_prev", ()) or getattr(v, "_viz_prev", ()) or ()

    # Collect nodes/edges reachable from root
    nodes: Set[object] = set()
    edges: Set[Tuple[object, object]] = set()

    def build(v):
        if v in nodes:
            return
        nodes.add(v)
        for p in parents_of(v):
            edges.add((p, v))
            build(p)

    build(root)

    # Stable topological order: parents before children
    topo = []
    seen = set()

    def topo_build(v):
        if v in seen:
            return
        seen.add(v)
        for p in parents_of(v):
            topo_build(p)
        topo.append(v)

    topo_build(root)

    # Deterministic ids based on topo index
    idmap = {v: f"v{idx}" for idx, v in enumerate(topo)}

    # --- build DOT text ------------------------------------------------------

    def _fmt_num(x):
        try:
            return f"{float(x):.6g}"
        except Exception:
            return str(x)

    lines = []
    lines.append("digraph G {")
    lines.append("  rankdir=LR;")
    # default style for value nodes
    lines.append("  node [shape=record, fontsize=12];")

    for v in topo:
        uid = idmap[v]
        label = getattr(v, "label", "") or ""
        data = getattr(v, "data", None)
        grad = getattr(v, "grad", None)
        op = getattr(v, "_op", "") or ""

        parts = [label] if label else []
        if data is not None:
            parts.append(f"data {_fmt_num(data)}")
        if grad is not None:
            parts.append(f"grad {_fmt_num(grad)}")
        rec = " | ".join(parts) if parts else "Value"

        # value node (record)  ✅ quote uid
        lines.append(f'  "{uid}" [label="{{ {rec} }}"];')

        # optional op node (small circle), then op -> value
        if op:
            op_id = f"{uid}_{op}"
            lines.append("  node [shape=circle, fontsize=10, width=0.3, height=0.3];")            
            lines.append(f'  "{op_id}" [label="{op}"];')
            lines.append("  node [shape=record, fontsize=12];")  # restore defaults
            lines.append(f'  "{op_id}" -> "{uid}";')

    # edges: child -> op-node (if exists) else -> value
    for child, parent in edges:
        cu, pu = idmap[child], idmap[parent]
        op = getattr(parent, "_op", "") or ""
        if op:
            op_id = f"{pu}_{op}"
            # ✅ quote IDs here too
            lines.append(f'  "{cu}" -> "{op_id}";')
        else:
            lines.append(f'  "{cu}" -> "{pu}";')

    lines.append("}")
    dot_source = "\n".join(lines)

    if _gv is not None:
        dot = _gv.Source(dot_source, filename=filename, directory=directory, format=format)
        if render:
            dot.render(filename=filename, directory=directory, cleanup=True)
        return dot
    else:
        return SimpleNamespace(source=dot_source, render=lambda **_: None)
