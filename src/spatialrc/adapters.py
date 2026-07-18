"""
spatialrc.adapters
==================

Optional glue for the two host libraries. Everything here imports its host
lazily, so the core package works without either installed.

conn2res
--------
:class:`~spatialrc.delayed_reservoir.DelayedEchoStateNetwork` already mirrors
conn2res' ``EchoStateNetwork.simulate(ext_input, w_in, ...)`` signature, so it is
a drop-in replacement: build it with ``DelayedEchoStateNetwork.from_conn(conn,
geometry)`` and use it wherever you would use a conn2res reservoir. No adapter is
needed for that path; :func:`conn_to_geometry` just helps you attach coordinates
to an existing ``Conn``.

reservoirpy
-----------
:func:`make_reservoirpy_node` wraps the delayed step as a custom reservoirpy
``Node`` so it composes with ``>>`` and the rest of the reservoirpy graph. It
keeps the ring buffer inside the node's parameters.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .geometry import Geometry


def _numpy_activation(kind: str):
    table = {
        "linear": lambda x: x,
        "relu": lambda x: np.maximum(0.0, x),
        "leaky_relu": lambda x: np.where(x > 0, x, 0.5 * x),
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "tanh": np.tanh,
        "elu": lambda x: np.where(x > 0, x, 0.5 * (np.exp(x) - 1.0)),
        "step": lambda x: np.where(x >= 0.5, 1.0, 0.0),
    }
    if kind not in table:
        raise ValueError(f"Unknown activation '{kind}'.")
    return table[kind]


def conn_to_geometry(
    conn,
    coords: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    length: Optional[np.ndarray] = None,
) -> Geometry:
    """Attach geometry to a conn2res ``Conn`` (or anything with ``.w``).

    Provide at least one of coords / dist / length; the connectome stays in
    ``conn.w`` and the returned ``Geometry`` is consumed by the reservoirs.
    """
    n = int(np.asarray(conn.w).shape[0])
    for label, m in (("coords", coords), ("dist", dist), ("length", length)):
        if m is not None and np.asarray(m).shape[0] != n:
            raise ValueError(
                f"{label} has {np.asarray(m).shape[0]} rows, expected {n} "
                f"to match conn.w."
            )
    return Geometry(coords=coords, dist=dist, length=length)


def make_reservoirpy_node(
    w: np.ndarray,
    delay: Optional[np.ndarray] = None,
    w_in: Optional[np.ndarray] = None,
    activation_function: str = "tanh",
    leak_rate: Optional[float] = None,
    name: str = "delayed_reservoir",
):
    """Wrap the delayed reservoir as a reservoirpy custom ``Node``.

    Requires ``reservoirpy`` installed. The node keeps a ring buffer in its
    parameters and applies the same per-edge delayed update as
    :class:`~spatialrc.delayed_reservoir.DelayedEchoStateNetwork`.

    Parameters
    ----------
    w : (N, N) connectome (source-row, target-col), pre-scaled.
    delay : (N, N) int delays; None -> lag-1.
    w_in : (N_inputs, N) input matrix; if None, an identity mapping is assumed.
    activation_function, leak_rate : as in DelayedEchoStateNetwork.
    """
    try:
        from reservoirpy.node import Node
    except Exception as exc:  # pragma: no cover - host optional
        raise ImportError(
            "reservoirpy is required for make_reservoirpy_node()."
        ) from exc

    w = np.asarray(w, dtype=float)
    n = w.shape[0]
    if delay is None:
        delay = np.ones_like(w, dtype=int)
    delay = np.asarray(delay).astype(int)
    edges = w != 0
    max_delay = int(delay[edges].max()) if edges.any() else 1
    buf_len = max_delay + 1
    lag = delay - 1
    src_cols = np.broadcast_to(np.arange(n)[:, None], (n, n))
    f = _numpy_activation(activation_function)

    def forward(node, x):
        hist = node.get_param("hist")
        ptr = int(node.get_param("ptr"))
        rows = (ptr - lag) % buf_len
        gathered = hist[rows, src_cols]
        recurrent = (w * gathered).sum(axis=0)
        drive = np.ravel(x) @ (w_in if w_in is not None else np.eye(n))
        pre = recurrent + drive
        if leak_rate is None:
            new_state = f(pre)
        else:
            new_state = (1.0 - leak_rate) * hist[ptr] + leak_rate * f(pre)
        ptr = (ptr + 1) % buf_len
        hist[ptr] = new_state
        node.set_param("hist", hist)
        node.set_param("ptr", ptr)
        return new_state.reshape(1, -1)

    def initialize(node, x=None, **kwargs):
        node.set_output_dim(n)
        if x is not None:
            node.set_input_dim(np.asarray(x).shape[-1])
        node.set_param("hist", np.zeros((buf_len, n)))
        node.set_param("ptr", 0)

    return Node(forward=forward, initializer=initialize, name=name)
