"""
spatialrc.backend.cpu
=====================

NumPy backend. This is the **numerical reference**: every other backend must
reproduce these results (up to floating point). It is always available.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Backend, validate_reservoir_inputs


def _make_activations():
    return {
        "linear": lambda x: x,
        "relu": lambda x: np.maximum(0.0, x),
        "leaky_relu": lambda x: np.where(x > 0, x, 0.5 * x),
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "tanh": np.tanh,
        "elu": lambda x: np.where(x > 0, x, 0.5 * (np.exp(x) - 1.0)),
        "step": lambda x: np.where(x >= 0.5, 1.0, 0.0),
    }


class CpuBackend(Backend):
    name = "cpu"
    default_float = "float64"

    def __init__(self) -> None:
        self._act = _make_activations()

    @classmethod
    def is_available(cls) -> bool:
        return True

    # ----------------------------- plumbing --------------------------- #
    def asarray(self, x, dtype: Optional[str] = None):
        return np.asarray(x, dtype=dtype or self.default_float)

    def to_numpy(self, x) -> np.ndarray:
        return np.asarray(x)

    # ----------------------------- kernels ---------------------------- #
    def run_delayed_reservoir(
        self,
        w,
        w_in,
        delay,
        ext_input,
        activation,
        leak_rate,
        ic=None,
    ):
        validate_reservoir_inputs(w, w_in, delay, ext_input, activation)
        w = np.asarray(w, dtype=self.default_float)
        w_in = np.asarray(w_in, dtype=self.default_float)
        delay = np.asarray(delay).astype(int)
        ext_input = np.asarray(ext_input, dtype=self.default_float)
        if ext_input.ndim == 1:
            ext_input = ext_input[:, None]

        n = w.shape[0]
        t_steps = ext_input.shape[0]
        edges = w != 0
        max_delay = int(delay[edges].max()) if edges.any() else 1
        buf_len = max_delay + 1

        f = self._act[activation]
        lag = delay - 1
        src_cols = np.broadcast_to(np.arange(n)[:, None], (n, n))

        hist = np.zeros((buf_len, n), dtype=self.default_float)
        ptr = 0
        if ic is not None:
            hist[ptr] = np.asarray(ic, dtype=self.default_float)

        states = np.zeros((t_steps, n), dtype=self.default_float)
        for t in range(t_steps):
            rows = (ptr - lag) % buf_len
            gathered = hist[rows, src_cols]          # G[j, i] = x_j[t - D[j,i]]
            recurrent = (w * gathered).sum(axis=0)   # sum over sources -> (N,)
            pre = recurrent + ext_input[t] @ w_in
            if leak_rate is None:
                new_state = f(pre)
            else:
                new_state = (1.0 - leak_rate) * hist[ptr] + leak_rate * f(pre)
            states[t] = new_state
            ptr = (ptr + 1) % buf_len
            hist[ptr] = new_state
        return states

    def eigvals(self, a):
        return np.linalg.eigvals(np.asarray(a, dtype=self.default_float))

    def linear_readout_fit(self, phi, y, ridge):
        phi = np.asarray(phi, dtype=self.default_float)
        y = np.asarray(y, dtype=self.default_float)
        p = phi.shape[1]
        gram = phi.T @ phi + ridge * np.eye(p, dtype=self.default_float)
        rhs = phi.T @ y
        # solve is more stable than explicit inverse
        return np.linalg.solve(gram, rhs)
