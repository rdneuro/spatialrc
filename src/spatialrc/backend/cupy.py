"""
spatialrc.backend.cupy
=====================

CuPy backend: a near-drop-in GPU mirror of the NumPy reference (CuPy exposes a
NumPy-compatible API, so the loop body is line-for-line the CPU version with
``cp`` in place of ``np``).

Memory hygiene: releases the default memory pool via ``empty_cache`` and
periodically synchronises the stream inside the time loop.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Backend, validate_reservoir_inputs

_SYNC_EVERY = 256


class CupyBackend(Backend):
    name = "cupy"
    default_float = "float32"

    def __init__(self, device: int = 0) -> None:
        import cupy as cp

        self.cp = cp
        self._device_id = int(device)
        self._dtype = getattr(cp, self.default_float)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import cupy as cp

            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False

    @property
    def device(self) -> str:
        return f"cuda:{self._device_id}"

    def sync(self) -> None:
        self.cp.cuda.Stream.null.synchronize()

    def empty_cache(self) -> None:
        self.cp.get_default_memory_pool().free_all_blocks()

    # ----------------------------- plumbing --------------------------- #
    def asarray(self, x, dtype: Optional[str] = None):
        cp = self.cp
        dt = getattr(cp, dtype) if dtype else self._dtype
        return cp.asarray(x, dtype=dt)

    def to_numpy(self, x) -> np.ndarray:
        return self.cp.asnumpy(x)

    def _activations(self):
        cp = self.cp
        return {
            "linear": lambda x: x,
            "relu": lambda x: cp.maximum(0.0, x),
            "leaky_relu": lambda x: cp.where(x > 0, x, 0.5 * x),
            "sigmoid": lambda x: 1.0 / (1.0 + cp.exp(-x)),
            "tanh": cp.tanh,
            "elu": lambda x: cp.where(x > 0, x, 0.5 * (cp.exp(x) - 1.0)),
            "step": lambda x: cp.where(x >= 0.5, 1.0, 0.0),
        }

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
        cp = self.cp
        w_c = self.asarray(w)
        w_in_c = self.asarray(w_in)
        ext = np.asarray(ext_input, dtype=float)
        if ext.ndim == 1:
            ext = ext[:, None]
        ext_c = self.asarray(ext)
        delay_np = np.asarray(delay).astype(int)

        n = w_c.shape[0]
        t_steps = ext_c.shape[0]
        edges = np.asarray(w) != 0
        max_delay = int(delay_np[edges].max()) if edges.any() else 1
        buf_len = max_delay + 1

        lag = cp.asarray(delay_np - 1)
        src_cols = cp.broadcast_to(cp.arange(n)[:, None], (n, n))
        f = self._activations()[activation]

        hist = cp.zeros((buf_len, n), dtype=self._dtype)
        ptr = 0
        if ic is not None:
            hist[ptr] = self.asarray(ic)
        states = cp.zeros((t_steps, n), dtype=self._dtype)
        for t in range(t_steps):
            rows = (ptr - lag) % buf_len
            gathered = hist[rows, src_cols]
            recurrent = (w_c * gathered).sum(axis=0)
            pre = recurrent + ext_c[t] @ w_in_c
            if leak_rate is None:
                new_state = f(pre)
            else:
                new_state = (1.0 - leak_rate) * hist[ptr] + leak_rate * f(pre)
            states[t] = new_state
            ptr = (ptr + 1) % buf_len
            hist[ptr] = new_state
            if (t + 1) % _SYNC_EVERY == 0:
                self.sync()
        return states

    def eigvals(self, a):
        cp = self.cp
        # cupy.linalg.eigvals exists for general matrices in recent versions;
        # fall back to host NumPy if unavailable.
        try:
            return cp.linalg.eigvals(cp.asarray(a, dtype=cp.float64))
        except (AttributeError, NotImplementedError):
            return np.linalg.eigvals(np.asarray(a, dtype=float))

    def linear_readout_fit(self, phi, y, ridge):
        cp = self.cp
        phi_c = self.asarray(phi, dtype="float64")
        y_c = self.asarray(y, dtype="float64")
        p = phi_c.shape[1]
        gram = phi_c.T @ phi_c + ridge * cp.eye(p, dtype=cp.float64)
        rhs = phi_c.T @ y_c
        return cp.linalg.solve(gram, rhs)
