"""
spatialrc.backend.jax
====================

JAX backend.

JAX arrays are immutable and Python ``for`` loops do not fuse under ``jit``, so
the naive ring-buffer mutation used by the other backends is inappropriate here.
The correct (and much faster) architecture is a **functional scan**: the delayed
recurrence is expressed as a ``jax.lax.scan`` whose carry is
``(history_buffer, pointer)`` and whose per-step body writes with
``buffer.at[ptr].set(...)``. The whole loop then compiles to a single fused,
differentiable kernel.

By default this backend enables 64-bit precision (``jax_enable_x64``) so the
eigenvalue / linear-solve diagnostics match the CPU reference; the reservoir
loop itself runs in the configured ``default_float``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Backend, validate_reservoir_inputs


class JaxBackend(Backend):
    name = "jax"
    default_float = "float32"

    def __init__(self, enable_x64: bool = True) -> None:
        import jax

        if enable_x64:
            jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        self.jax = jax
        self.jnp = jnp
        self._dtype = getattr(jnp, self.default_float)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import jax  # noqa: F401

            return True
        except Exception:
            return False

    @property
    def device(self) -> str:
        try:
            return str(self.jax.devices()[0])
        except Exception:  # pragma: no cover
            return "jax"

    def sync(self) -> None:
        # block_until_ready is applied per-array; nothing global to do here.
        pass

    # ----------------------------- plumbing --------------------------- #
    def asarray(self, x, dtype: Optional[str] = None):
        dt = getattr(self.jnp, dtype) if dtype else self._dtype
        return self.jnp.asarray(x, dtype=dt)

    def to_numpy(self, x) -> np.ndarray:
        return np.asarray(x)

    def _activation(self, kind: str):
        jnp = self.jnp
        table = {
            "linear": lambda x: x,
            "relu": lambda x: jnp.maximum(0.0, x),
            "leaky_relu": lambda x: jnp.where(x > 0, x, 0.5 * x),
            "sigmoid": lambda x: 1.0 / (1.0 + jnp.exp(-x)),
            "tanh": jnp.tanh,
            "elu": lambda x: jnp.where(x > 0, x, 0.5 * (jnp.exp(x) - 1.0)),
            "step": lambda x: jnp.where(x >= 0.5, 1.0, 0.0),
        }
        return table[kind]

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
        jnp = self.jnp
        jax = self.jax

        w_j = self.asarray(w)
        w_in_j = self.asarray(w_in)
        ext = np.asarray(ext_input, dtype=float)
        if ext.ndim == 1:
            ext = ext[:, None]
        ext_j = self.asarray(ext)
        delay_np = np.asarray(delay).astype(int)

        n = int(w_j.shape[0])
        edges = np.asarray(w) != 0
        max_delay = int(delay_np[edges].max()) if edges.any() else 1
        buf_len = max_delay + 1

        lag = jnp.asarray(delay_np - 1)
        src_cols = jnp.broadcast_to(jnp.arange(n)[:, None], (n, n))
        f = self._activation(activation)
        leak = leak_rate

        def step(carry, u_t):
            hist, ptr = carry
            rows = (ptr - lag) % buf_len
            gathered = hist[rows, src_cols]           # (N, N)
            recurrent = (w_j * gathered).sum(axis=0)  # (N,)
            pre = recurrent + u_t @ w_in_j
            if leak is None:
                new_state = f(pre)
            else:
                new_state = (1.0 - leak) * hist[ptr] + leak * f(pre)
            ptr_new = (ptr + 1) % buf_len
            hist_new = hist.at[ptr_new].set(new_state)
            return (hist_new, ptr_new), new_state

        hist0 = jnp.zeros((buf_len, n), dtype=self._dtype)
        if ic is not None:
            hist0 = hist0.at[0].set(self.asarray(ic))
        ptr0 = jnp.asarray(0, dtype=jnp.int32)  # scalar carry (type-stable scan)
        (_, _), states = jax.lax.scan(step, (hist0, ptr0), ext_j)
        return states

    def eigvals(self, a):
        return self.jnp.linalg.eigvals(self.jnp.asarray(a, dtype=self.jnp.float64))

    def linear_readout_fit(self, phi, y, ridge):
        jnp = self.jnp
        phi_j = self.asarray(phi, dtype="float64")
        y_j = self.asarray(y, dtype="float64")
        p = phi_j.shape[1]
        gram = phi_j.T @ phi_j + ridge * jnp.eye(p, dtype=jnp.float64)
        rhs = phi_j.T @ y_j
        return jnp.linalg.solve(gram, rhs)
