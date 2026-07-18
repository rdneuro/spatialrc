"""
spatialrc.backend.pytorch
=========================

PyTorch backend (CPU or CUDA). Mirrors the CPU reference loop exactly (dense
gather, same activation semantics) but runs on the selected torch device.

GPU hygiene (following the project's standing rules)
----------------------------------------------------
* runs the reservoir loop under ``torch.no_grad()`` (this is inference, not
  training);
* uses ``torch.cuda.get_device_properties(0).total_memory`` for the memory probe
  (never the non-existent ``.total_mem``);
* periodically synchronises the CUDA stream inside the long time loop so a
  runaway kernel surfaces early rather than after thousands of steps;
* exposes ``empty_cache`` to release the caching allocator between experiments,
  which matters on ~8 GB cards.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Backend, validate_reservoir_inputs

#: how often (in timesteps) to synchronise the CUDA stream inside the loop
_SYNC_EVERY = 256


def _import_torch():
    import torch  # noqa: F401

    return torch


class TorchBackend(Backend):
    name = "pytorch"
    default_float = "float32"

    def __init__(self, device: Optional[str] = None) -> None:
        torch = _import_torch()
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._dtype = getattr(torch, self.default_float)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch  # noqa: F401

            return True
        except Exception:
            return False

    @property
    def device(self) -> str:
        return str(self._device)

    def sync(self) -> None:
        if self._device.type == "cuda":
            self.torch.cuda.synchronize(self._device)

    def empty_cache(self) -> None:
        if self._device.type == "cuda":
            self.torch.cuda.empty_cache()

    def total_memory_bytes(self) -> Optional[int]:
        """Total VRAM of the current CUDA device, or None on CPU."""
        if self._device.type == "cuda":
            idx = self._device.index or 0
            # NOTE: correct attribute is total_memory (not total_mem)
            return int(self.torch.cuda.get_device_properties(idx).total_memory)
        return None

    # ----------------------------- plumbing --------------------------- #
    def asarray(self, x, dtype: Optional[str] = None):
        torch = self.torch
        dt = getattr(torch, dtype) if dtype else self._dtype
        if isinstance(x, torch.Tensor):
            return x.to(device=self._device, dtype=dt)
        return torch.as_tensor(np.asarray(x), device=self._device, dtype=dt)

    def to_numpy(self, x) -> np.ndarray:
        torch = self.torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _activation(self, kind: str):
        torch = self.torch
        table = {
            "linear": lambda x: x,
            "relu": torch.relu,
            "leaky_relu": lambda x: torch.where(x > 0, x, 0.5 * x),
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "elu": lambda x: torch.where(x > 0, x, 0.5 * (torch.exp(x) - 1.0)),
            "step": lambda x: torch.where(
                x >= 0.5, torch.ones_like(x), torch.zeros_like(x)
            ),
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
        torch = self.torch
        dev, dt = self._device, self._dtype

        w_t = self.asarray(w)
        w_in_t = self.asarray(w_in)
        ext = np.asarray(ext_input, dtype=float)
        if ext.ndim == 1:
            ext = ext[:, None]
        ext_t = self.asarray(ext)
        delay_np = np.asarray(delay).astype(int)

        n = w_t.shape[0]
        t_steps = ext_t.shape[0]
        edges = np.asarray(w) != 0
        max_delay = int(delay_np[edges].max()) if edges.any() else 1
        buf_len = max_delay + 1

        lag = torch.as_tensor(delay_np - 1, device=dev, dtype=torch.long)
        src_cols = (
            torch.arange(n, device=dev).unsqueeze(1).expand(n, n)
        )  # (N, N) source index
        f = self._activation(activation)

        with torch.no_grad():
            hist = torch.zeros(buf_len, n, device=dev, dtype=dt)
            ptr = 0
            if ic is not None:
                hist[ptr] = self.asarray(ic)
            states = torch.empty(t_steps, n, device=dev, dtype=dt)
            for t in range(t_steps):
                rows = (ptr - lag) % buf_len
                gathered = hist[rows, src_cols]          # (N, N)
                recurrent = (w_t * gathered).sum(dim=0)  # (N,)
                pre = recurrent + ext_t[t] @ w_in_t
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
        torch = self.torch
        at = self.asarray(a, dtype="float64")  # eig in double for stability
        return torch.linalg.eigvals(at)

    def linear_readout_fit(self, phi, y, ridge):
        torch = self.torch
        phi_t = self.asarray(phi, dtype="float64")
        y_t = self.asarray(y, dtype="float64")
        p = phi_t.shape[1]
        eye = torch.eye(p, device=self._device, dtype=torch.float64)
        gram = phi_t.T @ phi_t + ridge * eye
        rhs = phi_t.T @ y_t
        return torch.linalg.solve(gram, rhs)
