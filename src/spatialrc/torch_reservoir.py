"""
torch_reservoir.py
==================

PyTorch delayed reservoir with a sparse per-edge gather and an optional
*learnable* distance kernel.

Why a torch version in addition to the NumPy one:
    * GPU execution for whole-brain N and long sequences;
    * a differentiable distance kernel W_eff = W * exp(-lambda * Dist) whose
      ``lambda`` (or a Gaussian sigma) can be trained by backpropagating the
      readout loss -- a middle ground between a fixed reservoir and a fully
      trained seRNN;
    * sparse (COO) edge storage so only the actual connectome edges cost memory,
      which matters when the connectome is distance-thresholded and only a few
      long-delay edges survive.

The reservoir weights themselves are FIXED (registered as buffers, not
parameters) unless ``learn_kernel=True``, in which case only the scalar kernel
parameter is trainable -- the topology stays fixed, preserving the
reservoir-computing assumption.

NOTE -- relationship to ``spatialrc.backend.pytorch``:
    The pytorch *backend* is a generic, non-differentiable, DENSE mirror of the
    NumPy reference used to accelerate ``DelayedEchoStateNetwork`` on GPU. This
    class is different: it is a standalone, SPARSE (COO), optionally
    *differentiable* ``nn.Module`` for the learnable-distance-kernel use case.
    Use the backend for plain fast simulation; use ``TorchDelayReservoir`` when
    you want a trainable kernel or sparse edge storage.

Convention matches the rest of the package: ``w`` is (source-row, target-col);
``delay[j, i]`` is the integer delay from source j to target i (>= 1).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class TorchDelayReservoir(nn.Module):
    """Sparse, delayed, GPU-capable reservoir (fixed weights).

    Parameters
    ----------
    w : (N, N) array-like
        Connectome (source-row, target-col). Pre-scale as desired.
    delay : (N, N) int array-like, optional
        Integer conduction delays (>= 1). Defaults to all-ones (lag-1).
    dist : (N, N) array-like, optional
        Distance matrix (mm). Required only if ``learn_kernel=True``.
    activation : callable, default torch.tanh
    leak_rate : float in (0, 1], optional
    learn_kernel : bool
        If True, an effective weight ``W_eff = W * exp(-lambda * dist)`` is used
        with a trainable scalar ``lambda`` (softplus-parameterised to stay >= 0).
    dtype : torch dtype, default torch.float32
    """

    def __init__(
        self,
        w,
        delay=None,
        dist=None,
        activation=torch.tanh,
        leak_rate: Optional[float] = None,
        learn_kernel: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        w = np.asarray(w, dtype=float)
        n = w.shape[0]
        if w.shape[0] != w.shape[1]:
            raise ValueError("w must be square (N, N).")
        self.n_nodes = n
        self.activation = activation
        self.leak_rate = leak_rate
        self.learn_kernel = learn_kernel

        if delay is None:
            delay = np.ones_like(w, dtype=int)
        delay = np.asarray(delay).astype(int)
        if delay.min() < 1:
            raise ValueError("all delays must be >= 1.")

        edges = np.argwhere(w != 0)          # (E, 2): (source j, target i)
        if edges.size == 0:
            raise ValueError("connectome has no edges.")
        src = edges[:, 0]
        tgt = edges[:, 1]
        wvals = w[src, tgt]
        dvals = delay[src, tgt]
        self.max_delay = int(dvals.max())

        # register fixed edge tensors as buffers (move with .to(device))
        self.register_buffer("src", torch.as_tensor(src, dtype=torch.long))
        self.register_buffer("tgt", torch.as_tensor(tgt, dtype=torch.long))
        self.register_buffer("wvals", torch.as_tensor(wvals, dtype=dtype))
        self.register_buffer("dvals", torch.as_tensor(dvals, dtype=torch.long))

        if learn_kernel:
            if dist is None:
                raise ValueError("dist is required when learn_kernel=True.")
            dvals_mm = np.asarray(dist, dtype=float)[src, tgt]
            self.register_buffer("edge_dist", torch.as_tensor(dvals_mm, dtype=dtype))
            # softplus(raw_lambda) >= 0 ; init near a small decay
            self.raw_lambda = nn.Parameter(torch.tensor(-2.0, dtype=dtype))
        self._dtype = dtype

    # ------------------------------------------------------------------ #
    def _edge_weights(self) -> torch.Tensor:
        if not self.learn_kernel:
            return self.wvals
        lam = torch.nn.functional.softplus(self.raw_lambda)
        return self.wvals * torch.exp(-lam * self.edge_dist)

    def forward(
        self,
        ext_input: torch.Tensor,
        w_in: torch.Tensor,
        washout: int = 0,
    ) -> torch.Tensor:
        """Run the reservoir.

        Parameters
        ----------
        ext_input : (T, N_inputs) tensor
        w_in : (N_inputs, N) tensor
        washout : int
            Leading steps to drop. Should be >= max_delay.

        Returns
        -------
        states : (T - washout, N) tensor
        """
        device = self.wvals.device
        ext_input = ext_input.to(device=device, dtype=self._dtype)
        w_in = w_in.to(device=device, dtype=self._dtype)
        T = ext_input.shape[0]
        N = self.n_nodes
        L = self.max_delay + 1

        hist = torch.zeros(L, N, device=device, dtype=self._dtype)
        ptr = 0
        states = torch.empty(T, N, device=device, dtype=self._dtype)
        ew = self._edge_weights()
        lag = self.dvals - 1                        # (E,)
        src = self.src
        tgt = self.tgt
        leak = self.leak_rate

        for t in range(T):
            rows = (ptr - lag) % L                  # (E,) buffer rows per edge
            gathered = hist[rows, src]              # (E,) x_src[t - delay]
            contrib = ew * gathered                 # (E,)
            recurrent = torch.zeros(N, device=device, dtype=self._dtype)
            recurrent.index_add_(0, tgt, contrib)   # scatter-add into targets
            pre = recurrent + ext_input[t] @ w_in
            if leak is None:
                new_state = self.activation(pre)
            else:
                prev = hist[ptr]
                new_state = (1.0 - leak) * prev + leak * self.activation(pre)
            states[t] = new_state
            ptr = (ptr + 1) % L
            hist[ptr] = new_state

        return states[washout:]
