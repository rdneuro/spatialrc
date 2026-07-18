"""
spatialrc.criticality
=====================

Diagnostics to RE-LOCATE the operating point of a *delayed* reservoir, because
conn2res' ``spectral_radius(W) ~ 1`` criticality is invalidated once conduction
delays are introduced.

Two tools:

1. :func:`companion_spectral_radius` -- exact linear-stability boundary of the
   delayed recurrence around the origin, via the block-companion lift of the
   delayed linear map ``x[t] = sum_r A_r x[t-r]``. Its spectral radius crossing 1
   is the delayed analogue of ``spectral_radius(W) ~ 1``. The eigendecomposition
   (the O((N*D_max)^3) cost) runs on the selected backend.

2. :func:`memory_capacity` -- empirical short-term linear memory capacity, whose
   peak over a global-gain sweep is the practical operating point when the
   companion matrix is too large to diagonalise. The ridge readout solve runs on
   the selected backend.

Both address the central caveat: never inherit the non-delayed alpha.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np

from .backend import Backend, get_backend


def _resolve_backend(backend: Union[str, Backend]) -> Backend:
    return backend if isinstance(backend, Backend) else get_backend(backend)


def build_companion(
    w: np.ndarray,
    delay: np.ndarray,
    leak_rate: Optional[float] = None,
    activation_slope: float = 1.0,
) -> np.ndarray:
    """Assemble the block-companion matrix of the delayed linear map.

    The delayed linear recurrence ``x[t] = sum_{r=1..D_max} A_r x[t-r]`` (where
    ``(A_r)[i, j] = slope * w[j, i]`` for edges with delay ``r``) is lifted to a
    first-order system on the stacked state
    ``X[t] = [x[t], x[t-1], ..., x[t-D_max+1]]``.

    Parameters
    ----------
    w : (N, N) reservoir weights (source-row, target-col), already scaled.
    delay : (N, N) int delays (>= 1), source -> target.
    leak_rate : float in (0, 1], optional
        If set, includes the leaky-integrator retention ``(1-leak) I`` at lag 1
        and scales the recurrent contribution by ``leak``.
    activation_slope : float
        Slope of the activation at the origin (1.0 for tanh; 0.25 for sigmoid).

    Returns
    -------
    C : (N*D_max, N*D_max) companion matrix (host NumPy).
    """
    w = np.asarray(w, dtype=float)
    delay = np.asarray(delay).astype(int)
    n = w.shape[0]
    edges = w != 0
    d_max = int(delay[edges].max()) if edges.any() else 1
    slope = float(activation_slope)

    # A_r[i, j] = slope * w[j, i] for edges whose delay equals r.
    # Vectorised per delay level (<= d_max passes, each O(N^2)); avoids a
    # per-edge Python loop that is slow on whole-brain connectomes.
    a_blocks = [np.zeros((n, n)) for _ in range(d_max + 1)]  # index by r (1..)
    edge_mask = w != 0
    for r in range(1, d_max + 1):
        level = edge_mask & (delay == r)
        if level.any():
            a_blocks[r] = slope * (w * level).T  # transpose -> target-row/source-col

    if leak_rate is not None:
        a = float(leak_rate)
        for r in range(1, d_max + 1):
            a_blocks[r] *= a
        a_blocks[1] += (1.0 - a) * np.eye(n)

    dim = n * d_max
    companion = np.zeros((dim, dim))
    for r in range(1, d_max + 1):
        companion[0:n, (r - 1) * n : r * n] = a_blocks[r]
    if d_max > 1:
        companion[n:dim, 0 : (d_max - 1) * n] = np.eye((d_max - 1) * n)
    return companion


def companion_spectral_radius(
    w: np.ndarray,
    delay: np.ndarray,
    leak_rate: Optional[float] = None,
    activation_slope: float = 1.0,
    backend: Union[str, Backend] = "cpu",
) -> float:
    """Spectral radius of the delayed map's block-companion matrix.

    ``rho < 1`` implies the linearised delayed reservoir has the echo-state
    property; ``rho`` near 1 is the delayed analogue of criticality. Cost is
    O((N*D_max)^3); pushed to the selected backend.
    """
    be = _resolve_backend(backend)
    companion = build_companion(w, delay, leak_rate, activation_slope)
    ev = be.to_numpy(be.eigvals(companion))
    return float(np.abs(ev).max())


def memory_capacity(
    sim_fn: Callable[[np.ndarray], np.ndarray],
    n_steps: int = 2000,
    max_delay: int = 50,
    washout: int = 200,
    ridge: float = 1e-6,
    seed: int = 0,
    backend: Union[str, Backend] = "cpu",
) -> Tuple[float, np.ndarray]:
    """Empirical short-term linear memory capacity (Jaeger).

    Drives the reservoir with i.i.d. uniform scalar input and, for each lag
    ``k``, fits a linear readout to reconstruct ``u[t-k]`` from the reservoir
    state at ``t``; MC(k) is the squared correlation, and the total MC is their
    sum.

    Parameters
    ----------
    sim_fn : callable
        ``sim_fn(u) -> states`` mapping an (T, 1) input to (T, N) states (one
        state per input step; apply washout via this function's ``washout``).
    n_steps : length of the driving signal.
    max_delay : maximum reconstruction lag k.
    washout : leading steps dropped before fitting.
    ridge : ridge penalty for the linear readout.
    seed : RNG seed.
    backend : compute backend for the readout solve.

    Returns
    -------
    total_mc : float
    mc_per_lag : (max_delay,) array with MC(k) for k = 1..max_delay.
    """
    if washout < max_delay:
        raise ValueError(
            f"washout ({washout}) must be >= max_delay ({max_delay}); otherwise "
            f"the lag-aligned target slice would be ill-defined."
        )
    be = _resolve_backend(backend)
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=(n_steps, 1))
    states = np.asarray(sim_fn(u), dtype=float)

    t_len = min(len(states), len(u))
    states = states[:t_len]
    u = u[:t_len, 0]

    xs = states[washout:]
    phi = np.hstack([xs, np.ones((xs.shape[0], 1))])  # design + bias

    mc = np.zeros(max_delay)
    for k in range(1, max_delay + 1):
        target = u[washout - k : t_len - k]
        m = min(len(target), phi.shape[0])
        phi_k = phi[-m:]
        tgt_k = target[-m:]
        beta = be.to_numpy(be.linear_readout_fit(phi_k, tgt_k, ridge))
        pred = phi_k @ beta
        var = np.var(tgt_k)
        if var < 1e-12:
            mc[k - 1] = 0.0
        else:
            c = np.corrcoef(pred, tgt_k)[0, 1]
            mc[k - 1] = 0.0 if np.isnan(c) else c ** 2
    return float(mc.sum()), mc
