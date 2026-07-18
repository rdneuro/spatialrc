"""
spatialrc.backend.base
======================

Abstract interface for compute backends.

The design goal is to keep the *scientific* code (reservoir orchestration,
geometry, criticality) backend-agnostic while pushing the three genuinely
expensive kernels behind a uniform interface:

    1. ``run_delayed_reservoir`` -- the per-timestep delayed-gather recurrence
       (the dominant cost of any experiment; O(T * N^2) or O(T * E)),
    2. ``eigvals``               -- eigenvalues of the companion / weight matrix
       (criticality diagnostics; O((N*D_max)^3)),
    3. ``linear_readout_fit``    -- ridge normal-equations solve for memory
       capacity / readouts (O(N^3 + N^2 T)).

Everything else (matrix assembly, EDR fitting) is cheap and stays in host NumPy.

Each concrete backend (cpu / pytorch / cupy / jax) subclasses ``Backend`` and is
responsible for staying numerically identical to the CPU reference. The registry
in ``spatialrc.backend.__init__`` handles lazy import, availability detection and
graceful fallback to CPU.

Conventions (shared by all backends)
------------------------------------
* Weight matrices ``w`` are (N, N), source-row / target-col.
* ``delay[j, i]`` is the integer conduction delay from source j to target i,
  ``>= 1``. ``delay == 1`` everywhere reproduces a standard lag-1 ESN.
* ``run_delayed_reservoir`` returns states of shape (T, N) as a *native* array
  of the backend; callers use ``to_numpy`` to bring them back.
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np


class Backend(abc.ABC):
    """Abstract compute backend."""

    name: str = "abstract"
    #: default floating dtype as a string ("float64" on CPU, "float32" on GPU)
    default_float: str = "float64"

    # ------------------------------------------------------------------ #
    # availability / lifecycle
    # ------------------------------------------------------------------ #
    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Return True if the backend's dependency is importable/usable."""

    @property
    def device(self) -> str:
        """Human-readable device description (e.g. 'cpu', 'cuda:0')."""
        return "cpu"

    def sync(self) -> None:
        """Block until queued device work is finished (no-op on CPU)."""

    def empty_cache(self) -> None:
        """Release cached device memory (no-op on CPU)."""

    # ------------------------------------------------------------------ #
    # array plumbing
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def asarray(self, x, dtype: Optional[str] = None):
        """Convert to a native array of this backend."""

    @abc.abstractmethod
    def to_numpy(self, x) -> np.ndarray:
        """Convert a native array back to a host NumPy array."""

    # ------------------------------------------------------------------ #
    # expensive kernels (the reason this abstraction exists)
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def run_delayed_reservoir(
        self,
        w: np.ndarray,
        w_in: np.ndarray,
        delay: np.ndarray,
        ext_input: np.ndarray,
        activation: str,
        leak_rate: Optional[float],
        ic: Optional[np.ndarray] = None,
    ):
        """Integrate the delayed-gather reservoir; return native (T, N) states.

        Parameters
        ----------
        w : (N, N) reservoir weights (source-row, target-col).
        w_in : (N_inputs, N) input weights.
        delay : (N, N) int delays (>= 1).
        ext_input : (T, N_inputs) drive.
        activation : one of {'linear','relu','leaky_relu','sigmoid','tanh',
            'elu','step'}.
        leak_rate : leaky-integrator constant in (0, 1] or None.
        ic : optional (N,) initial state written to the most recent history slot.
        """

    @abc.abstractmethod
    def eigvals(self, a: np.ndarray):
        """Eigenvalues of a square matrix (native array)."""

    @abc.abstractmethod
    def linear_readout_fit(self, phi: np.ndarray, y: np.ndarray, ridge: float):
        """Ridge normal-equations solve: return beta minimizing ||phi beta - y||.

        ``phi`` is (T, P), ``y`` is (T,) or (T, K); returns (P,) or (P, K).
        """

    # ------------------------------------------------------------------ #
    # small shared helpers (host-side, backend-independent)
    # ------------------------------------------------------------------ #
    def spectral_radius(self, w: np.ndarray) -> float:
        """Largest absolute eigenvalue of ``w`` (uses backend eigvals)."""
        ev = self.to_numpy(self.eigvals(np.asarray(w, dtype=float)))
        return float(np.abs(ev).max())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Backend '{self.name}' device={self.device!r}>"


# Activation registry used by every backend's reference loop. Backends that run
# on their own array type re-implement these with their native ufuncs, but the
# names and semantics are fixed here so results stay identical.
ACTIVATION_NAMES = (
    "linear",
    "relu",
    "leaky_relu",
    "sigmoid",
    "tanh",
    "elu",
    "step",
)


def validate_reservoir_inputs(
    w: np.ndarray,
    w_in: np.ndarray,
    delay: np.ndarray,
    ext_input: np.ndarray,
    activation: str,
) -> None:
    """Shared fail-loud validation for the delayed reservoir kernels."""
    w = np.asarray(w)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"w must be square (N, N); got {w.shape}.")
    n = w.shape[0]
    if np.asarray(delay).shape != (n, n):
        raise ValueError("delay must have the same shape as w.")
    if np.asarray(delay).min() < 1:
        raise ValueError("all delays must be >= 1.")
    if np.asarray(w_in).shape[1] != n:
        raise ValueError(
            f"w_in must be (N_inputs, N={n}); got {np.asarray(w_in).shape}."
        )
    if np.asarray(ext_input).ndim != 2:
        raise ValueError("ext_input must be 2-D (time, N_inputs).")
    if np.asarray(ext_input).shape[1] != np.asarray(w_in).shape[0]:
        raise ValueError("ext_input and w_in input dimensions disagree.")
    if activation not in ACTIVATION_NAMES:
        raise ValueError(
            f"Unknown activation '{activation}'; choose from {ACTIVATION_NAMES}."
        )
