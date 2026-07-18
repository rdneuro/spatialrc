"""
spatialrc.delayed_reservoir
===========================

Delayed-gather Echo State Network.

This is the *conduction-delay* add-on: it replaces conn2res' instantaneous,
lag-1 recurrence

    x[t] = f( x[t-1] @ W + u[t] @ W_in )

with a per-edge delayed recurrence

    x[t] = f( sum_j W[j, i] * x_j[t - D[j, i]]  +  u[t] @ W_in )

where ``D`` is an integer conduction-delay matrix (see :mod:`spatialrc.geometry`).
With ``D == 1`` everywhere it reduces EXACTLY to a standard conn2res ESN.

The heavy per-timestep loop is delegated to a selectable compute backend
(:mod:`spatialrc.backend`), so the same class runs on NumPy, PyTorch (CPU/CUDA),
CuPy or JAX. Orchestration, validation and the conn2res-compatible
``simulate(...)`` signature live here.

CRITICAL CAVEAT (criticality)
-----------------------------
Adding delays changes the stability spectrum: conn2res' ``spectral_radius(W) ~
1`` criticality NO LONGER marks the edge of chaos. Use
:func:`spatialrc.criticality.companion_spectral_radius` and/or a memory-capacity
sweep to re-locate the operating point per ``D_max``. Never inherit conn2res'
alpha.

Washout
-------
The initial history is zero, so the first ``D_max`` states are contaminated by an
empty buffer. Discard a washout of at least ``D_max`` before fitting a readout; a
warning is emitted if you ask for fewer.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np

from .backend import Backend, get_backend
from .backend.base import ACTIVATION_NAMES


class DelayedEchoStateNetwork:
    """Echo State Network with heterogeneous per-edge conduction delays.

    Parameters
    ----------
    w : (N, N) numpy.ndarray
        Reservoir connectivity (source-row, target-col). Scale it BEFORE passing
        it in (EDR reweight -> spectral normalise -> global gain).
    delay : (N, N) int numpy.ndarray, optional
        Integer conduction delays (source -> target), ``>= 1``. If None, every
        edge uses lag 1 (plain conn2res-style ESN).
    activation_function : str, default 'tanh'
        One of {'linear','relu','leaky_relu','sigmoid','tanh','elu','step'}.
    leak_rate : float in (0, 1], optional
        Leaky-integrator constant. If None, no leak. NOTE: leak and delays both
        store linear memory, so tune them jointly.
    backend : str or Backend, default 'cpu'
        Compute backend name ({'auto','cpu','pytorch','cupy','jax'}) or an
        instantiated :class:`~spatialrc.backend.base.Backend`.
    """

    def __init__(
        self,
        w: np.ndarray,
        delay: Optional[np.ndarray] = None,
        activation_function: str = "tanh",
        leak_rate: Optional[float] = None,
        backend: Union[str, Backend] = "cpu",
    ) -> None:
        w = np.asarray(w, dtype=float)
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            raise ValueError(f"w must be square (N, N); got {w.shape}.")
        self.w = w
        self.n_nodes = w.shape[0]

        if delay is None:
            delay = np.ones_like(w, dtype=int)
        delay = np.asarray(delay).astype(int)
        if delay.shape != w.shape:
            raise ValueError("delay must have the same shape as w.")
        if delay.min() < 1:
            raise ValueError("all delays must be >= 1.")
        self.delay = delay

        edges = w != 0
        self.max_delay = int(delay[edges].max()) if edges.any() else 1

        if activation_function not in ACTIVATION_NAMES:
            raise ValueError(
                f"Unknown activation '{activation_function}'; "
                f"choose from {ACTIVATION_NAMES}."
            )
        self.activation_function = activation_function
        self.leak_rate = leak_rate
        self.backend: Backend = (
            backend if isinstance(backend, Backend) else get_backend(backend)
        )
        self._state: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def simulate(
        self,
        ext_input: Union[np.ndarray, list, tuple],
        w_in: np.ndarray,
        ic: Optional[np.ndarray] = None,
        output_nodes: Optional[np.ndarray] = None,
        return_states: bool = True,
        washout: Optional[int] = None,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Simulate delayed reservoir dynamics (conn2res-compatible signature).

        Parameters
        ----------
        ext_input : (time, N_inputs) array
        w_in : (N_inputs, N) array
        ic : (N,) array, optional
            Initial condition written to the most recent history slot.
        output_nodes : list/array, optional
            Subset of node indices to return.
        return_states : bool
        washout : int, optional
            Leading timesteps to drop. If None, nothing is dropped but a warning
            is issued when ``max_delay > 1`` (fitting a readout on contaminated
            states is a common silent bug).

        Returns
        -------
        states : (time - washout, N or len(output_nodes)) numpy array
        """
        ext_input = np.asarray(ext_input, dtype=float)
        if ext_input.ndim == 1:
            ext_input = ext_input[:, None]
        w_in = np.asarray(w_in, dtype=float)
        if w_in.ndim != 2 or w_in.shape[1] != self.n_nodes:
            raise ValueError(
                f"w_in must be (N_inputs, N={self.n_nodes}); got {w_in.shape}."
            )

        native_states = self.backend.run_delayed_reservoir(
            w=self.w,
            w_in=w_in,
            delay=self.delay,
            ext_input=ext_input,
            activation=self.activation_function,
            leak_rate=self.leak_rate,
            ic=ic,
        )
        states = self.backend.to_numpy(native_states)
        self._state = states

        washout = self._resolve_washout(washout)
        out = states[washout:]
        if not return_states:
            return None
        if output_nodes is not None:
            return out[:, output_nodes]
        return out

    # ------------------------------------------------------------------ #
    def _resolve_washout(self, washout: Optional[int]) -> int:
        if washout is None:
            if self.max_delay > 1:
                warnings.warn(
                    f"No washout requested but max_delay={self.max_delay}; the "
                    f"first {self.max_delay} states are contaminated by the "
                    f"zero-initialised history. Pass washout>=max_delay before "
                    f"fitting a readout.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return 0
        if washout < self.max_delay:
            warnings.warn(
                f"washout={washout} < max_delay={self.max_delay}; leading states "
                f"may be contaminated by the empty history buffer.",
                RuntimeWarning,
                stacklevel=3,
            )
        return int(washout)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_conn(
        cls,
        conn,
        geometry,
        velocity: float = 3.0,
        dt: float = 1.0,
        source: str = "length",
        edr_lambda: Optional[float] = None,
        alpha: float = 1.0,
        activation_function: str = "tanh",
        leak_rate: Optional[float] = None,
        backend: Union[str, Backend] = "cpu",
    ) -> "DelayedEchoStateNetwork":
        """Build a delayed reservoir from a conn2res ``Conn`` + a ``Geometry``.

        Applies, in the correct order: optional EDR reweighting -> spectral
        normalisation -> global gain ``alpha`` -> delay-matrix construction.

        Parameters
        ----------
        conn : object with a ``.w`` attribute (e.g. conn2res.connectivity.Conn).
        geometry : spatialrc.geometry.Geometry
        velocity, dt, source : passed to ``Geometry.delay_matrix``.
        edr_lambda : if not None, EDR-reweight before normalisation.
        alpha : global gain applied after spectral normalisation (the knob you
            sweep for criticality; but see the criticality caveat).
        backend : compute backend for simulation.
        """
        w = np.asarray(conn.w, dtype=float).copy()

        if edr_lambda is not None:
            w = geometry.edr_reweight(
                w, edr_lambda, dist=geometry.effective_distance(source)
            )

        w = _spectral_normalize(w) * float(alpha)
        delay = geometry.delay_matrix(w, velocity=velocity, dt=dt, source=source)

        return cls(
            w=w,
            delay=delay,
            activation_function=activation_function,
            leak_rate=leak_rate,
            backend=backend,
        )


def _spectral_normalize(w: np.ndarray) -> np.ndarray:
    """Divide by the largest absolute eigenvalue (spectral radius)."""
    ev = np.linalg.eigvals(w)
    r = np.abs(ev).max()
    return w if r == 0 else w / r
