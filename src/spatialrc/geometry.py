"""
geometry.py
===========

Spatial-embedding layer for connectome reservoir computing.

This module fills the gap left by conn2res' ``Conn`` class (which stores only a
topology matrix ``w`` and has no coordinates or distances) by providing a
``Geometry`` object that carries node coordinates and inter-regional distance /
streamline-length matrices, and derives from them:

    1. an Exponential-Distance-Rule (EDR) reweighting of the connectome,
    2. an integer conduction-delay matrix,

both of which the reservoir engines in this package consume.

Distance conventions
--------------------
* ``dist`` / ``length`` matrices are (N, N), in millimetres, row = source,
  column = target (matching conn2res' ``w`` convention where ``state @ w`` sends
  activation from source rows to target columns).
* The diagonal is meaningless (self-distance) and is ignored.
* Symmetric dMRI matrices and asymmetric tract-tracing matrices are both
  accepted; symmetry is never assumed.

IMPORTANT ORDER-OF-OPERATIONS CAVEAT
------------------------------------
EDR reweighting changes the spectrum of ``w``. Always reweight FIRST and let the
reservoir spectral-normalisation run AFTERWARDS, never the reverse, otherwise the
spectral radius you set is silently destroyed by the reweighting.

All docstrings/comments are in English by project convention.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist


class Geometry:
    """Container and toolbox for the spatial embedding of a connectome.

    Parameters
    ----------
    coords : (N, 3) numpy.ndarray, optional
        Region centroid coordinates (e.g. RAS from a parcellation).
    dist : (N, N) numpy.ndarray, optional
        Precomputed distance matrix (mm). Typically geodesic-over-surface or
        Euclidean. Used when no streamline length is available (e.g. T1w-only
        cohorts).
    length : (N, N) numpy.ndarray, optional
        Streamline / fiber-tract length matrix (mm), e.g. from
        ``tck2connectome ... -scale_length -stat_edge mean``. Preferred source
        for conduction delays when available (diffusion-MRI cohorts).
    """

    def __init__(
        self,
        coords: Optional[np.ndarray] = None,
        dist: Optional[np.ndarray] = None,
        length: Optional[np.ndarray] = None,
    ) -> None:
        self.coords = None if coords is None else np.asarray(coords, dtype=float)
        self.dist = None if dist is None else self._sanitize(dist)
        self.length = None if length is None else self._sanitize(length)

        if self.coords is not None and self.dist is None:
            # auto-fill a Euclidean distance matrix from coordinates
            self.dist = self.euclidean_distances()

        self.n_nodes = self._infer_n_nodes()

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_coordinates(cls, coords: np.ndarray) -> "Geometry":
        """Build a Geometry from region centroids alone (Euclidean distances)."""
        return cls(coords=coords)

    def euclidean_distances(self) -> np.ndarray:
        """Euclidean distance matrix (mm) between region centroids."""
        if self.coords is None:
            raise ValueError("coords is required to compute Euclidean distances.")
        d = cdist(self.coords, self.coords, metric="euclidean")
        np.fill_diagonal(d, 0.0)
        return d

    def set_geodesic(self, dist: np.ndarray) -> "Geometry":
        """Attach an externally computed geodesic-over-surface distance matrix.

        Compute it with e.g. ``potpourri3d`` (heat method) or ``gdist``/
        ``pygeodesic`` (exact), reducing the per-vertex field to a per-parcel
        matrix via the centroid-vertex or mean-of-pairs recipe, then pass it in.
        """
        self.dist = self._sanitize(dist)
        self.n_nodes = self._infer_n_nodes()
        return self

    def set_streamline_lengths(self, length: np.ndarray) -> "Geometry":
        """Attach a diffusion-MRI streamline length matrix (mm)."""
        self.length = self._sanitize(length)
        self.n_nodes = self._infer_n_nodes()
        return self

    # ------------------------------------------------------------------ #
    # the effective distance used downstream
    # ------------------------------------------------------------------ #
    def effective_distance(self, prefer: str = "length") -> np.ndarray:
        """Return the distance matrix used for delays / EDR.

        Parameters
        ----------
        prefer : {'length', 'dist', 'euclidean'}
            Preference order. 'length' uses streamline lengths if present and
            falls back to ``dist``; 'dist' forces the geodesic/Euclidean matrix;
            'euclidean' recomputes from coords.
        """
        if prefer == "euclidean":
            return self.euclidean_distances()
        if prefer == "length":
            if self.length is not None:
                return self.length
            if self.dist is not None:
                return self.dist
            raise ValueError("No length or dist matrix available.")
        if prefer == "dist":
            if self.dist is not None:
                return self.dist
            raise ValueError("No dist matrix available.")
        raise ValueError(f"Unknown prefer='{prefer}'.")

    # ------------------------------------------------------------------ #
    # Exponential Distance Rule (EDR)
    # ------------------------------------------------------------------ #
    def fit_edr_lambda(
        self,
        w: np.ndarray,
        dist: Optional[np.ndarray] = None,
        robust: bool = True,
    ) -> float:
        """Fit the EDR decay constant lambda (mm^-1) from the data.

        Fits ``log(w_ij) = a - lambda * d_ij`` over existing edges only.

        NOTE (caveat): plain log-linear OLS is biased by heteroscedasticity and
        by the exclusion of zero-weight edges; the value returned here is a
        practical estimate, not a substitute for a proper binomial/GLM fit of
        connection *probability* vs distance (Ercsey-Ravasz et al. 2013). lambda
        is also strongly brain-size dependent (macaque ~0.19, mouse ~0.78 mm^-1),
        so ALWAYS fit on your own cohort rather than importing a literature value.

        Parameters
        ----------
        w : (N, N) connectome (weights).
        dist : (N, N) distance matrix; defaults to ``effective_distance('dist')``.
        robust : if True, uses a median-based (Theil-Sen-like) slope that is less
            sensitive to outliers than OLS.

        Returns
        -------
        lam : float
            Estimated decay constant (mm^-1), guaranteed >= 0.
        """
        w = np.asarray(w, dtype=float)
        if dist is None:
            dist = self.effective_distance("dist")
        mask = (w > 0) & np.isfinite(w) & (dist > 0) & np.isfinite(dist)
        if mask.sum() < 3:
            raise ValueError("Not enough positive edges to fit EDR lambda.")
        x = dist[mask]
        y = np.log(w[mask])
        if robust:
            # Theil-Sen slope on a random subsample of pairs (bounded cost)
            slope = _theil_sen_slope(x, y, max_pairs=200_000)
        else:
            slope = np.polyfit(x, y, deg=1)[0]
        lam = float(max(0.0, -slope))
        return lam

    def edr_reweight(
        self,
        w: np.ndarray,
        lam: float,
        dist: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Multiply existing weights by exp(-lambda * distance).

        Returns a NEW matrix; the caller is responsible for spectral
        normalisation AFTERWARDS (see module docstring order-of-operations
        caveat). Sign is preserved (the kernel is strictly positive), so this is
        safe for Dale-constrained matrices at the level of individual edges, but
        you should re-check global E/I balance afterwards because long/short
        edges may be unevenly excitatory/inhibitory.
        """
        w = np.asarray(w, dtype=float).copy()
        if dist is None:
            dist = self.effective_distance("dist")
        kernel = np.exp(-float(lam) * dist)
        out = w * kernel
        # do not resurrect non-edges
        out[w == 0] = 0.0
        np.fill_diagonal(out, 0.0)
        return out

    # ------------------------------------------------------------------ #
    # conduction delays
    # ------------------------------------------------------------------ #
    def delay_matrix(
        self,
        w: np.ndarray,
        velocity: float = 3.0,
        dt: float = 1.0,
        source: str = "length",
        min_delay: int = 1,
        verbose: bool = True,
    ) -> np.ndarray:
        """Integer conduction-delay matrix D (steps), source -> target.

        ``delay_ms = distance_mm / velocity`` ; ``D = round(delay_ms / dt)``.

        Parameters
        ----------
        w : (N, N) connectome; delays are only meaningful on existing edges, and
            ``D_max`` is computed over edges (``w != 0``) only.
        velocity : float
            Conduction speed in mm/ms. The Virtual Brain library default is 3.0
            mm/ms (its tutorials often use 4.0). Physiological axon velocities
            span ~0.3-120 m/s across calibers; there is NO single correct value,
            so sweep this as a sensitivity axis.
        dt : float
            Simulation timestep in ms. Choose it so the median edge delay spans
            several integer steps; if every delay rounds to <=1 there is no
            spatial delay structure at all.
        source : {'length', 'dist', 'euclidean'}
            Which distance matrix to convert (see ``effective_distance``).
        min_delay : int
            Floor applied to on-edge delays (>=1 reproduces conn2res' native
            lag-1 recurrence as a special case).
        verbose : bool
            Print D_max and the fraction of edges collapsed to ``min_delay``.

        Returns
        -------
        D : (N, N) int array. Off-edge entries are set to ``min_delay`` but never
            used (their weight is zero).
        """
        w = np.asarray(w, dtype=float)
        d = self.effective_distance(source)
        with np.errstate(divide="ignore", invalid="ignore"):
            delay_ms = d / float(velocity)
        steps = np.rint(delay_ms / float(dt))
        steps[~np.isfinite(steps)] = min_delay
        D = np.clip(steps, min_delay, None).astype(int)
        np.fill_diagonal(D, min_delay)
        edges = w != 0
        if verbose:
            if edges.any():
                d_max = int(D[edges].max())
                collapsed = float(np.mean(D[edges] <= min_delay))
                print(
                    f"[delay_matrix] D_max={d_max} steps over edges; "
                    f"{100 * collapsed:.1f}% of edges at min_delay={min_delay}. "
                    f"(velocity={velocity} mm/ms, dt={dt} ms)"
                )
                if d_max <= min_delay:
                    warnings.warn(
                        "All edge delays collapsed to min_delay: no delay "
                        "structure. Decrease dt or velocity.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                warnings.warn("Connectome has no edges.", RuntimeWarning,
                              stacklevel=2)
        # off-edge delays are irrelevant; set to min_delay to keep D_max honest
        D[~edges] = min_delay
        return D

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sanitize(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=float).copy()
        m[~np.isfinite(m)] = 0.0
        np.fill_diagonal(m, 0.0)
        return m

    def _infer_n_nodes(self) -> Optional[int]:
        for m in (self.dist, self.length):
            if m is not None:
                return int(m.shape[0])
        if self.coords is not None:
            return int(self.coords.shape[0])
        return None


def _theil_sen_slope(x: np.ndarray, y: np.ndarray, max_pairs: int = 200_000) -> float:
    """Median pairwise slope (robust). Subsamples pairs if there are too many."""
    n = len(x)
    n_all_pairs = n * (n - 1) // 2
    rng = np.random.default_rng(0)
    if n_all_pairs <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
    dx = x[j] - x[i]
    ok = dx != 0
    slopes = (y[j][ok] - y[i][ok]) / dx[ok]
    return float(np.median(slopes))
