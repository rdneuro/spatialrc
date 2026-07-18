"""
spatialrc.viz.matrices
======================

2-D visualizations matched to the library's core outputs: the connectome, the
distance and delay matrices produced by :class:`~spatialrc.geometry.Geometry`,
the weight-vs-distance relationship (with the fitted EDR overlay), and the
reservoir state time-series produced by the delayed reservoir.

All functions return a matplotlib ``Figure`` and accept an optional ``ax`` so
they compose into multi-panel layouts.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .style import CB_PALETTE, DIVERGING_CMAP, SEQUENTIAL_CMAP, publication_style


def _new_ax(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def plot_connectome(
    w: np.ndarray,
    ax=None,
    title: str = "Connectome (source -> target)",
    symmetric_scale: bool = True,
):
    """Heatmap of the connectivity matrix ``w`` (rows = source, cols = target)."""
    w = np.asarray(w, dtype=float)
    with publication_style():
        fig, ax = _new_ax(ax, (4.2, 3.6))
        if symmetric_scale and np.any(w < 0):
            vmax = np.abs(w).max()
            im = ax.imshow(w, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(w, cmap=SEQUENTIAL_CMAP)
        ax.set_title(title)
        ax.set_xlabel("target node")
        ax.set_ylabel("source node")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight")
    return fig


def plot_distance_matrix(dist: np.ndarray, ax=None, unit: str = "mm"):
    """Heatmap of an inter-regional distance / streamline-length matrix."""
    dist = np.asarray(dist, dtype=float)
    with publication_style():
        fig, ax = _new_ax(ax, (4.2, 3.6))
        im = ax.imshow(dist, cmap="magma")
        ax.set_title("Inter-regional distance")
        ax.set_xlabel("node")
        ax.set_ylabel("node")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"distance ({unit})")
    return fig


def plot_delay_matrix(delay: np.ndarray, w: Optional[np.ndarray] = None, ax=None):
    """Heatmap of the integer conduction-delay matrix (steps).

    If ``w`` is given, off-edge entries are masked so only real edges are shown.
    """
    delay = np.asarray(delay).astype(float)
    shown = delay.copy()
    if w is not None:
        shown[np.asarray(w) == 0] = np.nan
    with publication_style():
        fig, ax = _new_ax(ax, (4.2, 3.6))
        im = ax.imshow(shown, cmap="cividis")
        ax.set_title("Conduction delays")
        ax.set_xlabel("target node")
        ax.set_ylabel("source node")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="delay (steps)")
    return fig


def plot_weight_distance(
    w: np.ndarray,
    dist: np.ndarray,
    lam: Optional[float] = None,
    ax=None,
    logy: bool = True,
):
    """Scatter of edge weight vs distance with an optional EDR fit overlay.

    Parameters
    ----------
    w, dist : (N, N) matrices.
    lam : float, optional
        If given, overlays the fitted exponential-distance-rule decay
        ``w ~ exp(-lam * d)`` (normalised to the data's geometric mean).
    logy : bool
        Log-scale the weight axis (natural for the heavy-tailed EDR).
    """
    w = np.asarray(w, dtype=float)
    dist = np.asarray(dist, dtype=float)
    mask = (w > 0) & (dist > 0) & np.isfinite(w) & np.isfinite(dist)
    x = dist[mask]
    y = w[mask]
    with publication_style():
        fig, ax = _new_ax(ax, (4.4, 3.4))
        ax.scatter(x, y, s=6, alpha=0.3, color=CB_PALETTE["blue"],
                   edgecolors="none", rasterized=True)
        if lam is not None and len(x) > 0:
            xs = np.linspace(x.min(), x.max(), 100)
            scale = np.exp(np.mean(np.log(y)) + lam * np.mean(x))
            ax.plot(xs, scale * np.exp(-lam * xs), color=CB_PALETTE["vermillion"],
                    lw=2, label=f"EDR fit λ={lam:.3g} mm⁻¹")
            ax.legend()
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("distance (mm)")
        ax.set_ylabel("edge weight")
        ax.set_title("Weight–distance relationship")
    return fig


def plot_reservoir_states(
    states: np.ndarray,
    max_nodes: int = 40,
    mode: str = "heatmap",
    ax=None,
):
    """Visualize reservoir state time-series.

    Parameters
    ----------
    states : (T, N) array from ``DelayedEchoStateNetwork.simulate``.
    max_nodes : cap on the number of nodes drawn (subsampled if exceeded).
    mode : {'heatmap', 'traces'}.
    """
    states = np.asarray(states, dtype=float)
    t_len, n = states.shape
    idx = np.arange(n)
    if n > max_nodes:
        idx = np.linspace(0, n - 1, max_nodes).astype(int)
    with publication_style():
        if mode == "heatmap":
            fig, ax = _new_ax(ax, (5.2, 3.2))
            vmax = np.abs(states).max()
            im = ax.imshow(
                states[:, idx].T, aspect="auto", cmap=DIVERGING_CMAP,
                vmin=-vmax, vmax=vmax, interpolation="nearest",
            )
            ax.set_xlabel("time step")
            ax.set_ylabel("node")
            ax.set_title("Reservoir states")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="activation")
        elif mode == "traces":
            fig, ax = _new_ax(ax, (5.2, 3.2))
            for k in idx:
                ax.plot(states[:, k], lw=0.6, alpha=0.6)
            ax.set_xlabel("time step")
            ax.set_ylabel("activation")
            ax.set_title("Reservoir states")
        else:
            raise ValueError("mode must be 'heatmap' or 'traces'.")
    return fig
