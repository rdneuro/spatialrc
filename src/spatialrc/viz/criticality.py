"""
spatialrc.viz.criticality
=========================

Visualizations for the dynamical-regime and criticality diagnostics: the
memory-capacity curve, global-gain (alpha) sweeps, the alpha x D_max criticality
landscape, and the companion-matrix eigenspectrum relative to the unit circle
(the delayed analogue of the spectral-radius picture).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .style import CB_PALETTE, SEQUENTIAL_CMAP, publication_style


def _new_ax(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def plot_memory_capacity_curve(mc_per_lag: np.ndarray, ax=None):
    """MC(k) vs lag k, with the total MC annotated."""
    mc = np.asarray(mc_per_lag, dtype=float)
    lags = np.arange(1, len(mc) + 1)
    with publication_style():
        fig, ax = _new_ax(ax, (4.4, 3.0))
        ax.bar(lags, mc, color=CB_PALETTE["blue"], width=0.9, alpha=0.85)
        ax.set_xlabel("reconstruction lag k")
        ax.set_ylabel("MC(k) = corr²")
        ax.set_title(f"Memory capacity (total = {mc.sum():.2f})")
    return fig


def plot_alpha_sweep(
    alphas: Sequence[float],
    values,
    labels: Optional[Sequence[str]] = None,
    critical_alpha: Optional[float] = None,
    ylabel: str = "performance",
    ax=None,
):
    """Performance / memory capacity as a function of the global gain alpha.

    Parameters
    ----------
    alphas : (A,) sweep values.
    values : (A,) or (A, C) array; multiple columns are plotted as separate
        curves (e.g. different activation functions).
    labels : legend labels for the columns of ``values``.
    critical_alpha : if given, marks the estimated critical point with a line.
    """
    alphas = np.asarray(alphas, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    with publication_style():
        fig, ax = _new_ax(ax, (4.6, 3.2))
        for c in range(values.shape[1]):
            lab = labels[c] if labels is not None else None
            ax.plot(alphas, values[:, c], marker="o", ms=3, label=lab)
        if critical_alpha is not None:
            ax.axvline(critical_alpha, color=CB_PALETTE["vermillion"], ls="--",
                       lw=1.2, label=f"critical α≈{critical_alpha:.2g}")
        ax.set_xlabel("global gain α (spectral scaling)")
        ax.set_ylabel(ylabel)
        ax.set_title("Dynamical-regime sweep")
        if labels is not None or critical_alpha is not None:
            ax.legend()
    return fig


def plot_criticality_heatmap(
    alphas: Sequence[float],
    d_max_values: Sequence[float],
    grid: np.ndarray,
    metric: str = "memory capacity",
    ax=None,
):
    """alpha x D_max landscape of a scalar metric (e.g. memory capacity).

    Parameters
    ----------
    alphas : (A,) global-gain values (x-axis).
    d_max_values : (D,) maximum-delay values (y-axis).
    grid : (D, A) metric values.
    """
    grid = np.asarray(grid, dtype=float)
    with publication_style():
        fig, ax = _new_ax(ax, (4.8, 3.6))
        im = ax.imshow(
            grid, aspect="auto", origin="lower", cmap=SEQUENTIAL_CMAP,
            extent=[min(alphas), max(alphas), min(d_max_values), max(d_max_values)],
        )
        ax.set_xlabel("global gain α")
        ax.set_ylabel("max delay D_max (steps)")
        ax.set_title(f"{metric} landscape")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric)
    return fig


def plot_eigenspectrum(
    eigenvalues: np.ndarray,
    ax=None,
    title: str = "Companion eigenspectrum",
):
    """Scatter companion (or weight) eigenvalues in the complex plane.

    The unit circle is drawn for reference: mass crossing outside it marks the
    loss of the echo-state property (the delayed analogue of spectral radius 1).
    """
    ev = np.asarray(eigenvalues).ravel()
    rho = float(np.abs(ev).max())
    with publication_style():
        fig, ax = _new_ax(ax, (3.8, 3.6))
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color=CB_PALETTE["vermillion"],
                lw=1.2, label="unit circle")
        ax.scatter(ev.real, ev.imag, s=10, color=CB_PALETTE["blue"],
                   alpha=0.6, edgecolors="none")
        ax.axhline(0, color="0.7", lw=0.5)
        ax.axvline(0, color="0.7", lw=0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title(f"{title}  (ρ={rho:.3f})")
        ax.legend(loc="upper right")
    return fig
