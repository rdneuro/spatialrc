"""
spatialrc.viz.style
===================

Shared, publication-oriented styling for the visualization module.

Provides a colorblind-safe palette, a context manager to apply a consistent
matplotlib style without clobbering the user's global rcParams, and a couple of
small figure helpers. Nothing here is imported at package import time, so
matplotlib is only required when you actually plot.
"""

from __future__ import annotations

import contextlib

# Wong (2011) colorblind-safe qualitative palette
CB_PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
CB_CYCLE = [
    CB_PALETTE["blue"],
    CB_PALETTE["vermillion"],
    CB_PALETTE["green"],
    CB_PALETTE["orange"],
    CB_PALETTE["purple"],
    CB_PALETTE["sky"],
    CB_PALETTE["yellow"],
]

# Sequential / diverging maps chosen for perceptual uniformity
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

_PUBLICATION_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "image.cmap": SEQUENTIAL_CMAP,
}


@contextlib.contextmanager
def publication_style():
    """Context manager applying the publication rcParams locally.

    Example
    -------
    >>> with publication_style():
    ...     fig, ax = plt.subplots()
    ...     ax.plot(...)
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(_PUBLICATION_RC):
        yield


def savefig(fig, path, **kwargs):
    """Save a figure at publication resolution and close it."""
    import matplotlib.pyplot as plt

    fig.savefig(path, **kwargs)
    plt.close(fig)
    return path
