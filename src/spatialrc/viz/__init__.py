"""
spatialrc.viz
=============

Visualization module matched to the library's outputs.

2-D (matplotlib / seaborn):
    * connectome, distance and delay matrices              -> viz.matrices
    * weight-vs-distance with EDR overlay                  -> viz.matrices
    * reservoir state time-series                          -> viz.matrices
    * memory-capacity curve, alpha sweep, alpha x D_max
      criticality landscape, companion eigenspectrum       -> viz.criticality

3-D (vedo offscreen, matplotlib fallback):
    * spatial node embedding with delay/weight-coloured edges -> viz.embedding3d

Matplotlib/vedo are imported lazily inside each function, so importing this
subpackage does not require any plotting backend.
"""

from __future__ import annotations

from .criticality import (
    plot_alpha_sweep,
    plot_criticality_heatmap,
    plot_eigenspectrum,
    plot_memory_capacity_curve,
)
from .embedding3d import plot_node_embedding_3d
from .matrices import (
    plot_connectome,
    plot_delay_matrix,
    plot_distance_matrix,
    plot_reservoir_states,
    plot_weight_distance,
)
from .style import CB_PALETTE, publication_style, savefig

__all__ = [
    # matrices
    "plot_connectome",
    "plot_distance_matrix",
    "plot_delay_matrix",
    "plot_weight_distance",
    "plot_reservoir_states",
    # criticality
    "plot_memory_capacity_curve",
    "plot_alpha_sweep",
    "plot_criticality_heatmap",
    "plot_eigenspectrum",
    # 3d
    "plot_node_embedding_3d",
    # style
    "publication_style",
    "savefig",
    "CB_PALETTE",
]
