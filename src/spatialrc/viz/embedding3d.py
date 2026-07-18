"""
spatialrc.viz.embedding3d
========================

3-D visualizations of the *spatial embedding*: parcel centroids in anatomical
space, coloured by a node-level scalar (degree, memory-capacity contribution,
seRNN module, ...), with optional edges rendered as tubes coloured by weight or
conduction delay.

Two rendering paths:

* **vedo** (preferred; VTK-based, the same engine family as the project's
  SpectralBrain/yabplot stack) -- used when ``vedo`` is importable. Renders
  offscreen so it works headless (HPC/CI). Set the offscreen flag BEFORE import.
* **matplotlib 3D** fallback -- always available, used when vedo is missing.

Both paths share the same signature and save a PNG.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .style import SEQUENTIAL_CMAP


def _vedo_available() -> bool:
    try:
        import vedo  # noqa: F401

        return True
    except Exception:
        return False


def plot_node_embedding_3d(
    coords: np.ndarray,
    node_values: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    delay: Optional[np.ndarray] = None,
    edge_color_by: str = "delay",
    out_path: str = "embedding3d.png",
    point_radius: float = 2.5,
    max_edges: int = 2000,
    cmap: str = SEQUENTIAL_CMAP,
    backend: str = "auto",
    title: str = "Spatial embedding",
):
    """Render parcel centroids (and optionally edges) in 3-D.

    Parameters
    ----------
    coords : (N, 3) node coordinates (mm).
    node_values : (N,) scalar per node for colouring (default: degree of ``w``).
    w : (N, N) connectome; if given, edges are drawn.
    delay : (N, N) delay matrix; used when ``edge_color_by='delay'``.
    edge_color_by : {'delay', 'weight'}.
    out_path : output PNG path.
    point_radius : node marker size.
    max_edges : cap on drawn edges (strongest by |weight| are kept).
    cmap : colormap name.
    backend : {'auto', 'vedo', 'matplotlib'}.

    Returns
    -------
    out_path : str
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3); got {coords.shape}.")
    n = coords.shape[0]

    if node_values is None:
        if w is not None:
            node_values = (np.asarray(w) != 0).sum(axis=1).astype(float)  # out-degree
        else:
            node_values = np.zeros(n)
    node_values = np.asarray(node_values, dtype=float).ravel()

    edges = _select_edges(w, delay, edge_color_by, max_edges) if w is not None else None

    use_vedo = backend == "vedo" or (backend == "auto" and _vedo_available())
    if use_vedo:
        return _render_vedo(coords, node_values, edges, out_path, point_radius,
                            cmap, title)
    return _render_matplotlib(coords, node_values, edges, out_path, point_radius,
                              cmap, title)


def _select_edges(w, delay, edge_color_by, max_edges):
    w = np.asarray(w, dtype=float)
    src, tgt = np.nonzero(w)
    weights = w[src, tgt]
    if edge_color_by == "delay" and delay is not None:
        scalar = np.asarray(delay)[src, tgt].astype(float)
    else:
        scalar = np.abs(weights)
        edge_color_by = "weight"
    # keep the strongest edges if capped
    if len(src) > max_edges:
        keep = np.argsort(-np.abs(weights))[:max_edges]
        src, tgt, scalar = src[keep], tgt[keep], scalar[keep]
    return {"src": src, "tgt": tgt, "scalar": scalar, "label": edge_color_by}


def _render_vedo(coords, node_values, edges, out_path, point_radius, cmap, title):
    os.environ.setdefault("VTK_USE_OFFSCREEN", "1")
    import vedo

    plt = vedo.Plotter(offscreen=True, size=(1000, 900))
    objects = []

    pts = vedo.Points(coords, r=point_radius * 3)
    pts.cmap(cmap, node_values).add_scalarbar(title="node value")
    objects.append(pts)

    if edges is not None and len(edges["src"]) > 0:
        scalar = edges["scalar"]
        smin, smax = float(scalar.min()), float(scalar.max())
        lines = []
        for j, i, s in zip(edges["src"], edges["tgt"], scalar):
            ln = vedo.Line(coords[j], coords[i], lw=1)
            ln.cmap(cmap, [s, s], vmin=smin, vmax=smax)
            lines.append(ln)
        objects.extend(lines)

    plt.show(objects, title, axes=1, viewup="z")
    plt.screenshot(out_path, scale=2)
    plt.close()
    return out_path


def _render_matplotlib(coords, node_values, edges, out_path, point_radius,
                       cmap, title):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    if edges is not None and len(edges["src"]) > 0:
        from matplotlib import colormaps
        from matplotlib.colors import Normalize

        scalar = edges["scalar"]
        norm = Normalize(vmin=scalar.min(), vmax=scalar.max())
        colormap = colormaps[cmap]
        for j, i, s in zip(edges["src"], edges["tgt"], scalar):
            ax.plot(
                [coords[j, 0], coords[i, 0]],
                [coords[j, 1], coords[i, 1]],
                [coords[j, 2], coords[i, 2]],
                color=colormap(norm(s)), lw=0.4, alpha=0.4,
            )

    p = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=node_values, cmap=cmap, s=point_radius * 12,
        edgecolors="k", linewidths=0.2, depthshade=True,
    )
    fig.colorbar(p, ax=ax, fraction=0.03, pad=0.04, label="node value")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(title)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
