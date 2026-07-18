"""
spatialrc
=========

A base-agnostic add-on for connectome reservoir computing that adds three
spatial-embedding capabilities plus multi-backend execution and visualization:

    [1] spatial constraints / EDR reweighting          -> geometry.Geometry
    [2] conduction delays proportional to distance      -> delayed_reservoir /
                                                           torch_reservoir
    [3] spatially-embedded RNN as a topology generator  -> sernn

Supporting layers
-----------------
* ``spatialrc.backend`` -- pluggable compute backends ({cpu, pytorch, cupy,
  jax}) behind a uniform interface; the expensive kernels (delayed simulation,
  eigendecomposition, ridge readout) dispatch here.
* ``spatialrc.criticality`` -- diagnostics that re-locate the operating point
  once delays break conn2res' spectral-radius criticality.
* ``spatialrc.viz`` -- 2-D and 3-D visualizations matched to the outputs above.
* ``spatialrc.adapters`` -- optional conn2res / reservoirpy glue.

The NumPy core has no heavy dependencies. Torch / JAX / CuPy / matplotlib / vedo
are imported lazily, so ``import spatialrc`` works with only numpy + scipy.
"""

from __future__ import annotations

from .backend import available_backends, get_backend
from .criticality import (
    build_companion,
    companion_spectral_radius,
    memory_capacity,
)
from .delayed_reservoir import DelayedEchoStateNetwork
from .geometry import Geometry

__all__ = [
    "Geometry",
    "DelayedEchoStateNetwork",
    "build_companion",
    "companion_spectral_radius",
    "memory_capacity",
    "get_backend",
    "available_backends",
    # lazily available (torch):
    "TorchDelayReservoir",
    "SpatiallyEmbeddedRNN",
    "train_sernn",
    "grid_coordinates",
    # lazily available (matplotlib/vedo):
    "viz",
]

__version__ = "0.2.0"


def __getattr__(name):
    """Lazily expose optional-dependency symbols (PEP 562).

    Submodules are imported via ``importlib`` rather than ``from . import x`` so
    that resolving the attribute does not re-enter this ``__getattr__`` (which
    would recurse infinitely for subpackages such as ``viz``).
    """
    import importlib

    if name == "TorchDelayReservoir":
        mod = importlib.import_module(f"{__name__}.torch_reservoir")
        return mod.TorchDelayReservoir
    if name in {"SpatiallyEmbeddedRNN", "train_sernn", "grid_coordinates"}:
        mod = importlib.import_module(f"{__name__}.sernn")
        return getattr(mod, name)
    if name == "viz":
        return importlib.import_module(f"{__name__}.viz")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
