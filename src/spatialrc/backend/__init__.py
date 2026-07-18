"""
spatialrc.backend
=================

Compute-backend registry and dispatch.

Four backends live in the submodules ``cpu``, ``pytorch``, ``cupy`` and ``jax``,
each implementing the expensive kernels behind the uniform :class:`Backend`
interface. Only the CPU (NumPy) backend is a hard dependency; the others are
optional and imported lazily, so importing this package never requires torch,
cupy or jax to be installed.

Usage
-----
>>> from spatialrc.backend import get_backend
>>> b = get_backend("auto")          # cupy > pytorch(cuda) > jax > cpu
>>> b = get_backend("pytorch")       # explicit; raises if torch is missing
>>> print(b.name, b.device)

The auto-selection order prefers GPU backends when they are usable and always
falls back to CPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base import Backend
from .cpu import CpuBackend

__all__ = ["Backend", "get_backend", "available_backends", "register_backend"]

#: name -> Backend subclass. Populated lazily to avoid importing torch/cupy/jax.
_REGISTRY: Dict[str, Type[Backend]] = {"cpu": CpuBackend}

#: preference order used by ``get_backend('auto')``
_AUTO_ORDER = ("cupy", "pytorch", "jax", "cpu")

#: cache of instantiated singletons keyed by (name, device)
_INSTANCES: Dict[str, Backend] = {}


def _load_class(name: str) -> Optional[Type[Backend]]:
    """Import a backend class on demand; return None if its deps are missing."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    try:
        if name == "pytorch":
            from .pytorch import TorchBackend as cls
        elif name == "cupy":
            from .cupy import CupyBackend as cls
        elif name == "jax":
            from .jax import JaxBackend as cls
        else:
            return None
    except Exception:
        return None
    _REGISTRY[name] = cls
    return cls


def register_backend(name: str, cls: Type[Backend]) -> None:
    """Register a custom backend class under ``name``."""
    if not issubclass(cls, Backend):
        raise TypeError("cls must subclass spatialrc.backend.base.Backend.")
    _REGISTRY[name] = cls


def available_backends() -> List[str]:
    """List backend names whose dependencies are importable and usable."""
    out = []
    for name in _AUTO_ORDER:
        cls = _load_class(name)
        if cls is not None and cls.is_available():
            out.append(name)
    return out


def get_backend(name: str = "auto", device: Optional[str] = None) -> Backend:
    """Return a backend instance.

    Parameters
    ----------
    name : str
        One of {'auto', 'cpu', 'pytorch', 'cupy', 'jax'} or a registered custom
        name. 'auto' selects the first usable backend in the order
        cupy > pytorch(cuda) > jax > cpu.
    device : str, optional
        Device hint forwarded to backends that accept one (e.g. 'cuda:0' for
        pytorch). Ignored by cpu/jax.

    Returns
    -------
    Backend
        A ready-to-use backend instance (cached per name+device).

    Raises
    ------
    RuntimeError
        If an explicitly requested backend is unavailable.
    """
    if name == "auto":
        for candidate in _AUTO_ORDER:
            cls = _load_class(candidate)
            if cls is not None and cls.is_available():
                name = candidate
                break

    cls = _load_class(name)
    if cls is None or not cls.is_available():
        raise RuntimeError(
            f"Backend '{name}' is not available. "
            f"Installed/usable backends: {available_backends()}."
        )

    key = f"{name}:{device}"
    if key not in _INSTANCES:
        try:
            _INSTANCES[key] = cls(device=device) if device is not None else cls()
        except TypeError:
            # backend __init__ does not accept a device argument
            _INSTANCES[key] = cls()
    return _INSTANCES[key]
