# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to Semantic Versioning.

## [0.2.0] - 2026
### Added
- Pluggable compute backends `spatialrc.backend.{cpu,pytorch,cupy,jax}` behind a
  uniform `Backend` interface, with a lazy registry, availability detection and
  `auto` selection (cupy > pytorch(cuda) > jax > cpu).
- The three expensive kernels now dispatch to the selected backend: the delayed
  reservoir loop (`run_delayed_reservoir`), the criticality eigendecomposition
  (`eigvals`), and the ridge readout solve (`linear_readout_fit`).
- JAX backend uses a functional `lax.scan` for the delayed recurrence.
- `spatialrc.viz`: 2-D (connectome / distance / delay matrices, weight-distance
  with EDR overlay, reservoir states, memory-capacity curve, alpha sweep,
  alpha x D_max landscape, companion eigenspectrum) and 3-D spatial embedding
  (vedo offscreen with a matplotlib fallback).
- `src/` layout, `pyproject.toml` (hatchling), test suite and a pilot example.

### Changed
- `DelayedEchoStateNetwork` now accepts a `backend=` argument and delegates its
  hot loop; the conn2res-compatible `simulate(...)` signature is unchanged.
- `criticality.companion_spectral_radius` / `memory_capacity` accept `backend=`.

## [0.1.0]
### Added
- Initial flat package: `Geometry`, `DelayedEchoStateNetwork` (NumPy),
  `TorchDelayReservoir`, `SpatiallyEmbeddedRNN`, criticality diagnostics,
  conn2res / reservoirpy adapters.
