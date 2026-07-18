"""Tests for spatialrc.backend registry and cross-backend numerical agreement."""

import numpy as np
import pytest

from spatialrc.backend import available_backends, get_backend
from spatialrc.backend.cpu import CpuBackend


def _toy_system(seed=0, n=24, sparsity=0.25):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, n)) * (rng.random((n, n)) < sparsity)
    np.fill_diagonal(w, 0.0)
    w = w / (np.abs(np.linalg.eigvals(w)).max()) * 0.9
    delay = np.clip(rng.integers(1, 6, size=(n, n)), 1, None)
    t_steps = 200
    u = rng.uniform(-1, 1, size=(t_steps, 1))
    w_in = np.zeros((1, n))
    w_in[0, :4] = 1.0
    return w, delay, u, w_in


def test_cpu_always_available():
    assert "cpu" in available_backends()
    b = get_backend("cpu")
    assert isinstance(b, CpuBackend)


def test_auto_returns_usable_backend():
    b = get_backend("auto")
    assert b.name in available_backends()


def test_missing_backend_raises():
    with pytest.raises(RuntimeError):
        get_backend("does-not-exist")


def test_cpu_lag1_matches_plain_esn():
    w, _, u, w_in = _toy_system()
    n = w.shape[0]
    delay1 = np.ones_like(w, dtype=int)
    b = get_backend("cpu")
    states = b.run_delayed_reservoir(w, w_in, delay1, u, "tanh", None)

    # reference plain lag-1 loop
    ref = np.zeros((len(u), n))
    prev = np.zeros(n)
    for t in range(len(u)):
        prev = np.tanh(prev @ w + u[t] @ w_in)
        ref[t] = prev
    assert np.allclose(states, ref, atol=1e-10)


def test_companion_lag1_equals_spectral_radius():
    w, _, _, _ = _toy_system()
    b = get_backend("cpu")
    rho = b.spectral_radius(w)
    assert abs(rho - 0.9) < 1e-6  # we normalised to 0.9 above


@pytest.mark.parametrize("other", ["pytorch", "cupy", "jax"])
def test_cross_backend_agreement(other):
    if other not in available_backends():
        pytest.skip(f"{other} backend unavailable")
    w, delay, u, w_in = _toy_system()
    ref = get_backend("cpu").run_delayed_reservoir(w, w_in, delay, u, "tanh", 0.3)
    b = get_backend(other)
    got = b.to_numpy(b.run_delayed_reservoir(w, w_in, delay, u, "tanh", 0.3))
    # backends may run in float32; loosen tolerance accordingly
    assert np.allclose(ref, got, atol=1e-3, rtol=1e-3)
