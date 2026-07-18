"""Tests for the delayed reservoir and criticality diagnostics."""

import numpy as np
import pytest

from spatialrc import (
    DelayedEchoStateNetwork,
    build_companion,
    companion_spectral_radius,
    memory_capacity,
)


def _toy(seed=0, n=30):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.2)
    np.fill_diagonal(w, 0.0)
    w = w / (np.abs(np.linalg.eigvals(w)).max()) * 0.9
    delay = np.clip(rng.integers(1, 7, size=(n, n)), 1, None)
    return w, delay


def test_delay1_reduces_to_plain_esn():
    w, _ = _toy()
    n = w.shape[0]
    esn = DelayedEchoStateNetwork(w, delay=np.ones_like(w, int),
                                  activation_function="tanh")
    rng = np.random.default_rng(1)
    u = rng.uniform(-1, 1, size=(150, 1))
    w_in = np.zeros((1, n))
    w_in[0, :3] = 1.0
    states = esn.simulate(u, w_in, washout=0)
    ref = np.zeros((len(u), n))
    prev = np.zeros(n)
    for t in range(len(u)):
        prev = np.tanh(prev @ w + u[t] @ w_in)
        ref[t] = prev
    assert np.allclose(states, ref, atol=1e-10)


def test_delayed_states_finite_and_shaped():
    w, delay = _toy()
    n = w.shape[0]
    esn = DelayedEchoStateNetwork(w, delay=delay)
    rng = np.random.default_rng(2)
    u = rng.uniform(-1, 1, size=(200, 1))
    w_in = np.zeros((1, n))
    w_in[0, :3] = 1.0
    wo = esn.max_delay
    states = esn.simulate(u, w_in, washout=wo)
    assert states.shape == (200 - wo, n)
    assert np.isfinite(states).all()


def test_output_nodes_subset():
    w, delay = _toy()
    n = w.shape[0]
    esn = DelayedEchoStateNetwork(w, delay=delay)
    rng = np.random.default_rng(3)
    u = rng.uniform(-1, 1, size=(120, 1))
    w_in = np.zeros((1, n))
    w_in[0, :3] = 1.0
    nodes = np.array([0, 5, 10])
    states = esn.simulate(u, w_in, washout=esn.max_delay, output_nodes=nodes)
    assert states.shape[1] == 3


def test_companion_lag1_equals_weight_spectral_radius():
    w, _ = _toy()
    delay1 = np.ones_like(w, dtype=int)
    rho_comp = companion_spectral_radius(w, delay1)
    rho_w = float(np.abs(np.linalg.eigvals(w)).max())
    assert np.isclose(rho_comp, rho_w, atol=1e-8)


def test_delays_shift_criticality():
    w, delay = _toy()
    rho_lag1 = companion_spectral_radius(w, np.ones_like(w, int))
    rho_del = companion_spectral_radius(w, delay)
    # delayed companion generally differs from the lag-1 spectral radius
    assert not np.isclose(rho_lag1, rho_del, atol=1e-3)


def test_companion_shape():
    w, delay = _toy(n=12)
    c = build_companion(w, delay)
    d_max = int(delay[w != 0].max())
    assert c.shape == (12 * d_max, 12 * d_max)


def test_memory_capacity_runs_and_bounded():
    w, _ = _toy()
    n = w.shape[0]
    esn = DelayedEchoStateNetwork(w, delay=np.ones_like(w, int))
    w_in = np.zeros((1, n))
    w_in[0, :3] = 1.0
    total, curve = memory_capacity(
        lambda u: esn.simulate(u, w_in, washout=0),
        n_steps=1200, max_delay=25, washout=100,
    )
    assert curve.shape == (25,)
    assert 0.0 <= total <= 25.0  # bounded by number of lags
    assert (curve >= -1e-9).all()
