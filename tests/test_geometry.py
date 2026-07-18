"""Tests for spatialrc.geometry."""

import numpy as np
import pytest

from spatialrc.geometry import Geometry


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_euclidean_from_coords(rng):
    coords = rng.uniform(0, 50, size=(20, 3))
    geom = Geometry(coords=coords)
    d = geom.euclidean_distances()
    assert d.shape == (20, 20)
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(d, d.T)  # Euclidean is symmetric


def test_delay_matrix_units_and_floor(rng):
    coords = rng.uniform(0, 90, size=(15, 3))
    geom = Geometry(coords=coords)
    w = (rng.random((15, 15)) < 0.3).astype(float)
    np.fill_diagonal(w, 0)
    D = geom.delay_matrix(w, velocity=3.0, dt=1.0, source="euclidean",
                          min_delay=1, verbose=False)
    assert D.shape == (15, 15)
    assert D.min() >= 1
    assert D.dtype.kind == "i"


def test_edr_reweight_preserves_edges_and_sign(rng):
    coords = rng.uniform(0, 50, size=(25, 3))
    geom = Geometry(coords=coords)
    w = rng.normal(size=(25, 25)) * (rng.random((25, 25)) < 0.25)
    np.fill_diagonal(w, 0)
    wr = geom.edr_reweight(w, lam=0.05)
    assert np.array_equal(w != 0, wr != 0)
    nz = w != 0
    assert np.all(np.sign(w[nz]) == np.sign(wr[nz]))


def test_fit_edr_lambda_recovers_planted_decay(rng):
    coords = rng.uniform(0, 60, size=(60, 3))
    geom = Geometry(coords=coords)
    d = geom.euclidean_distances()
    lam_true = 0.08
    w = np.exp(-lam_true * d) * (rng.random(d.shape) < 0.5)
    np.fill_diagonal(w, 0)
    lam_hat = geom.fit_edr_lambda(w)
    # robust fit should land within a reasonable tolerance of the planted value
    assert abs(lam_hat - lam_true) < 0.03


def test_effective_distance_prefers_length(rng):
    coords = rng.uniform(0, 40, size=(10, 3))
    geom = Geometry(coords=coords)
    length = np.abs(rng.normal(size=(10, 10)))
    np.fill_diagonal(length, 0)
    geom.set_streamline_lengths(length)
    assert np.allclose(geom.effective_distance("length"), length)
