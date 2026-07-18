"""Tests for the visualization module and the (optional) seRNN generator."""

import importlib.util
import os

import numpy as np
import pytest

# use a headless backend for matplotlib in CI
os.environ.setdefault("MPLBACKEND", "Agg")

_HAS_MPL = importlib.util.find_spec("matplotlib") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(not _HAS_MPL, reason="matplotlib not installed")


@pytest.fixture
def toy(tmp_path):
    rng = np.random.default_rng(0)
    n = 20
    coords = rng.uniform(0, 60, size=(n, 3))
    w = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.25)
    np.fill_diagonal(w, 0)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    delay = np.clip((dist / 3.0).round(), 1, None).astype(int)
    states = rng.normal(size=(80, n))
    return dict(coords=coords, w=w, dist=dist, delay=delay, states=states,
               tmp=tmp_path)


def test_matrix_plots(toy):
    from spatialrc.viz import (
        plot_connectome, plot_delay_matrix, plot_distance_matrix,
        plot_reservoir_states, plot_weight_distance, savefig,
    )

    savefig(plot_connectome(toy["w"]), toy["tmp"] / "conn.png")
    savefig(plot_distance_matrix(toy["dist"]), toy["tmp"] / "dist.png")
    savefig(plot_delay_matrix(toy["delay"], w=toy["w"]), toy["tmp"] / "delay.png")
    savefig(plot_weight_distance(np.abs(toy["w"]), toy["dist"], lam=0.05),
            toy["tmp"] / "wd.png")
    savefig(plot_reservoir_states(toy["states"], mode="heatmap"),
            toy["tmp"] / "states.png")
    for name in ["conn.png", "dist.png", "delay.png", "wd.png", "states.png"]:
        assert (toy["tmp"] / name).exists()


def test_criticality_plots(toy):
    from spatialrc.viz import (
        plot_alpha_sweep, plot_criticality_heatmap, plot_eigenspectrum,
        plot_memory_capacity_curve, savefig,
    )

    savefig(plot_memory_capacity_curve(np.abs(np.random.randn(20))),
            toy["tmp"] / "mc.png")
    alphas = np.linspace(0.2, 2.0, 8)
    savefig(plot_alpha_sweep(alphas, np.random.rand(8, 2),
                             labels=["tanh", "sigmoid"], critical_alpha=1.0),
            toy["tmp"] / "sweep.png")
    savefig(plot_criticality_heatmap(alphas, [1, 5, 10], np.random.rand(3, 8)),
            toy["tmp"] / "heat.png")
    ev = np.linalg.eigvals(toy["w"])
    savefig(plot_eigenspectrum(ev), toy["tmp"] / "spec.png")
    for name in ["mc.png", "sweep.png", "heat.png", "spec.png"]:
        assert (toy["tmp"] / name).exists()


def test_embedding3d_matplotlib_fallback(toy):
    from spatialrc.viz import plot_node_embedding_3d

    out = plot_node_embedding_3d(
        toy["coords"], w=toy["w"], delay=toy["delay"],
        out_path=str(toy["tmp"] / "emb.png"), backend="matplotlib",
    )
    assert os.path.exists(out)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_sernn_train_and_freeze():
    import torch

    from spatialrc import SpatiallyEmbeddedRNN, grid_coordinates, train_sernn

    coords = grid_coordinates((3, 3, 3))
    model = SpatiallyEmbeddedRNN(coords, n_in=2, n_out=3, gamma=0.08)
    x = torch.randn(32, 8, 2)
    y = torch.randint(0, 3, (32,))
    _, hist = train_sernn(model, x, y, epochs=2, lr=1e-3, verbose=False)
    assert len(hist) == 2
    w_frozen = model.export_frozen_reservoir(spectral_radius=0.9)
    assert w_frozen.shape == (27, 27)
    assert np.isclose(np.abs(np.linalg.eigvals(w_frozen)).max(), 0.9, atol=1e-6)
