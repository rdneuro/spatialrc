"""
Pilot experiment: spatial embedding x conduction delays x criticality.

Staged Spyder-style script (# %% cells). Each cell runs its stage inline and
produces a figure/table so the pipeline is inspectable step by step. Replace the
synthetic connectome/coordinates with your CamCAN (streamline lengths) or CEPESC
(geodesic-only) data.

Run cell-by-cell in Spyder/VS Code, or top-to-bottom as a script:

    python examples/pilot_experiment.py
"""

# %% [0] imports and a synthetic substrate --------------------------------------
import numpy as np

from spatialrc import (
    Geometry,
    DelayedEchoStateNetwork,
    companion_spectral_radius,
    memory_capacity,
    available_backends,
    get_backend,
)
from spatialrc.delayed_reservoir import _spectral_normalize

print("available backends:", available_backends())
BACKEND = "auto"  # cupy > pytorch(cuda) > jax > cpu

rng = np.random.default_rng(42)
N = 120
coords = rng.uniform(0, 90, size=(N, 3))                 # parcel centroids (mm)
geom = Geometry(coords=coords)                            # Euclidean auto-filled
dist = geom.euclidean_distances()

# distance-dependent connectome (planted EDR so the demo is realistic)
lam_true = 0.06
w = np.exp(-lam_true * dist) * (rng.random((N, N)) < 0.15)
np.fill_diagonal(w, 0.0)
w *= rng.choice([-1.0, 1.0], size=w.shape)               # mixed E/I
print("edges:", int((w != 0).sum()))

# %% [1] geometry: fit EDR, build delays ----------------------------------------
lam = geom.fit_edr_lambda(np.abs(w))
print(f"fitted EDR lambda = {lam:.4f} mm^-1 (planted {lam_true})")

w_norm = _spectral_normalize(w)                          # baseline gain-1 reservoir
D = geom.delay_matrix(w_norm, velocity=3.0, dt=1.0, source="euclidean")

# %% [2] sweep alpha x D_max, measuring criticality + memory capacity ------------
alphas = np.linspace(0.4, 1.6, 7)
velocities = [1.0, 3.0, 10.0]        # mm/ms -> different D_max regimes
input_nodes = np.arange(6)
w_in = np.zeros((1, N)); w_in[0, input_nodes] = 1.0

mc_grid = np.zeros((len(velocities), len(alphas)))
rho_grid = np.zeros_like(mc_grid)
dmax_by_v = []

for vi, vel in enumerate(velocities):
    Dv = geom.delay_matrix(w_norm, velocity=vel, dt=1.0, source="euclidean",
                           verbose=False)
    dmax_by_v.append(int(Dv[w_norm != 0].max()))
    for ai, a in enumerate(alphas):
        w_scaled = w_norm * a
        esn = DelayedEchoStateNetwork(w_scaled, delay=Dv,
                                      activation_function="tanh",
                                      backend=BACKEND)
        rho_grid[vi, ai] = companion_spectral_radius(w_scaled, Dv, backend=BACKEND)
        total_mc, _ = memory_capacity(
            lambda u: esn.simulate(u, w_in, washout=esn.max_delay),
            n_steps=1500, max_delay=40, washout=esn.max_delay + 100,
            backend=BACKEND,
        )
        mc_grid[vi, ai] = total_mc
    print(f"velocity={vel} mm/ms  D_max={dmax_by_v[-1]}  MC(alpha)={mc_grid[vi]}")

# %% [3] figures ----------------------------------------------------------------
from spatialrc import viz

viz.savefig(viz.plot_weight_distance(np.abs(w), dist, lam=lam),
            "pilot_weight_distance.png")
viz.savefig(viz.plot_delay_matrix(D, w=w_norm), "pilot_delay_matrix.png")

# criticality landscape (rows = velocity/D_max, cols = alpha)
viz.savefig(
    viz.plot_criticality_heatmap(alphas, dmax_by_v, mc_grid,
                                 metric="memory capacity"),
    "pilot_mc_landscape.png",
)

# companion eigenspectrum at the nominal operating point
from spatialrc.criticality import build_companion
be = get_backend(BACKEND)
comp = build_companion(w_norm * 1.0, D)
ev = be.to_numpy(be.eigvals(comp))
viz.savefig(viz.plot_eigenspectrum(ev), "pilot_eigenspectrum.png")

# 3-D spatial embedding coloured by out-degree, edges coloured by delay
viz.plot_node_embedding_3d(coords, w=w_norm, delay=D, edge_color_by="delay",
                           out_path="pilot_embedding3d.png", max_edges=1500)

print("\nWrote: pilot_weight_distance.png, pilot_delay_matrix.png, "
      "pilot_mc_landscape.png, pilot_eigenspectrum.png, pilot_embedding3d.png")

# %% [4] takeaways --------------------------------------------------------------
# The MC landscape shows the operating point moving off alpha=1 as D_max grows:
# never inherit conn2res' spectral-radius criticality once delays are on.
# Compare against degree-preserving and distance-preserving nulls before
# attributing any effect to spatial embedding.
