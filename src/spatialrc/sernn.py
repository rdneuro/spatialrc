"""
sernn.py
========

Spatially-Embedded RNN (seRNN) used as a *topology generator*.

This implements the Achterberg / Akarca / Astle (Nat. Mach. Intell. 2023)
spatial regularizer

    L = L_task + gamma * || W  (x)  D  (x)  C ||_1

where ``(x)`` is the element-wise product, ``D`` is the Euclidean distance
matrix between units placed in 3D space, and ``C`` is the weighted
communicability

    C = expm( S^{-1/2}  |W|  S^{-1/2} ),   S = diag(node strengths),

with the diagonal of ``C`` zeroed (Crofts-Higham unbiased weighted
communicability).

CONCEPTUAL NOTE (why "generator" and not "reservoir")
-----------------------------------------------------
An seRNN has *trained* recurrent weights, so it is NOT a reservoir. The
legitimate way to use it in a reservoir-computing pipeline is:

    1. train the seRNN on a task with the spatial regularizer,
    2. FREEZE the recurrent matrix,
    3. import it as a fixed reservoir (``export_frozen_reservoir``) and train
       only a linear readout on top.

That preserves the reservoir assumption (fixed internal weights) while letting
the *topology* be shaped by wiring-cost + communicability optimisation.

CAVEATS
-------
* ``C`` is a full matrix exponential recomputed every optimisation step
  (O(N^3)) and can produce vanishing/exploding gradients through ``expm``; keep
  N modest (module-sized) if you must train at scale, or truncate the series.
* The original paper uses a synthetic 5x5x4 integer grid, NOT brain
  coordinates. Porting to real, irregular parcel coordinates is untested --
  validate that the emergent topology (modularity, small-worldness) survives.
* No Dale's law is imposed here (as in the original); do not assume E/I
  segregation of the resulting weights.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _matrix_exp(m: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.linalg, "matrix_exp"):
        return torch.linalg.matrix_exp(m)
    return torch.matrix_exp(m)


def grid_coordinates(shape=(5, 5, 4)) -> np.ndarray:
    """Integer grid coordinates (the original seRNN 5x5x4 = 100-unit layout)."""
    axes = [np.arange(s) for s in shape]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1).astype(float)


class SpatiallyEmbeddedRNN(nn.Module):
    """A minimal trainable seRNN with the spatial (D (x) C) regularizer.

    Parameters
    ----------
    coords : (N, 3) array-like
        Unit coordinates. Use ``grid_coordinates()`` to reproduce the original
        5x5x4 layout, or pass real parcel centroids.
    n_in, n_out : int
        Input and output dimensions of the wrapped task.
    activation : callable, default torch.relu
    gamma : float
        Spatial regularization strength (paper's headline network uses ~0.08;
        the public demo defaults to 0.5 -- sweep it).
    dtype : torch dtype
    """

    def __init__(
        self,
        coords,
        n_in: int,
        n_out: int,
        activation: Callable = torch.relu,
        gamma: float = 0.08,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        coords = np.asarray(coords, dtype=float)
        n = coords.shape[0]
        self.n_units = n
        self.activation = activation
        self.gamma = float(gamma)
        self._dtype = dtype

        # Euclidean distance matrix between units (fixed)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        self.register_buffer("D", torch.as_tensor(dist, dtype=dtype))

        # trainable weights
        self.Wr = nn.Parameter(torch.empty(n, n, dtype=dtype))
        self.Win = nn.Parameter(torch.empty(n, n_in, dtype=dtype))
        self.Wout = nn.Parameter(torch.empty(n_out, n, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(n, dtype=dtype))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # orthogonal recurrent init (as in the reference implementation)
        nn.init.orthogonal_(self.Wr)
        nn.init.xavier_uniform_(self.Win)
        nn.init.xavier_uniform_(self.Wout)

    # ------------------------------------------------------------------ #
    def communicability(self) -> torch.Tensor:
        """Weighted communicability C = expm(S^-1/2 |W| S^-1/2), diag zeroed."""
        A = self.Wr.abs()
        s = A.sum(dim=1)
        s_inv_sqrt = torch.where(
            s > 0, s.clamp(min=1e-12).pow(-0.5), torch.zeros_like(s)
        )
        M = s_inv_sqrt[:, None] * A * s_inv_sqrt[None, :]
        C = _matrix_exp(M)
        C = C - torch.diag_embed(torch.diagonal(C))
        return C

    def spatial_regularizer(self) -> torch.Tensor:
        """gamma * || W (x) D (x) C ||_1 ."""
        C = self.communicability()
        return self.gamma * (self.Wr.abs() * self.D * C).sum()

    def forward(self, x: torch.Tensor):
        """Run the RNN over a sequence.

        Parameters
        ----------
        x : (batch, T, n_in) tensor

        Returns
        -------
        logits : (batch, n_out) tensor  -- readout of the final hidden state
        hidden : (batch, T, n_units) tensor
        """
        x = x.to(dtype=self._dtype)
        batch, T, _ = x.shape
        h = torch.zeros(batch, self.n_units, device=x.device, dtype=self._dtype)
        hs = []
        for t in range(T):
            pre = h @ self.Wr.t() + x[:, t, :] @ self.Win.t() + self.b
            h = self.activation(pre)
            hs.append(h)
        hidden = torch.stack(hs, dim=1)
        logits = h @ self.Wout.t()
        return logits, hidden

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def export_frozen_reservoir(
        self, spectral_radius: Optional[float] = 0.9
    ) -> np.ndarray:
        """Return the trained recurrent matrix as a fixed reservoir (NumPy).

        Optionally rescales to a target spectral radius (for the non-delayed
        case; remember delays invalidate this target -- re-tune afterwards).
        Convention flip: ``Wr`` is (target-row, source-col) here (h @ Wr.T), but
        the reservoir engines use (source-row, target-col), so we transpose.
        """
        W = self.Wr.detach().cpu().numpy().T.copy()  # -> (source, target)
        if spectral_radius is not None:
            ev = np.linalg.eigvals(W)
            r = np.abs(ev).max()
            if r > 0:
                W = W / r * float(spectral_radius)
        return W


def train_sernn(
    model: SpatiallyEmbeddedRNN,
    X: torch.Tensor,
    y: torch.Tensor,
    task_loss_fn: Callable = F.cross_entropy,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[str] = None,
    verbose: bool = True,
):
    """Generic seRNN training loop (task loss + spatial regularizer).

    Parameters
    ----------
    model : SpatiallyEmbeddedRNN
    X : (n_samples, T, n_in) tensor
    y : (n_samples,) or (n_samples, n_out) tensor (task targets)
    task_loss_fn : callable(logits, y) -> scalar
    epochs, lr, batch_size : optimisation hyper-parameters
    device : 'cpu' / 'cuda' / None (auto)

    Returns
    -------
    model : the trained model (in-place).
    history : list of dict with per-epoch losses.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    n = X.shape[0]
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        task_acc, reg_acc, nb = 0.0, 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits, _ = model(X[idx])
            task = task_loss_fn(logits, y[idx])
            reg = model.spatial_regularizer()
            loss = task + reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            task_acc += float(task.detach())
            reg_acc += float(reg.detach())
            nb += 1
        rec = {"epoch": epoch, "task": task_acc / nb, "reg": reg_acc / nb}
        history.append(rec)
        if verbose:
            print(f"[seRNN] epoch {epoch:02d} | task={rec['task']:.4f} "
                  f"| reg={rec['reg']:.4f}")
    return model, history
