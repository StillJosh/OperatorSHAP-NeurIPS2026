"""Exact analytical solver for the 3D heat equation."""

import numpy as np

from .base import PDESolver


class HeatSolver3D(PDESolver):
    """3D heat equation: u_t = kappa * laplace(u) on a periodic domain.

    Exact solution via 3D Fourier exponential decay.

    Parameters
    ----------
    grid : dict
        Must contain ``"x"``, ``"y"``, and ``"z"`` arrays.
    params : dict
        Must contain ``"kappa"`` (diffusivity).
    """

    def rhs(self, t, u_flat):
        raise NotImplementedError("Analytical solver does not use rhs.")

    def solve(self, u0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        x, y, z = self.grid["x"], self.grid["y"], self.grid["z"]
        kappa = self.params["kappa"]

        u0_hat = np.fft.fftn(u0)
        kx = np.fft.fftfreq(len(x), d=x[1] - x[0]) * 2 * np.pi
        ky = np.fft.fftfreq(len(y), d=y[1] - y[0]) * 2 * np.pi
        kz = np.fft.fftfreq(len(z), d=z[1] - z[0]) * 2 * np.pi
        kxg, kyg, kzg = np.meshgrid(kx, ky, kz, indexing="ij")
        k_sq = kxg**2 + kyg**2 + kzg**2

        solution = np.zeros((len(t_eval), len(x), len(y), len(z)))
        for i, t in enumerate(t_eval):
            decay = np.exp(-kappa * k_sq * t)
            solution[i] = np.real(np.fft.ifftn(u0_hat * decay))
        return solution
