"""Exact analytical solver for the 2D heat equation."""

import numpy as np

from .base import PDESolver


class HeatSolver2D(PDESolver):
    """2D heat equation: u_t = kappa * (u_xx + u_yy) on periodic domain.

    Exact solution via 2D Fourier exponential decay.

    Parameters
    ----------
    grid : dict
        Must contain ``"x"`` and ``"y"`` arrays.
    params : dict
        Must contain ``"kappa"`` (diffusivity).
    """

    def rhs(self, t, u_flat):
        raise NotImplementedError("Analytical solver does not use rhs.")

    def solve(self, u0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        x, y = self.grid["x"], self.grid["y"]
        kappa = self.params["kappa"]
        dx, dy = x[1] - x[0], y[1] - y[0]
        n_x, n_y = len(x), len(y)

        u0_hat = np.fft.fft2(u0)
        kx = np.fft.fftfreq(n_x, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(n_y, d=dy) * 2 * np.pi
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
        k_sq = kx_grid**2 + ky_grid**2

        solution = np.zeros((len(t_eval), n_x, n_y))
        for i, t in enumerate(t_eval):
            decay = np.exp(-kappa * k_sq * t)
            solution[i] = np.real(np.fft.ifft2(u0_hat * decay))
        return solution
