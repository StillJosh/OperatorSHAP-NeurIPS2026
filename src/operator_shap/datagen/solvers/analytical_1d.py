"""Exact analytical solvers for linear 1D PDEs (advection, heat)."""

import numpy as np

from .base import PDESolver


class AdvectionSolver1D(PDESolver):
    """1D advection: u_t + beta * u_x = 0 on periodic domain.

    Exact solution via Fourier phase shift.

    Parameters
    ----------
    grid : dict
        Must contain ``"x"`` array on [0, L).
    params : dict
        Must contain ``"beta"`` (advection speed).
    """

    def rhs(self, t, u_flat):
        raise NotImplementedError("Analytical solver does not use rhs.")

    def solve(self, u0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        x = self.grid["x"]
        beta = self.params["beta"]
        dx = x[1] - x[0]
        n_points = len(x)

        u0_hat = np.fft.fft(u0)
        k = np.fft.fftfreq(n_points, d=dx) * 2 * np.pi

        solution = np.zeros((len(t_eval), n_points))
        for i, t in enumerate(t_eval):
            phase = np.exp(-1j * k * beta * t)
            solution[i] = np.real(np.fft.ifft(u0_hat * phase))
        return solution


class HeatSolver1D(PDESolver):
    """1D heat equation: u_t = kappa * u_xx on periodic domain.

    Exact solution via Fourier exponential decay.

    Parameters
    ----------
    grid : dict
        Must contain ``"x"`` array on [0, L).
    params : dict
        Must contain ``"kappa"`` (diffusivity).
    """

    def rhs(self, t, u_flat):
        raise NotImplementedError("Analytical solver does not use rhs.")

    def solve(self, u0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        x = self.grid["x"]
        kappa = self.params["kappa"]
        dx = x[1] - x[0]
        n_points = len(x)

        u0_hat = np.fft.fft(u0)
        k = np.fft.fftfreq(n_points, d=dx) * 2 * np.pi

        solution = np.zeros((len(t_eval), n_points))
        for i, t in enumerate(t_eval):
            decay = np.exp(-kappa * k**2 * t)
            solution[i] = np.real(np.fft.ifft(u0_hat * decay))
        return solution
