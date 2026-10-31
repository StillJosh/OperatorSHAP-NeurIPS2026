"""Pseudo-spectral method-of-lines solvers for nonlinear 1D PDEs.

Uses FFT-based spatial derivatives with scipy.integrate.solve_ivp for
time integration. The Burgers product u * u_x is low-pass filtered to |k| <= n/6;
the Allen-Cahn cubic term is not filtered.
"""

import numpy as np

from .base import PDESolver


def _spectral_derivative(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Compute du/dx via FFT."""
    return np.real(np.fft.ifft(1j * k * np.fft.fft(u)))


def _spectral_laplacian(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Compute d^2u/dx^2 via FFT."""
    return np.real(np.fft.ifft(-(k**2) * np.fft.fft(u)))


def _dealias_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the product of a and b, each low-pass filtered to |k| <= n/6 before multiplying."""
    n_points = len(a)
    a_hat = np.fft.fft(a)
    b_hat = np.fft.fft(b)
    cutoff = n_points // 3
    a_hat[n_points // 2 - cutoff : n_points // 2 + cutoff + 1] = 0
    b_hat[n_points // 2 - cutoff : n_points // 2 + cutoff + 1] = 0
    a_f = np.real(np.fft.ifft(a_hat))
    b_f = np.real(np.fft.ifft(b_hat))
    return a_f * b_f


class BurgersSolver1D(PDESolver):
    """1D viscous Burgers: u_t + u * u_x = nu * u_xx on periodic domain.

    Parameters
    ----------
    grid : dict
        Must contain ``"x"``.
    params : dict
        Must contain ``"nu"`` (viscosity).
    """

    method = "RK45"

    def __init__(self, grid: dict, params: dict):
        super().__init__(grid, params)
        dx = grid["x"][1] - grid["x"][0]
        n_points = len(grid["x"])
        self.k = np.fft.fftfreq(n_points, d=dx) * 2 * np.pi
        self.nu = params["nu"]

    def rhs(self, t: float, u: np.ndarray) -> np.ndarray:
        u_x = _spectral_derivative(u, self.k)
        u_xx = _spectral_laplacian(u, self.k)
        nonlinear = _dealias_product(u, u_x)
        return -nonlinear + self.nu * u_xx


class AllenCahnSolver1D(PDESolver):
    """1D Allen-Cahn: u_t = eps^2 * u_xx + u - u^3 on periodic domain.

    Uses Radau (implicit) solver for the stiff diffusion term.

    Parameters
    ----------
    grid : dict
        Must contain ``"x"``.
    params : dict
        Must contain ``"eps"`` (interface width parameter).
    """

    method = "Radau"
    rtol = 1e-5
    atol = 1e-7

    def __init__(self, grid: dict, params: dict):
        super().__init__(grid, params)
        dx = grid["x"][1] - grid["x"][0]
        n_points = len(grid["x"])
        self.k = np.fft.fftfreq(n_points, d=dx) * 2 * np.pi
        self.eps2 = params["eps"] ** 2

    def rhs(self, t: float, u: np.ndarray) -> np.ndarray:
        u_xx = _spectral_laplacian(u, self.k)
        return self.eps2 * u_xx + u - u**3
