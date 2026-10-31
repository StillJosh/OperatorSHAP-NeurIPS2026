"""Finite-difference method-of-lines solver for 1D diffusion-sorption.

Uses centered finite differences (non-periodic BCs) with
scipy.integrate.solve_ivp.
"""

import numpy as np

from .base import PDESolver


class DiffusionSorptionSolver1D(PDESolver):
    """1D diffusion-sorption with Freundlich isotherm.

    du/dt = D * d^2u/dx^2  (with sorption retardation)

    The retardation factor R(u) = 1 + (rho_s / porosity) * k_f * n_f * u^{n_f - 1}
    slows down the diffusion front nonlinearly.

    Boundary conditions: Dirichlet left (u = c_left), zero-flux right.

    Parameters
    ----------
    grid : dict
        Must contain ``"x"``.
    params : dict
        Must contain ``"D"`` (diffusivity), ``"k_f"`` (Freundlich coefficient),
        ``"n_f"`` (Freundlich exponent), ``"rho_s"`` (bulk density),
        ``"porosity"``, ``"c_left"`` (left BC concentration).
    """

    method = "Radau"
    rtol = 1e-5
    atol = 1e-7

    def __init__(self, grid: dict, params: dict):
        super().__init__(grid, params)
        self.x = grid["x"]
        self.dx = self.x[1] - self.x[0]
        self.N = len(self.x)
        self.D = params["D"]
        self.k_f = params["k_f"]
        self.n_f = params["n_f"]
        self.rho_s = params["rho_s"]
        self.porosity = params["porosity"]
        self.c_left = params["c_left"]

    def _retardation(self, u: np.ndarray) -> np.ndarray:
        """Freundlich retardation factor R(u)."""
        safe_u = np.maximum(u, 1e-12)
        return 1.0 + (self.rho_s / self.porosity) * self.k_f * self.n_f * safe_u ** (
            self.n_f - 1
        )

    def rhs(self, t: float, u: np.ndarray) -> np.ndarray:
        dx2 = self.dx**2

        # Interior: centered differences
        d2u = np.zeros_like(u)
        d2u[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx2

        # Left BC: Dirichlet u[0] = c_left (ghost node approach)
        d2u[0] = (u[1] - 2 * u[0] + self.c_left) / dx2

        # Right BC: zero flux (du/dx = 0 => u[N] = u[N-1])
        d2u[-1] = (u[-2] - u[-1]) / dx2

        R = self._retardation(u)
        dudt = self.D * d2u / R
        return dudt
