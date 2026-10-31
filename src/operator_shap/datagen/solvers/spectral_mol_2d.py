"""Pseudo-spectral method-of-lines solver for 2D reaction-diffusion (Gray-Scott)."""

import numpy as np

from .base import PDESolver


class GrayScottSolver2D(PDESolver):
    """2D Gray-Scott reaction-diffusion on periodic domain.

    u_t = Du * laplacian(u) - u*v^2 + F*(1-u)
    v_t = Dv * laplacian(v) + u*v^2 - (F+k)*v

    The state is stored as a flat vector [u.ravel(), v.ravel()].

    Parameters
    ----------
    grid : dict
        Must contain ``"x"`` and ``"y"`` arrays.
    params : dict
        Must contain ``"Du"``, ``"Dv"``, ``"F"``, ``"k"``.
    """

    method = "RK45"
    rtol = 1e-4
    atol = 1e-6

    def __init__(self, grid: dict, params: dict):
        super().__init__(grid, params)
        x, y = grid["x"], grid["y"]
        self.n_x, self.n_y = len(x), len(y)
        self.n_dof = self.n_x * self.n_y

        kx = np.fft.fftfreq(self.n_x, d=x[1] - x[0]) * 2 * np.pi
        ky = np.fft.fftfreq(self.n_y, d=y[1] - y[0]) * 2 * np.pi
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
        self.k_sq = kx_grid**2 + ky_grid**2

        self.Du = params["Du"]
        self.Dv = params["Dv"]
        self.F = params["F"]
        self.k = params["k"]

    def rhs(self, t: float, uv_flat: np.ndarray) -> np.ndarray:
        u = uv_flat[: self.n_dof].reshape(self.n_x, self.n_y)
        v = uv_flat[self.n_dof :].reshape(self.n_x, self.n_y)

        # Spectral Laplacians
        lap_u = np.real(np.fft.ifft2(-self.k_sq * np.fft.fft2(u)))
        lap_v = np.real(np.fft.ifft2(-self.k_sq * np.fft.fft2(v)))

        uv2 = u * v**2

        du = self.Du * lap_u - uv2 + self.F * (1.0 - u)
        dv = self.Dv * lap_v + uv2 - (self.F + self.k) * v

        return np.concatenate([du.ravel(), dv.ravel()])
