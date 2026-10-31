"""Base class for PDE solvers using scipy.integrate.solve_ivp."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp


class PDESolver(ABC):
    """Base PDE solver using method of lines with scipy.integrate.solve_ivp.

    Subclasses define ``rhs`` (du/dt = f(t, u)) and the solver handles
    time integration. For PDEs with analytical solutions, override ``solve``
    directly.

    Parameters
    ----------
    grid : dict
        Spatial grid arrays. 1D: ``{"x": array}``. 2D: ``{"x": array, "y": array}``.
    params : dict
        PDE-specific parameters (e.g. ``{"nu": 0.01}``).
    """

    method: str = "RK45"
    rtol: float = 1e-6
    atol: float = 1e-8

    def __init__(self, grid: dict, params: dict):
        self.grid = grid
        self.params = params

    @abstractmethod
    def rhs(self, t: float, u_flat: np.ndarray) -> np.ndarray:
        """Right-hand side du/dt = f(t, u) for method of lines.

        Parameters
        ----------
        t : float
            Current time.
        u_flat : np.ndarray
            Flattened state vector.

        Returns
        -------
        np.ndarray
            Time derivative, same shape as ``u_flat``.
        """

    def solve(self, u0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        """Integrate the PDE from u0, returning snapshots at t_eval.

        Parameters
        ----------
        u0 : np.ndarray
            Initial condition, one spatial snapshot (multi-channel for systems).
        t_eval : np.ndarray
            Times at which to store snapshots.

        Returns
        -------
        np.ndarray
            Solution array of shape ``(len(t_eval), *u0.shape)``.
        """
        sol = solve_ivp(
            self.rhs,
            [t_eval[0], t_eval[-1]],
            u0.ravel(),
            t_eval=t_eval,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        return sol.y.T.reshape(len(t_eval), *u0.shape)
