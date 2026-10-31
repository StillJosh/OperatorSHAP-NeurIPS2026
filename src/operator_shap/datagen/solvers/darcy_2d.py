"""Solver for 2D Darcy flow: -div(a(x) grad u) = f.

Static elliptic PDE solved via sparse finite differences + direct solve.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import PDESolver


class DarcySolver2D(PDESolver):
    """2D Darcy flow on [0,1]^2 with homogeneous Dirichlet BC.

    Given permeability field a(x,y) and constant forcing f, solves
    -div(a * grad(u)) = f for pressure u.

    This is a static PDE — ``solve`` takes a permeability field as ``u0``
    and returns the pressure field (no time dimension).

    Parameters
    ----------
    grid : dict
        Must contain ``"x"`` and ``"y"`` arrays.
    params : dict
        Must contain ``"forcing"`` (constant RHS value).
    """

    def rhs(self, t, u_flat):
        raise NotImplementedError("Static solver does not use rhs.")

    def solve(self, u0: np.ndarray, t_eval: np.ndarray = None) -> np.ndarray:
        """Solve -div(a * grad(u)) = f for given permeability a = u0.

        Parameters
        ----------
        u0 : np.ndarray
            Permeability field a(x,y), shape ``(n_x, n_y)``.
        t_eval : np.ndarray, optional
            Ignored (static PDE).

        Returns
        -------
        np.ndarray
            Pressure field, shape ``(n_x, n_y)``.
        """
        a = u0
        f_val = self.params["forcing"]
        n_x, n_y = a.shape
        dx = self.grid["x"][1] - self.grid["x"][0]
        dy = self.grid["y"][1] - self.grid["y"][0]

        n_dof = n_x * n_y

        def idx(i, j):
            return i * n_y + j

        # Build sparse matrix for -div(a * grad(u)) using harmonic averaging
        rows, cols, vals = [], [], []
        rhs_vec = np.full(n_dof, f_val)

        for i in range(n_x):
            for j in range(n_y):
                n = idx(i, j)
                center_coeff = 0.0

                # x-direction neighbors
                for di, dj in [(-1, 0), (1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n_x:
                        # Harmonic average of permeability at interface
                        a_face = 2 * a[i, j] * a[ni, nj] / (a[i, j] + a[ni, nj] + 1e-30)
                        coeff = a_face / dx**2
                        rows.append(n)
                        cols.append(idx(ni, nj))
                        vals.append(coeff)
                        center_coeff -= coeff
                    else:
                        # Dirichlet u=0 ghost cell: the face term stays on the diagonal.
                        a_face = a[i, j]
                        center_coeff -= a_face / dx**2

                # y-direction neighbors
                for di, dj in [(0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= nj < n_y:
                        a_face = 2 * a[i, j] * a[ni, nj] / (a[i, j] + a[ni, nj] + 1e-30)
                        coeff = a_face / dy**2
                        rows.append(n)
                        cols.append(idx(ni, nj))
                        vals.append(coeff)
                        center_coeff -= coeff
                    else:
                        a_face = a[i, j]
                        center_coeff -= a_face / dy**2

                rows.append(n)
                cols.append(n)
                vals.append(center_coeff)

        A = sp.csr_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
        # Solve A @ u = -f (since we built -div(a grad u) as negative definite)
        pressure = spla.spsolve(-A, rhs_vec)
        return pressure.reshape(n_x, n_y)
