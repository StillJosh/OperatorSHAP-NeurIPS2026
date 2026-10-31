"""Finite volume solver for 2D shallow water equations.

First-order FV with HLL Riemann solver. Time integration via SSP-RK3.
"""

import numpy as np

from .base import PDESolver


class ShallowWaterSolver2D(PDESolver):
    """2D shallow water equations with reflective boundary conditions.

    h_t + (h*u)_x + (h*v)_y = 0
    (h*u)_t + (h*u^2 + g*h^2/2)_x + (h*u*v)_y = 0
    (h*v)_t + (h*u*v)_x + (h*v^2 + g*h^2/2)_y = 0

    Parameters
    ----------
    grid : dict
        Must contain ``"x"`` and ``"y"`` arrays.
    params : dict
        Must contain ``"g"`` (gravity). Optional ``"cfl"`` (default 0.4).
    """

    def rhs(self, t, u_flat):
        raise NotImplementedError("FV solver uses custom time stepping.")

    def _hll_flux_x(self, qL: np.ndarray, qR: np.ndarray) -> np.ndarray:
        """HLL flux in x-direction. q = [h, hu, hv]."""
        g = self.params["g"]
        hL, huL, hvL = qL[0], qL[1], qL[2]
        hR, huR, hvR = qR[0], qR[1], qR[2]

        uL = np.where(hL > 1e-6, huL / hL, 0.0)
        uR = np.where(hR > 1e-6, huR / hR, 0.0)
        cL = np.sqrt(g * np.maximum(hL, 0))
        cR = np.sqrt(g * np.maximum(hR, 0))

        sL = np.minimum(uL - cL, uR - cR)
        sR = np.maximum(uL + cL, uR + cR)

        fL = np.array([huL, huL * uL + 0.5 * g * hL**2, huL * np.where(hL > 1e-6, hvL / hL, 0.0)])
        fR = np.array([huR, huR * uR + 0.5 * g * hR**2, huR * np.where(hR > 1e-6, hvR / hR, 0.0)])

        flux = np.where(
            sL >= 0,
            fL,
            np.where(
                sR <= 0,
                fR,
                (sR * fL - sL * fR + sL * sR * (qR - qL)) / (sR - sL + 1e-30),
            ),
        )
        return flux

    def _hll_flux_y(self, qL: np.ndarray, qR: np.ndarray) -> np.ndarray:
        """HLL flux in y-direction. Swap u/v roles."""
        g = self.params["g"]
        hL, huL, hvL = qL[0], qL[1], qL[2]
        hR, huR, hvR = qR[0], qR[1], qR[2]

        vL = np.where(hL > 1e-6, hvL / hL, 0.0)
        vR = np.where(hR > 1e-6, hvR / hR, 0.0)
        cL = np.sqrt(g * np.maximum(hL, 0))
        cR = np.sqrt(g * np.maximum(hR, 0))

        sL = np.minimum(vL - cL, vR - cR)
        sR = np.maximum(vL + cL, vR + cR)

        gL = np.array([hvL, hvL * np.where(hL > 1e-6, huL / hL, 0.0), hvL * vL + 0.5 * g * hL**2])
        gR = np.array([hvR, hvR * np.where(hR > 1e-6, huR / hR, 0.0), hvR * vR + 0.5 * g * hR**2])

        flux = np.where(
            sL >= 0,
            gL,
            np.where(
                sR <= 0,
                gR,
                (sR * gL - sL * gR + sL * sR * (qR - qL)) / (sR - sL + 1e-30),
            ),
        )
        return flux

    def _compute_rhs_fv(self, q: np.ndarray) -> np.ndarray:
        """Compute -div(F) using first-order FV + HLL."""
        dx = self.grid["x"][1] - self.grid["x"][0]
        dy = self.grid["y"][1] - self.grid["y"][0]
        _, n_x, n_y = q.shape

        # Pad with 1 ghost cell per side (reflective BC)
        qp = np.zeros((3, n_x + 2, n_y + 2))
        qp[:, 1:-1, 1:-1] = q
        # x-boundaries: reflect and negate hu
        qp[:, 0, 1:-1] = q[:, 0, :]
        qp[1, 0, 1:-1] *= -1
        qp[:, -1, 1:-1] = q[:, -1, :]
        qp[1, -1, 1:-1] *= -1
        # y-boundaries: reflect and negate hv
        qp[:, 1:-1, 0] = q[:, :, 0]
        qp[2, 1:-1, 0] *= -1
        qp[:, 1:-1, -1] = q[:, :, -1]
        qp[2, 1:-1, -1] *= -1

        # First-order: left/right states at x-interfaces (n_x + 1 interfaces)
        qL_x = qp[:, :-1, 1:-1]   # (3, n_x+1, n_y)
        qR_x = qp[:, 1:, 1:-1]    # (3, n_x+1, n_y)
        fx = self._hll_flux_x(qL_x, qR_x)

        # First-order: left/right states at y-interfaces (n_y + 1 interfaces)
        qL_y = qp[:, 1:-1, :-1]   # (3, n_x, n_y+1)
        qR_y = qp[:, 1:-1, 1:]    # (3, n_x, n_y+1)
        gy = self._hll_flux_y(qL_y, qR_y)

        rhs = np.zeros_like(q)
        rhs -= (fx[:, 1:, :] - fx[:, :-1, :]) / dx
        rhs -= (gy[:, :, 1:] - gy[:, :, :-1]) / dy
        return rhs

    def solve(self, u0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        """Solve shallow water from initial height field u0.

        Parameters
        ----------
        u0 : np.ndarray
            Initial water height, shape ``(n_x, n_y)``. Velocity is zero initially.
        t_eval : np.ndarray
            Output times.

        Returns
        -------
        np.ndarray
            Solution shape ``(len(t_eval), 3, n_x, n_y)``.
            Channel 0 = h, 1 = hu, 2 = hv.
        """
        g = self.params["g"]
        cfl = self.params.get("cfl", 0.4)
        dx = self.grid["x"][1] - self.grid["x"][0]
        dy = self.grid["y"][1] - self.grid["y"][0]
        n_x, n_y = u0.shape

        # Conservative variables: [h, hu, hv]
        q = np.zeros((3, n_x, n_y))
        q[0] = u0  # height
        # hu, hv = 0 initially

        solution = np.zeros((len(t_eval), 3, n_x, n_y))
        t = t_eval[0]
        solution[0] = q
        save_idx = 1

        while save_idx < len(t_eval):
            # Adaptive time step
            h = np.maximum(q[0], 1e-6)
            max_speed = np.max(np.abs(q[1] / h) + np.sqrt(g * h)) + np.max(
                np.abs(q[2] / h) + np.sqrt(g * h)
            )
            dt = cfl * min(dx, dy) / (max_speed + 1e-30)
            dt = min(dt, t_eval[save_idx] - t)

            # SSP-RK3
            rhs1 = self._compute_rhs_fv(q)
            q1 = q + dt * rhs1

            rhs2 = self._compute_rhs_fv(q1)
            q2 = 0.75 * q + 0.25 * (q1 + dt * rhs2)

            rhs3 = self._compute_rhs_fv(q2)
            q = (1.0 / 3.0) * q + (2.0 / 3.0) * (q2 + dt * rhs3)

            # Ensure non-negative height
            q[0] = np.maximum(q[0], 0.0)

            t += dt

            if abs(t - t_eval[save_idx]) < 1e-12:
                solution[save_idx] = q.copy()
                save_idx += 1

        return solution
