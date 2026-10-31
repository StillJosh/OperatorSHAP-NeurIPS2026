"""Dict-input wrapper of the multiwavelet operator ``MWT1d`` (Gupta et al., NeurIPS 2021).

``python download_mwt.py`` places the upstream ``models.py`` and ``utils.py``
(https://github.com/gaurav71531/mwt-operator) next to this file.
"""

from __future__ import annotations

import torch


class MWTGuptaBackbone(torch.nn.Module):
    """``MWT1d`` on ``{"u0": (B, C, X)}`` batches, returning ``(B, 1, X)``.

    Parameters
    ----------
    net : torch.nn.Module
        ``MWT1d``; takes channel-last ``(B, X, ich)`` and returns ``(B, X)``.
    concat_grid : bool
        Append the grid coordinate as the last input channel (counted in ``ich``).
    grid_range : float
        Upper end of the coordinate channel ``linspace(0, grid_range)``.
    """

    def __init__(self, net: torch.nn.Module, concat_grid: bool, grid_range: float) -> None:
        super().__init__()
        self.grid_range = float(grid_range)
        self.concat_grid = bool(concat_grid)
        self.model = net

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        u0 = data["u0"]
        b, _, x_len = u0.shape
        if self.concat_grid:
            grid = torch.linspace(0.0, self.grid_range, x_len, device=u0.device, dtype=u0.dtype).view(1, 1, x_len).expand(b, 1, x_len)
            u0 = torch.cat([u0, grid], dim=1)
        y = self.model(u0.permute(0, 2, 1).contiguous())
        # MWT1d ends with .squeeze(), which also drops a batch of one.
        return y.reshape(b, 1, x_len)
