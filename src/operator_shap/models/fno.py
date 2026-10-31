"""Dict-input wrapper of a grid network such as ``neuralop.models.FNO`` or ``neuralop.models.local_no.LocalNO``."""

import torch


class FNOModel(torch.nn.Module):
    """``net`` on ``{"u0": (B, C, *spatial)}`` batches; ``default_in_shape`` of a LocalNO must match the largest input grid."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.model = net

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(data["u0"])
