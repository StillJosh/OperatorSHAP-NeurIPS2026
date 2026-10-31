"""Resolution-aware batch sampling: batches are ``(resolution, sample_idx)`` tuples the datasets stride by."""

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler, Subset


class ResolutionBatchSampler(Sampler[list[tuple[int, int]]]):
    """Batches of ``(resolution, sample_idx)`` with every sample fixed to one resolution.

    The samples are split into ``len(resolutions)`` contiguous groups of
    near-equal size; a sample keeps its resolution across epochs. With
    ``shuffle`` the order within each group and the order of the batches
    are drawn per epoch.

    Parameters
    ----------
    n_samples : int
    resolutions : list[int]
    batch_size : int
    shuffle : bool
    drop_last : bool
        Drop an incomplete tail batch of each group.
    """

    def __init__(self, n_samples: int, resolutions: list[int], batch_size: int, shuffle: bool = True, drop_last: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        base, extra = divmod(n_samples, len(resolutions))
        self.groups: list[tuple[int, list[int]]] = []
        cursor = 0
        for j, r in enumerate(resolutions):
            size = base + (1 if j < extra else 0)
            self.groups.append((r, list(range(cursor, cursor + size))))
            cursor += size

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        batches: list[list[tuple[int, int]]] = []
        for r, idx_list in self.groups:
            order = [idx_list[p] for p in torch.randperm(len(idx_list)).tolist()] if self.shuffle else idx_list
            for s in range(0, len(order), self.batch_size):
                chunk = order[s : s + self.batch_size]
                if not (self.drop_last and len(chunk) < self.batch_size):
                    batches.append([(r, i) for i in chunk])
        if self.shuffle:
            batches = [batches[p] for p in torch.randperm(len(batches)).tolist()]
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(idx_list) // self.batch_size for _, idx_list in self.groups)
        return sum((len(idx_list) + self.batch_size - 1) // self.batch_size for _, idx_list in self.groups)


class MultiResSubset(Subset):
    """``Subset`` forwarding ``(resolution, idx)`` tuples as ``(resolution, indices[idx])``."""

    def __getitem__(self, idx):
        r, i = idx
        return self.dataset[(r, self.indices[i])]

    def __getitems__(self, indices):
        return [self[idx] for idx in indices]
