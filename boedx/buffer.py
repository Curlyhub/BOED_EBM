"""
Pre-allocated numpy-backed replay buffer.

Design notes
------------
All fields are stored as contiguous numpy arrays of shape
``(capacity, *field_shape)``.  This eliminates the per-sample Python-object
loop that ``np.stack`` over a list-of-dicts incurs on every sample call.

``t_idx`` / ``next_t_idx`` are stored as ``int64`` because PyTorch's
embedding and indexing ops expect long tensors; all other fields are
``float32``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer backed by numpy arrays.

    Functionally equivalent to a list-of-dicts implementation but avoids
    per-sample object allocation and ``np.stack`` overhead, which was the
    dominant CPU bottleneck for the source-location task (bank size H=2401).
    """

    _LONG_KEYS = frozenset({"t_idx", "next_t_idx"})

    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self._size = 0
        self._ptr = 0
        self._arrays: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_arrays(self, item: Dict) -> None:
        for k, v in item.items():
            v_arr = np.asarray(v)
            shape = (self.capacity,) + v_arr.shape
            dtype = np.int64 if k in self._LONG_KEYS else np.float32
            self._arrays[k] = np.zeros(shape, dtype=dtype)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, item: Dict) -> None:
        """Insert one transition.  Initialises storage on the first call."""
        if not self._arrays:
            self._init_arrays(item)
        for k, v in item.items():
            self._arrays[k][self._ptr] = v
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Uniform random sample of *batch_size* transitions.

        Scalars stored as shape ``()`` per sample are unsqueezed to ``(B, 1)``
        to match the shapes expected by the training loop.
        """
        idxs = np.random.randint(0, self._size, size=batch_size)
        out: Dict[str, torch.Tensor] = {}
        for k, arr in self._arrays.items():
            sampled = arr[idxs]
            if k in self._LONG_KEYS:
                t = torch.tensor(sampled, dtype=torch.long, device=device)
            else:
                t = torch.tensor(sampled, dtype=torch.float32, device=device)
            # Scalar fields arrive as (B,); restore (B, 1) for downstream ops
            if t.dim() == 1 and k not in self._LONG_KEYS:
                t = t.unsqueeze(-1)
            out[k] = t
        return out

    def __len__(self) -> int:
        return self._size
