"""
Monotonic Alignment Search (MAS) for VITS-style training.

Given the per-(text, frame) prior log-likelihood `neg_cent`, MAS finds the
optimal *monotonic, surjective* alignment between text tokens and audio frames
via dynamic programming.

This module provides TWO backends:
  1. maximum_path_gpu  — Pure PyTorch, stays on GPU, zero graph breaks for
                         torch.compile.  Vectorises the x-axis and batch
                         dimensions at each DP step.
  2. maximum_path_cpu  — Numba-JIT on CPU (original VITS approach).  Used as
                         fallback when CUDA is unavailable.

The public entry point `maximum_path` dispatches automatically.
"""

import torch

# ---------------------------------------------------------------------------
# Backend 1: Pure-PyTorch GPU  (no numpy, no graph breaks)
# ---------------------------------------------------------------------------

_NEGINF = -1e9


@torch.no_grad()
def maximum_path_gpu(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    GPU-native MAS using vectorised dynamic programming.

    neg_cent : [b, t_x, t_y]  prior log-likelihood (text tokens × audio frames).
    mask     : [b, t_x, t_y]  rectangular validity mask.
    Returns  : [b, t_x, t_y]  hard alignment path (same device / dtype).

    The DP recurrence (for each frame y, vectorised across x and batch):
        value[x, y] = neg_cent[x, y] + max(
            value[x,   y-1],       # extend current token to this frame
            value[x-1, y-1]        # start a new token at this frame
        )
    with boundary / reachability constraints applied via masks.
    """
    device, dtype = neg_cent.device, neg_cent.dtype
    b, t_x, t_y = neg_cent.shape

    # Work in float32 for numerical stability during DP accumulation.
    value = (neg_cent * mask).detach().float()

    # Valid lengths from the rectangular mask.
    t_xs = mask[:, :, 0].sum(1).long()          # [b]
    t_ys = mask[:, 0, :].sum(1).long()          # [b]

    # Indices used repeatedly.
    x_idx     = torch.arange(t_x, device=device).unsqueeze(0)   # [1, t_x]
    batch_idx = torch.arange(b,   device=device)                # [b]

    # --- y = 0 initialisation: only x = 0 is reachable. ---
    value[:, 1:, 0] = _NEGINF

    # --- Forward DP (sequential on y, vectorised on x & batch) ---
    for y in range(1, t_y):
        # v_cur: score if we *extend* the current token to frame y.
        #        = value[x, y-1]   but  -∞  when x == y  (the monotonicity
        #        constraint: if x == y every token so far has exactly one
        #        frame, so there is nothing to extend from).
        v_cur = value[:, :, y - 1].clone()
        if y < t_x:
            v_cur[:, y] = _NEGINF

        # v_prev: score if we *start* token x at frame y.
        #         = value[x-1, y-1]   but  -∞  when x == 0  (no x = -1).
        v_prev = torch.full_like(v_cur, _NEGINF)
        v_prev[:, 1:] = value[:, :-1, y - 1]

        best = torch.max(v_cur, v_prev)

        # Reachability mask — only cells where the path can still finish
        # in time are valid:
        #   x_start = max(0, t_x + y - t_y)
        #   x_end   = min(t_x, y + 1)
        x_starts = (t_xs + y - t_ys).clamp(min=0).unsqueeze(1)  # [b, 1]
        x_ends   = t_xs.clamp(max=y + 1).unsqueeze(1)           # [b, 1]
        valid    = (x_idx >= x_starts) & (x_idx < x_ends)       # [b, t_x]

        value[:, :, y] = value[:, :, y] + torch.where(valid, best, _NEGINF)

    # --- Backtrack (sequential on y, vectorised across batch) ---
    path  = torch.zeros(b, t_x, t_y, device=device, dtype=dtype)
    index = (t_xs - 1).clamp(min=0)                             # [b]

    for y in range(t_y - 1, -1, -1):
        active = y < t_ys                                        # [b] bool
        path[batch_idx[active], index[active], y] = 1.0

        if y > 0:
            cur_val  = value[batch_idx, index, y - 1]
            prev_idx = (index - 1).clamp(min=0)
            prev_val = value[batch_idx, prev_idx, y - 1]

            should_dec = active & (index > 0) & (
                (index == y) | (cur_val < prev_val)
            )
            index = index - should_dec.long()

    return path


# ---------------------------------------------------------------------------
# Backend 2: Numba CPU  (fallback when no GPU)
# ---------------------------------------------------------------------------

try:
    import numpy as np
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:                       # pragma: no cover
    _HAVE_NUMBA = False
    np = None

    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _wrap

    def prange(*args):
        return range(*args)


@njit(cache=True)
def _maximum_path_each(path, value, t_x, t_y):
    _MAX_NEG = -1e9
    for y in range(t_y):
        x_start = max(0, t_x + y - t_y)
        x_end   = min(t_x, y + 1)
        for x in range(x_start, x_end):
            v_cur  = _MAX_NEG if x == y else value[x, y - 1]
            v_prev = (0.0 if y == 0 else _MAX_NEG) if x == 0 else value[x - 1, y - 1]
            value[x, y] = value[x, y] + max(v_cur, v_prev)

    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index -= 1


@njit(parallel=True, cache=True)
def _maximum_path_batch(paths, values, t_xs, t_ys):
    for b in prange(paths.shape[0]):
        _maximum_path_each(paths[b], values[b], t_xs[b], t_ys[b])


def maximum_path_cpu(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Numba-JIT MAS on CPU. Used as fallback."""
    device, dtype = neg_cent.device, neg_cent.dtype
    value = (neg_cent * mask).detach().cpu().numpy().astype(np.float32)
    path  = np.zeros_like(value, dtype=np.int32)
    t_xs  = mask[:, :, 0].sum(1).detach().cpu().numpy().astype(np.int32)
    t_ys  = mask[:, 0, :].sum(1).detach().cpu().numpy().astype(np.int32)

    _maximum_path_batch(path, value, t_xs, t_ys)

    return torch.from_numpy(path).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Public entry point — auto-dispatch
# ---------------------------------------------------------------------------

def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Find the optimal monotonic alignment via dynamic programming.

    Dispatches to the GPU backend when the input lives on CUDA;
    falls back to the Numba CPU backend otherwise.
    """
    if neg_cent.is_cuda:
        return maximum_path_gpu(neg_cent, mask)
    if _HAVE_NUMBA:
        return maximum_path_cpu(neg_cent, mask)
    # Ultimate fallback: run GPU version on CPU (slower but correct).
    return maximum_path_gpu(neg_cent, mask)
