from __future__ import annotations

"""Minimal rotary embedding compatibility implementation."""

from typing import Optional

import torch


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> torch.Tensor:
    """Apply rotary position embedding rotation to an input tensor."""
    del seqlen_offsets, cu_seqlens, max_seqlen

    output = x if inplace else x.clone()

    cos = cos.unsqueeze(-2).to(output.dtype)
    sin = sin.unsqueeze(-2).to(output.dtype)

    if interleaved:
        x1 = output[..., ::2]
        x2 = output[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        rotated = torch.stack((o1, o2), dim=-1).flatten(-2)
    else:
        x1, x2 = torch.chunk(output, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        rotated = torch.cat((o1, o2), dim=-1)

    if inplace:
        output.copy_(rotated)
        return output
    return rotated
