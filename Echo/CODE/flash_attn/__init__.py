"""Local compatibility exports for the minimal ``flash_attn`` shim."""

from flash_attn.ops.triton.rotary import apply_rotary

__all__ = ["apply_rotary"]
