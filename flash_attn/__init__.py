"""本地 `flash_attn` 兼容层导出。"""

from flash_attn.ops.triton.rotary import apply_rotary

__all__ = ["apply_rotary"]
