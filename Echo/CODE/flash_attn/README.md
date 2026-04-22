# flash_attn Compatibility Layer

This folder provides a minimal local compatibility shim for the `apply_rotary` symbol expected by some model-loading paths.

It is not a full replacement for the upstream `flash_attn` package. It only keeps the submission package runnable when the project code imports `flash_attn.ops.triton.rotary.apply_rotary`.
