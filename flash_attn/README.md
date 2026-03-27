# `flash_attn` 本地兼容层说明

这个目录不是完整的 `flash_attn` 第三方库源码，而是仓库为了兼容某些模型依赖而保留的一个最小本地实现。

## 文件说明

- `__init__.py`
  对外暴露 `apply_rotary` 接口。
- `ops/__init__.py`
  `ops` 子模块初始化文件。
- `ops/triton/__init__.py`
  `triton` 子模块初始化文件。
- `ops/triton/rotary.py`
  本地实现的 `apply_rotary`，用于提供 rotary embedding 相关计算接口。

## 为什么需要这个目录

有些模型或推理路径会尝试导入 `flash_attn.ops.triton.rotary.apply_rotary`。这个兼容层的作用是：

- 让仓库在没有完整外部 `flash_attn` 包的情况下，仍能满足所需接口
- 避免因为缺少导入而直接中断训练或推理

如果后续切换成完整外部依赖，这个目录是否继续保留，可以再根据实际运行环境决定。
