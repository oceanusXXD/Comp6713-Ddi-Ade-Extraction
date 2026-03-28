# 预验证输出目录说明

这个目录用于保存旧版预验证脚本的默认输出。

## 说明

- `src/prevalidation/run_pretest_vllm.py` 在未显式传 `--output_path` 时会默认写到这里。
- 这类结果属于历史工具输出，不是当前主线 benchmark 的一部分。
