# Source Modules

Core Python modules used by the retained training, inference, and evaluation pipeline.

## Files

- `data_utils.py`: dataset loading, normalization, and supervised example preparation
- `inference_backends.py`: `transformers` and `vllm` inference backends
- `inference_config.py`: inference config loading and validation
- `model_utils.py`: training config loading, model/tokenizer setup, and adapter export helpers
- `observability.py`: JSON and JSONL logging helpers
- `parser.py`: prediction parsing and metric computation
- `prompting.py`: prompt loading and chat-template helpers
- `runtime_env.py`: lightweight virtual-environment checks
