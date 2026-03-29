"""数据集统计脚本。

这个脚本通常用于训练前或数据整理后，快速输出某个 JSONL 文件的长度、
token 分布和标签统计，结果会写成 JSON 供后续报告复用。
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import compute_dataset_statistics
from src.model_utils import load_tokenizer, load_training_config
from src.observability import write_json
from src.prompting import load_system_prompt


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="统计 ADE/DDI ChatML 数据集信息。")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen3_8b_lora_ddi_ade_final.yaml",
        help="用于解析 tokenizer 与 max_seq_length 的训练配置文件。",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="待统计的数据集 JSONL 路径。",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="统计结果 JSON 的输出路径。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="可选样本上限，用于加快统计速度。",
    )
    return parser.parse_args()


def main() -> None:
    """加载配置与 tokenizer，然后对指定数据集做统计。"""
    args = parse_args()
    config = load_training_config(args.config)
    tokenizer = load_tokenizer(config)
    system_prompt = load_system_prompt(
        str(config["system_prompt_path"]) if config.get("system_prompt_path") is not None else None
    )
    stats = compute_dataset_statistics(
        Path(args.input_path).expanduser().resolve(),
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_length=config["max_seq_length"],
        enable_thinking=config.get("enable_thinking"),
        limit=args.limit,
    )
    write_json(Path(args.output_path).expanduser().resolve(), stats)
    print(f"数据集统计结果已保存到：{Path(args.output_path).expanduser().resolve()}")


if __name__ == "__main__":
    main()
