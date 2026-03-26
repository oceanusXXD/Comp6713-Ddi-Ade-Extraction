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
    parser = argparse.ArgumentParser(description="Analyze ADE/DDI ChatML dataset statistics.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_qwen3_8b_lora.yaml",
        help="Training config used to resolve tokenizer and max_seq_length.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Dataset JSONL path to analyze.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Where to save the computed statistics JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample cap for faster analysis.",
    )
    return parser.parse_args()


def main() -> None:
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
    print(f"Saved dataset statistics to: {Path(args.output_path).expanduser().resolve()}")


if __name__ == "__main__":
    main()
