"""预览 ChatML 样本结构的辅助脚本。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件。"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_parse_assistant_content(content: str):
    """尽量把 assistant 文本解析成 JSON。"""
    try:
        return json.loads(content)
    except Exception:
        return None


def pretty_print_sample(sample: Dict[str, Any], idx: int) -> None:
    """把单条样本格式化打印出来，方便人工快速检查。"""
    messages = sample.get("messages", [])

    system_msg = ""
    user_msg = ""
    assistant_msg = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_msg = content
        elif role == "user":
            user_msg = content
        elif role == "assistant":
            assistant_msg = content

    parsed_gold = safe_parse_assistant_content(assistant_msg)

    print("=" * 100)
    print(f"SAMPLE #{idx}")
    print("-" * 100)
    print("SYSTEM:")
    print(system_msg[:500] + ("..." if len(system_msg) > 500 else ""))
    print("-" * 100)
    print("USER TEXT:")
    print(user_msg)
    print("-" * 100)
    print("ASSISTANT RAW:")
    print(assistant_msg)
    print("-" * 100)
    print("ASSISTANT PARSED:")
    if parsed_gold is None:
        print("[Failed to parse assistant JSON]")
    else:
        print(json.dumps(parsed_gold, ensure_ascii=False, indent=2))
    print("=" * 100)
    print()


def main() -> None:
    """预览指定文件中的前若干条样本。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to chatml jsonl file")
    parser.add_argument("--num_samples", type=int, default=3, help="How many samples to preview")
    args = parser.parse_args()

    path = Path(args.path)
    rows = read_jsonl(path)

    print(f"Loaded {len(rows)} samples from: {path}")
    print()

    preview_n = min(args.num_samples, len(rows))
    for i in range(preview_n):
        pretty_print_sample(rows[i], i)


if __name__ == "__main__":
    main()
    
