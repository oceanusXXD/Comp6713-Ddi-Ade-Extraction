import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def extract_gold_relations(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "").strip()
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return []
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="data/prevalidation/pretest_sample.jsonl")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_path))

    random.seed(args.seed)
    idxs = list(range(len(rows)))
    random.shuffle(idxs)
    idxs = idxs[: args.num_samples]

    output_rows = []
    for i, idx in enumerate(idxs):
        row = rows[idx]
        messages = row.get("messages", [])

        text = extract_user_text(messages)
        gold_relations = extract_gold_relations(messages)

        output_rows.append(
            {
                "sample_id": f"pretest_{i:04d}",
                "text": text,
                "gold_relations": gold_relations,
            }
        )

    write_jsonl(Path(args.output_path), output_rows)

    print(f"Saved {len(output_rows)} samples to {args.output_path}")


if __name__ == "__main__":
    main()
    