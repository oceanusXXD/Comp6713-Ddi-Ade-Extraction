import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.pred_path))
    total = len(rows)
    valid = sum(1 for r in rows if r.get("json_valid"))
    invalid = total - valid

    print("=" * 80)
    print(f"Prediction file: {args.pred_path}")
    print(f"Total samples: {total}")
    print(f"Valid JSON: {valid}")
    print(f"Invalid JSON: {invalid}")
    print(f"JSON validity rate: {valid / total:.2%}" if total else "No rows")
    print("=" * 80)

    print("\nExamples:")
    for row in rows[:3]:
        print("-" * 80)
        print(f"sample_id: {row['sample_id']}")
        print("TEXT:")
        print(row["text"])
        print("\nGOLD:")
        print(json.dumps(row["gold_relations"], ensure_ascii=False, indent=2))
        print("\nPRED:")
        print(json.dumps(row["parsed_output"], ensure_ascii=False, indent=2) if row["parsed_output"] is not None else row["raw_output"])


if __name__ == "__main__":
    main()
    