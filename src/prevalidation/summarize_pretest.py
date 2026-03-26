import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def relations_to_set(relations: List[Dict[str, Any]]) -> Set[Tuple[str, str, str]]:
    result = set()
    for r in relations or []:
        head = normalize_text(r.get("head_entity", ""))
        tail = normalize_text(r.get("tail_entity", ""))
        rel_type = normalize_text(r.get("relation_type", ""))
        if head and tail and rel_type:
            result.add((head, tail, rel_type))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.pred_path))

    total_samples = len(rows)
    json_valid_count = sum(1 for r in rows if r.get("json_valid"))

    exact_match_count = 0
    tp = fp = fn = 0

    for row in rows:
        gold_set = relations_to_set(row.get("gold_relations", []))
        pred_set = relations_to_set(row.get("parsed_output", []) if row.get("parsed_output") is not None else [])

        if gold_set == pred_set:
            exact_match_count += 1

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match_acc = exact_match_count / total_samples if total_samples > 0 else 0.0
    json_valid_rate = json_valid_count / total_samples if total_samples > 0 else 0.0

    print("=" * 80)
    print(f"Prediction file: {args.pred_path}")
    print(f"Total samples: {total_samples}")
    print(f"JSON valid: {json_valid_count}/{total_samples} = {json_valid_rate:.2%}")
    print(f"Exact match accuracy: {exact_match_count}/{total_samples} = {exact_match_acc:.2%}")
    print(f"TP={tp}, FP={fp}, FN={fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("=" * 80)

    print("\nExamples of mismatched cases:")
    shown = 0
    for row in rows:
        gold_set = relations_to_set(row.get("gold_relations", []))
        pred_set = relations_to_set(row.get("parsed_output", []) if row.get("parsed_output") is not None else [])

        if gold_set != pred_set:
            print("-" * 80)
            print(f"sample_id: {row['sample_id']}")
            print("TEXT:")
            print(row["text"])
            print("\nGOLD:")
            print(json.dumps(row.get("gold_relations", []), ensure_ascii=False, indent=2))
            print("\nPRED:")
            print(json.dumps(row.get("parsed_output", []), ensure_ascii=False, indent=2))
            shown += 1
            if shown >= 5:
                break


if __name__ == "__main__":
    main()
    