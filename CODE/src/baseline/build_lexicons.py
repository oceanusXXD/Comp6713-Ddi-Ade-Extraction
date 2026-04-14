import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def load_drugbank_names(path: Path) -> Set[str]:
    names: Set[str] = set()
    if not path.exists():
        return names

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            name = norm(line)
            if name:
                names.add(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/merged_chatml_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="resources/baseline")
    parser.add_argument("--drugbank_path", type=str, default="")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_path))

    drug_lexicon: Set[str] = set()
    effect_lexicon: Set[str] = set()

    for row in rows:
        messages = row.get("messages", [])
        gold_relations = extract_gold_relations(messages)

        for rel in gold_relations:
            head = norm(rel.get("head_entity", ""))
            tail = norm(rel.get("tail_entity", ""))
            rel_type = rel.get("relation_type", "")

            if head:
                drug_lexicon.add(head)

            if rel_type == "ADE" and tail:
                effect_lexicon.add(tail)
            elif rel_type.startswith("DDI") and tail:
                drug_lexicon.add(tail)

    train_drug_count = len(drug_lexicon)

    drugbank_count = 0
    if args.drugbank_path:
        drugbank_names = load_drugbank_names(Path(args.drugbank_path))
        drugbank_count = len(drugbank_names)
        drug_lexicon.update(drugbank_names)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "drug_lexicon.json").open("w", encoding="utf-8") as f:
        json.dump(sorted(drug_lexicon), f, ensure_ascii=False, indent=2)

    with (out_dir / "effect_lexicon.json").open("w", encoding="utf-8") as f:
        json.dump(sorted(effect_lexicon), f, ensure_ascii=False, indent=2)

    print(f"Train-derived drug lexicon size: {train_drug_count}")
    print(f"DrugBank lexicon size: {drugbank_count}")
    print(f"Merged drug lexicon size: {len(drug_lexicon)}")
    print(f"Effect lexicon size: {len(effect_lexicon)}")
    print(f"Saved drug lexicon: {out_dir / 'drug_lexicon.json'}")
    print(f"Saved effect lexicon: {out_dir / 'effect_lexicon.json'}")


if __name__ == "__main__":
    main()
    