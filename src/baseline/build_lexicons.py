"""从训练集提取规则基线词典。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

# 作用是从训练集提取：
#   1. 药物词典
#   2. effect 词典

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件。"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_gold_relations(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 ChatML 中提取 assistant gold 关系列表。"""
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
    """转小写并折叠空白，便于去重。"""
    return " ".join(str(s).strip().lower().split())


def main() -> None:
    """扫描训练集并生成药物词典与 effect 词典。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/merged_chatml_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="resources/baseline")
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "drug_lexicon.json").open("w", encoding="utf-8") as f:
        json.dump(sorted(drug_lexicon), f, ensure_ascii=False, indent=2)

    with (out_dir / "effect_lexicon.json").open("w", encoding="utf-8") as f:
        json.dump(sorted(effect_lexicon), f, ensure_ascii=False, indent=2)

    print(f"Saved drug lexicon: {out_dir / 'drug_lexicon.json'} ({len(drug_lexicon)} items)")
    print(f"Saved effect lexicon: {out_dir / 'effect_lexicon.json'} ({len(effect_lexicon)} items)")


if __name__ == "__main__":
    main()
    
