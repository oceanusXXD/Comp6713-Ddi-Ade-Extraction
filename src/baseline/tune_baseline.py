"""规则基线自动调参脚本。"""

import argparse
import json
import itertools
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import spacy

from src.baseline.rule_config import DEFAULT_CONFIG
from src.baseline.run_baseline import (
    read_jsonl,
    extract_user_text,
    extract_gold_relations,
    predict_relations,
    load_lexicon,
    norm,
)


def relations_to_set(relations: List[Dict[str, Any]]) -> Set[Tuple[str, str, str]]:
    """把关系列表转换成集合。"""
    result = set()
    if not isinstance(relations, list):
        return result

    for r in relations:
        if not isinstance(r, dict):
            continue

        head = norm(r.get("head_entity", ""))
        tail = norm(r.get("tail_entity", ""))
        rel_type = norm(r.get("relation_type", ""))

        if head and tail and rel_type:
            result.add((head, tail, rel_type))

    return result


def evaluate_rows(
    rows: List[Dict[str, Any]],
    nlp,
    drug_lexicon: List[str],
    effect_lexicon: List[str],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """评估某个规则配置在一批样本上的表现。"""
    tp = fp = fn = 0
    exact_match_count = 0

    for row in rows:
        messages = row.get("messages", [])
        text = extract_user_text(messages)
        gold_relations = extract_gold_relations(messages)

        pred_relations = predict_relations(
            text=text,
            nlp=nlp,
            drug_lexicon=drug_lexicon,
            effect_lexicon=effect_lexicon,
            config=config,
        )

        gold_set = relations_to_set(gold_relations)
        pred_set = relations_to_set(pred_relations)

        if gold_set == pred_set:
            exact_match_count += 1

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    total = len(rows)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    exact_match = exact_match_count / total if total > 0 else 0.0

    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    """遍历规则搜索空间并保存最佳配置。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/merged_chatml_validation.jsonl")
    parser.add_argument("--drug_lexicon_path", type=str, default="resources/baseline/drug_lexicon.json")
    parser.add_argument("--effect_lexicon_path", type=str, default="resources/baseline/effect_lexicon.json")
    parser.add_argument("--output_path", type=str, default="outputs/baseline/best_config.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_path))
    if args.limit > 0:
        rows = rows[: args.limit]

    drug_lexicon = load_lexicon(Path(args.drug_lexicon_path))
    effect_lexicon = load_lexicon(Path(args.effect_lexicon_path))

    print(f"Loaded {len(rows)} validation rows from: {args.input_path}")
    print(f"Loaded {len(drug_lexicon)} drug lexicon entries")
    print(f"Loaded {len(effect_lexicon)} effect lexicon entries")

    nlp = spacy.load("en_core_web_sm")

    # ========= 搜索空间 =========
    min_effect_len_options = [2, 3]
    min_drug_len_options = [2, 3]
    max_drugs_per_sent_options = [4, 6, 8]

    ade_trigger_extra_options = [
        [],
        ["triggered by"],
        ["secondary to"],
        ["triggered by", "secondary to"],
    ]

    ddi_int_extra_options = [
        [],
        ["combined with"],
    ]

    best_config = None
    best_metrics = None
    best_f1 = -1.0

    trial_id = 0

    for (
        min_effect_len,
        min_drug_len,
        max_drugs_per_sent,
        ade_extra,
        ddi_int_extra,
    ) in itertools.product(
        min_effect_len_options,
        min_drug_len_options,
        max_drugs_per_sent_options,
        ade_trigger_extra_options,
        ddi_int_extra_options,
    ):
        trial_id += 1

        config = dict(DEFAULT_CONFIG)
        config["MIN_EFFECT_LEN"] = min_effect_len
        config["MIN_DRUG_LEN"] = min_drug_len
        config["MAX_DRUGS_PER_SENT_FOR_DDI"] = max_drugs_per_sent
        config["ADE_TRIGGERS"] = list(DEFAULT_CONFIG["ADE_TRIGGERS"]) + list(ade_extra)
        config["DDI_INT_TRIGGERS"] = list(DEFAULT_CONFIG["DDI_INT_TRIGGERS"]) + list(ddi_int_extra)

        metrics = evaluate_rows(
            rows=rows,
            nlp=nlp,
            drug_lexicon=drug_lexicon,
            effect_lexicon=effect_lexicon,
            config=config,
        )

        print(
            f"[Trial {trial_id:03d}] "
            f"MIN_EFFECT_LEN={min_effect_len}, "
            f"MIN_DRUG_LEN={min_drug_len}, "
            f"MAX_DRUGS_PER_SENT_FOR_DDI={max_drugs_per_sent}, "
            f"ADE_EXTRA={ade_extra}, "
            f"DDI_INT_EXTRA={ddi_int_extra} "
            f"=> F1={metrics['f1']:.4f}, "
            f"P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, "
            f"EM={metrics['exact_match']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_config = config
            best_metrics = metrics

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best_config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Best config saved to:", out_path)
    print("Best metrics:")
    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))
    print("Best config:")
    print(json.dumps(best_config, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
