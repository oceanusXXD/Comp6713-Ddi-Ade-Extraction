import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import spacy

from src.baseline.rule_config import DEFAULT_CONFIG


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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_lexicon(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def sentence_contains_any(text: str, triggers: List[str]) -> bool:
    text_norm = norm(text)
    return any(t in text_norm for t in triggers)


def dedup_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str, str]] = set()
    deduped = []
    for rel in relations:
        key = (
            norm(rel.get("head_entity", "")),
            norm(rel.get("tail_entity", "")),
            norm(rel.get("relation_type", "")),
        )
        if all(key) and key not in seen:
            seen.add(key)
            deduped.append(rel)
    return deduped


def find_mentions_with_positions(text: str, lexicon: List[str], min_len: int) -> List[Dict[str, Any]]:
    """
    Match simple strings in normalized text and track approximate token offsets
    """
    text_norm = norm(text)
    text_tokens = text_norm.split()

    mentions = []
    seen = set()

    for item in lexicon:
        item_norm = norm(item)
        if len(item_norm) < min_len:
            continue
        if not item_norm:
            continue

        item_tokens = item_norm.split()
        n = len(item_tokens)
        if n == 0:
            continue

        for i in range(len(text_tokens) - n + 1):
            span_tokens = text_tokens[i : i + n]
            if span_tokens == item_tokens:
                key = (item_norm, i, i + n - 1)
                if key not in seen:
                    seen.add(key)
                    mentions.append(
                        {
                            "text": item_norm,
                            "start_token": i,
                            "end_token": i + n - 1,
                        }
                    )

    mentions.sort(key=lambda x: (x["start_token"], -(x["end_token"] - x["start_token"])))
    return mentions


def token_distance(m1: Dict[str, Any], m2: Dict[str, Any]) -> int:
    """
    Minimum token distance between mentions; 0 if overlapping
    """
    if m1["end_token"] < m2["start_token"]:
        return m2["start_token"] - m1["end_token"]
    if m2["end_token"] < m1["start_token"]:
        return m1["start_token"] - m2["end_token"]
    return 0


def choose_nearest_drug(effect_mention: Dict[str, Any], drug_mentions: List[Dict[str, Any]], max_distance: int):
    candidates = []
    for drug in drug_mentions:
        dist = token_distance(effect_mention, drug)
        if dist <= max_distance:
            candidates.append((dist, drug["start_token"], drug))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def choose_adjacent_drug_pairs(drug_mentions: List[Dict[str, Any]], max_distance: int) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    只取相邻 drug mention 对，避免全组合爆炸。
    """
    if len(drug_mentions) < 2:
        return []

    sorted_mentions = sorted(drug_mentions, key=lambda x: x["start_token"])
    pairs = []

    for i in range(len(sorted_mentions) - 1):
        d1 = sorted_mentions[i]
        d2 = sorted_mentions[i + 1]
        dist = token_distance(d1, d2)
        if dist <= max_distance and d1["text"] != d2["text"]:
            pairs.append((d1, d2))

    return pairs


def predict_relations(
    text: str,
    nlp,
    drug_lexicon: List[str],
    effect_lexicon: List[str],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    ade_triggers = config["ADE_TRIGGERS"]
    ddi_trigger_groups = {
        "DDI-advice": config["DDI_ADVICE_TRIGGERS"],
        "DDI-effect": config["DDI_EFFECT_TRIGGERS"],
        "DDI-mechanism": config["DDI_MECHANISM_TRIGGERS"],
        "DDI-int": config["DDI_INT_TRIGGERS"],
    }
    min_effect_len = config["MIN_EFFECT_LEN"]
    min_drug_len = config["MIN_DRUG_LEN"]
    max_drugs_per_sent_for_ddi = config["MAX_DRUGS_PER_SENT_FOR_DDI"]
    max_ade_token_distance = config.get("MAX_ADE_TOKEN_DISTANCE", 12)
    max_ddi_token_distance = config.get("MAX_DDI_TOKEN_DISTANCE", 10)

    doc = nlp(text)
    preds: List[Dict[str, Any]] = []

    for sent in doc.sents:
        sent_text = sent.text

        drug_mentions = find_mentions_with_positions(sent_text, drug_lexicon, min_drug_len)
        effect_mentions = find_mentions_with_positions(sent_text, effect_lexicon, min_effect_len)

        # ===== ADE rules =====
        if drug_mentions and effect_mentions and sentence_contains_any(sent_text, ade_triggers):
            for effect in effect_mentions:
                nearest_drug = choose_nearest_drug(
                    effect_mention=effect,
                    drug_mentions=drug_mentions,
                    max_distance=max_ade_token_distance,
                )
                if nearest_drug is not None:
                    preds.append(
                        {
                            "head_entity": nearest_drug["text"],
                            "tail_entity": effect["text"],
                            "relation_type": "ADE",
                        }
                    )

        # ===== DDI rules =====
        if 2 <= len(drug_mentions) <= max_drugs_per_sent_for_ddi:
            ddi_type = None

            # priority：advice -> mechanism -> effect -> int
            ddi_priority = ["DDI-advice", "DDI-mechanism", "DDI-effect", "DDI-int"]
            for rel_type in ddi_priority:
                triggers = ddi_trigger_groups[rel_type]
                if sentence_contains_any(sent_text, triggers):
                    ddi_type = rel_type
                    break

            if ddi_type is not None:
                ddi_pairs = choose_adjacent_drug_pairs(
                    drug_mentions=drug_mentions,
                    max_distance=max_ddi_token_distance,
                )
                for d1, d2 in ddi_pairs:
                    preds.append(
                        {
                            "head_entity": d1["text"],
                            "tail_entity": d2["text"],
                            "relation_type": ddi_type,
                        }
                    )

    return dedup_relations(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--drug_lexicon_path", type=str, default="resources/baseline/drug_lexicon.json")
    parser.add_argument("--effect_lexicon_path", type=str, default="resources/baseline/effect_lexicon.json")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_path))
    if args.limit > 0:
        rows = rows[: args.limit]

    drug_lexicon = load_lexicon(Path(args.drug_lexicon_path))
    effect_lexicon = load_lexicon(Path(args.effect_lexicon_path))

    config = DEFAULT_CONFIG if not args.config_path else load_json(Path(args.config_path))
    if isinstance(config, dict) and "best_config" in config:
        config = config["best_config"]

    nlp = spacy.load("en_core_web_sm")

    results = []
    for i, row in enumerate(rows):
        messages = row.get("messages", [])
        text = extract_user_text(messages)
        gold_relations = extract_gold_relations(messages)

        parsed_output = predict_relations(
            text=text,
            nlp=nlp,
            drug_lexicon=drug_lexicon,
            effect_lexicon=effect_lexicon,
            config=config,
        )

        results.append(
            {
                "sample_id": f"sample_{i:04d}",
                "text": text,
                "gold_relations": gold_relations,
                "raw_output": json.dumps(parsed_output, ensure_ascii=False),
                "parsed_output": parsed_output,
                "json_valid": True,
            }
        )

        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            print(f"Processed {i + 1}/{len(rows)}")

    write_jsonl(Path(args.output_path), results)
    print(f"Saved baseline predictions to: {args.output_path}")


if __name__ == "__main__":
    main()
