import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def extract_json_list(text: str) -> Optional[List[Dict[str, Any]]]:
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and start < end:
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return None

    return None


def run_model(model_name: str, system_prompt: str, user_text: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    return decoded.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/prevalidation/pretest_sample.jsonl")
    parser.add_argument("--prompt_path", type=str, default="src/prevalidation/prompt.txt")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_path))[: args.limit]
    system_prompt = load_prompt(Path(args.prompt_path))

    results = []
    for row in rows:
        raw_output = run_model(
            model_name=args.model_name,
            system_prompt=system_prompt,
            user_text=row["text"],
        )

        parsed_output = extract_json_list(raw_output)

        results.append(
            {
                "sample_id": row["sample_id"],
                "text": row["text"],
                "gold_relations": row["gold_relations"],
                "raw_output": raw_output,
                "parsed_output": parsed_output,
                "json_valid": parsed_output is not None,
            }
        )

        print(f"Done: {row['sample_id']} | json_valid={parsed_output is not None}")

    write_jsonl(Path(args.output_path), results)
    print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()
    