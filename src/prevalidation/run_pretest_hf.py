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


def load_model_and_tokenizer(model_path_or_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_or_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    return tokenizer, model


def build_messages(system_prompt: str, user_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def run_one_sample(
    tokenizer,
    model,
    system_prompt: str,
    user_text: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    messages = build_messages(system_prompt, user_text)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    do_sample = temperature > 0.0
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    outputs = model.generate(**inputs, **generate_kwargs)

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    return decoded.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/merged_chatml_test.jsonl",
        help="Path to the cleaned test jsonl file",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="src/prevalidation/prompt.txt",
        help="Path to the shared system prompt",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model repo id or local model path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Where to save prediction results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="How many samples to run",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature; recommended 0.0 for extraction tasks",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max number of generated tokens",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    prompt_path = Path(args.prompt_path)
    output_path = Path(args.output_path)

    rows = read_jsonl(input_path)[: args.limit]
    system_prompt = load_prompt(prompt_path)

    print(f"Loaded {len(rows)} samples from: {input_path}")
    print(f"Loading model from: {args.model_name}")

    tokenizer, model = load_model_and_tokenizer(args.model_name)

    print("Model loaded. Start inference...")

    results = []
    for i, row in enumerate(rows):
        messages = row.get("messages", [])
        text = extract_user_text(messages)
        gold_relations = extract_gold_relations(messages)

        raw_output = run_one_sample(
            tokenizer=tokenizer,
            model=model,
            system_prompt=system_prompt,
            user_text=text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        parsed_output = extract_json_list(raw_output)

        results.append(
            {
                "sample_id": f"test_{i:04d}",
                "text": text,
                "gold_relations": gold_relations,
                "raw_output": raw_output,
                "parsed_output": parsed_output,
                "json_valid": parsed_output is not None,
            }
        )

        print(
            f"Done {i + 1}/{len(rows)} | sample_id=test_{i:04d} | json_valid={parsed_output is not None}"
        )

    write_jsonl(output_path, results)
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
