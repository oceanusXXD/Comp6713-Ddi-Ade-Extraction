import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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

def build_messages(system_prompt: str, user_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/merged_chatml_test.jsonl")
    parser.add_argument("--prompt_path", type=str, default="src/prevalidation/prompt.txt")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    prompt_path = Path(args.prompt_path)
    output_path = Path(args.output_path)

    rows = read_jsonl(input_path)
    if args.limit > 0:
        rows = rows[:args.limit]
    
    system_prompt = load_prompt(prompt_path)
    print(f"Loaded {len(rows)} samples from: {input_path}")
    print(f"Loading model from: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    print("Model loaded. Preparing prompts...")
    prompts = []
    sample_texts = []
    sample_golds = []

    for row in rows:
        messages = row.get("messages", [])
        text = extract_user_text(messages)
        gold_relations = extract_gold_relations(messages)
        
        sample_texts.append(text)
        sample_golds.append(gold_relations)
        
        msgs = build_messages(system_prompt, text)
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)

    print("Start vLLM batched inference...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        raw_output = output.outputs[0].text.strip()
        parsed_output = extract_json_list(raw_output)

        results.append({
            "sample_id": f"test_{i:04d}",
            "text": sample_texts[i],
            "gold_relations": sample_golds[i],
            "raw_output": raw_output,
            "parsed_output": parsed_output,
            "json_valid": parsed_output is not None,
        })
        
        if (i + 1) % 50 == 0 or (i + 1) == len(outputs):
            print(f"Processed {i + 1}/{len(outputs)} samples.")

    write_jsonl(output_path, results)
    print(f"Saved predictions to: {output_path}")

if __name__ == "__main__":
    main()
