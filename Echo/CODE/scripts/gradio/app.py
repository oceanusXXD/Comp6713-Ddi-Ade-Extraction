"""Gradio demo for direct text inference (English UI).

This demo reuses the same lower-level inference stack as `scripts/inference/predict.py`,
but it does not call the CLI script itself and it does not require a dataset file for
free-form text input.

Inference stack matches `scripts/inference/predict.py`:
- Base model: causal LM (e.g. Qwen3-8B), loaded via `AutoModelForCausalLM.from_pretrained`.
- Optional LoRA: if `adapter_path` is set in YAML, `PeftModel.from_pretrained` wraps the frozen
  base with adapter weights; if `adapter_path` is null, inference is base-only.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import random
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import gradio as gr

from src.inference_backends import (
    generate_predictions,
    load_model_and_tokenizer_transformers,
    load_model_and_tokenizer_vllm,
)
from src.inference_config import load_inference_config
from src.model_utils import resolve_model_source
from src.parser import (
    DatasetExample,
    load_dataset_examples,
    normalize_relation_list,
    relation_set,
)
from src.prompting import load_system_prompt

LOGGER = logging.getLogger(__name__)

RAW_CHAT_MODE = "Free chat"
EXTRACTION_MODE = "ADE/DDI extraction"
DEFAULT_CHAT_SYSTEM_PROMPT = "You are a helpful assistant."

MODEL_PRESETS: List[Tuple[str, str]] = [
    ("Base only (Qwen3-8B)", "configs/infer_gradio_base.yaml"),
    ("+ LoRA (repo adapter)", "configs/infer_gradio_balanced_e3.yaml"),
]

DEMO_CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600&display=swap');
:root {
  --demo-ink: #14213d;
  --demo-muted: #5b6475;
  --demo-panel: #ffffff;
  --demo-panel-soft: #f6f8fb;
  --demo-line: #d8dee8;
  --demo-accent: #c86b2a;
  --demo-accent-deep: #8f4617;
}
.gradio-container {
  font-family: "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
  font-size: 15px;
  -webkit-font-smoothing: antialiased;
  background:
    radial-gradient(circle at top left, rgba(200, 107, 42, 0.10), transparent 24%),
    linear-gradient(180deg, #fbfcfe 0%, #f4f7fb 100%);
}
.gradio-container .gr-form label,
.gradio-container .label-wrap {
  font-weight: 500;
  letter-spacing: 0.02em;
}
.gradio-container .block {
  border-color: var(--demo-line);
}
.gradio-container button.primary {
  background: linear-gradient(135deg, var(--demo-accent) 0%, var(--demo-accent-deep) 100%);
}
.gradio-container .gr-button-secondary {
  border-color: var(--demo-line);
}
.demo-hero {
  padding: 20px 24px;
  margin-bottom: 14px;
  border: 1px solid rgba(200, 107, 42, 0.16);
  border-radius: 20px;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(248, 242, 236, 0.98) 100%);
  box-shadow: 0 18px 50px rgba(20, 33, 61, 0.07);
}
.demo-eyebrow {
  margin: 0 0 8px 0;
  color: var(--demo-accent-deep);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.demo-title {
  margin: 0;
  color: var(--demo-ink);
  font-size: 34px;
  line-height: 1.05;
}
.demo-subtitle {
  margin: 12px 0 0 0;
  max-width: 860px;
  color: var(--demo-muted);
  font-size: 16px;
  line-height: 1.55;
}
.demo-section-note {
  margin: 0 0 10px 0;
  color: var(--demo-muted);
  font-size: 14px;
}
.demo-side-title {
  margin: 0 0 12px 0;
  color: var(--demo-ink);
  font-size: 18px;
  font-weight: 700;
}
"""


@dataclass
class LoadedDemoModel:
    preset_label: str
    config: Dict[str, Any]
    bundle: Any
    tokenizer: Any


_RUNTIME: Optional[LoadedDemoModel] = None


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_example_catalog(limit: int = 128) -> List[Dict[str, Any]]:
    """Build optional example list from test JSONL for quick-fill / gold display."""
    candidates = [
        PROJECT_ROOT.parent
        / "MISC"
        / "data"
        / "processed"
        / "Comp6713-Ddi-Ade-Extraction_latest_raw_clean"
        / "merged_chatml_test.jsonl",
        PROJECT_ROOT.parent / "MISC" / "data" / "merged_chatml_test.jsonl",
    ]
    path = next((p for p in candidates if p.exists()), candidates[0])
    if not path.exists():
        return []
    examples = load_dataset_examples(path, split="test", limit=limit)
    return [
        {
            "id": e.sample_id,
            "text": e.user_text,
            "gold": e.gold_relations,
        }
        for e in examples
    ]


def _prepare_config_for_demo(config_rel: str) -> Dict[str, Any]:
    cfg_path = (PROJECT_ROOT / config_rel).resolve()
    cfg = load_inference_config(cfg_path, validate=False)
    cfg["backend"] = "transformers"
    cfg["data"]["input_path"] = None
    cfg["output"]["predictions_path"] = None
    cfg["output"]["metrics_path"] = None
    cfg["output"]["metrics_json_path"] = None
    cfg["inference"]["batch_size"] = 1
    cfg["inference"]["max_new_tokens"] = min(int(cfg["inference"]["max_new_tokens"]), 256)
    env_base = os.environ.get("COMP6713_BASE_MODEL", "").strip()
    if env_base:
        cfg["allow_remote_model_source"] = True
        cfg["model"]["base_model_name_or_path"] = resolve_model_source(
            env_base,
            allow_remote=True,
        )
    configured_adapter_path = cfg["model"].get("adapter_path")
    resolved_adapter_path, resolution_note = resolve_adapter_for_demo(configured_adapter_path)
    cfg["_demo_config_rel"] = config_rel
    cfg["_demo_adapter_resolution"] = resolution_note
    cfg["_demo_adapter_configured_path"] = configured_adapter_path
    cfg["model"]["adapter_path"] = resolved_adapter_path
    return cfg


def available_model_presets() -> List[Tuple[str, str]]:
    """Keep only runnable presets so the dropdown does not advertise broken adapters."""
    available: List[Tuple[str, str]] = []
    for preset_label, config_rel in MODEL_PRESETS:
        try:
            cfg = _prepare_config_for_demo(config_rel)
        except Exception:
            LOGGER.exception("Skipping preset %s because config preparation failed.", preset_label)
            continue
        adapter_path = cfg["model"].get("adapter_path")
        configured_adapter_path = cfg.get("_demo_adapter_configured_path")
        if configured_adapter_path is not None and adapter_path is None:
            LOGGER.warning(
                "Skipping preset %s because no runnable adapter was found from configured path %s",
                preset_label,
                configured_adapter_path,
            )
            continue
        available.append((preset_label, config_rel))
    return available


def is_peft_adapter_dir(path: Path) -> bool:
    """Minimal PEFT adapter directory check for demo-time path discovery."""
    return path.is_dir() and (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists()


def checkpoint_step(path: Path) -> int:
    """Sort checkpoint-123 style directories by step number."""
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def resolve_adapter_for_demo(adapter_path: Optional[Path]) -> Tuple[Optional[Path], str]:
    """Resolve a runnable adapter path for the Gradio demo."""
    if adapter_path is None:
        return None, "base-only"

    candidate = Path(adapter_path).expanduser()
    if is_peft_adapter_dir(candidate):
        return candidate, "configured adapter path"

    search_root = candidate.parent if candidate.name == "final_adapter" else candidate
    if not search_root.exists():
        return None, f"missing adapter path: {candidate}"

    if is_peft_adapter_dir(search_root):
        return search_root, f"fallback to run directory adapter: {search_root}"

    final_adapter = search_root / "final_adapter"
    if is_peft_adapter_dir(final_adapter):
        return final_adapter, f"fallback to final_adapter under run directory: {final_adapter}"

    checkpoints = sorted(
        [path for path in search_root.iterdir() if path.is_dir() and checkpoint_step(path) >= 0 and is_peft_adapter_dir(path)],
        key=checkpoint_step,
        reverse=True,
    )
    if checkpoints:
        return checkpoints[0], f"fallback to latest checkpoint adapter: {checkpoints[0]}"

    return None, f"no runnable adapter files found under {search_root}"


def default_extraction_prompt() -> str:
    """Default extraction prompt used by the repo when the user leaves system prompt blank."""
    prompt_path = PROJECT_ROOT / "prompts" / "medical_relation_extraction_system_prompt.txt"
    return load_system_prompt(str(prompt_path) if prompt_path.exists() else None)


def resolve_demo_system_prompt(
    task_mode: str,
    system_prompt_override: str,
    cfg: Dict[str, Any],
) -> str:
    """Choose a system prompt without requiring dataset-backed examples."""
    override = (system_prompt_override or "").strip()
    if override:
        return override
    if task_mode == EXTRACTION_MODE:
        prompt_path = str(cfg["system_prompt_path"]) if cfg.get("system_prompt_path") is not None else None
        return load_system_prompt(prompt_path)
    return DEFAULT_CHAT_SYSTEM_PROMPT


def _display_name_from_model_source(value: Any) -> str:
    """Front-end friendly model name without exposing full local paths."""
    text = str(value or "").strip()
    if not text:
        return "Unknown"
    if "/" in text or "\\" in text:
        return Path(text).name or text
    return text


def _adapter_resolution_summary(resolution_note: str, adapter_present: bool) -> str:
    """Short adapter source summary for UI display."""
    if not adapter_present:
        return "None"
    if resolution_note == "configured adapter path":
        return "Direct"
    if resolution_note.startswith("fallback to latest checkpoint adapter"):
        return "Recovered from latest checkpoint"
    if resolution_note.startswith("fallback to final_adapter under run directory"):
        return "Recovered from run directory"
    if resolution_note.startswith("fallback to run directory adapter"):
        return "Recovered from run directory"
    return "Resolved"


def _config_profile_label(config_rel: str) -> str:
    """Readable config profile name for the compact model card."""
    stem = Path(config_rel).stem
    stem = stem.removeprefix("infer_")
    stem = stem.removeprefix("gradio_")
    token_map = {
        "ade": "ADE",
        "base": "Base",
        "chat": "Chat",
        "ddi": "DDI",
        "e3": "E3",
        "e4": "E4",
        "final": "Final",
        "lora": "LoRA",
        "qwen3": "Qwen3",
        "8b": "8B",
    }
    tokens = [token_map.get(token.lower(), token.capitalize()) for token in stem.split("_") if token]
    return " ".join(tokens) if tokens else "Default"


def format_model_status_md(cfg: Dict[str, Any], preset_label: str, config_rel: str) -> str:
    adapter = cfg["model"].get("adapter_path")
    base = _display_name_from_model_source(cfg["model"]["base_model_name_or_path"])
    backend = str(cfg.get("backend", "transformers")).title()
    adapter_resolution = str(cfg.get("_demo_adapter_resolution", ""))
    mode_line = "LoRA adapter attached" if adapter else "Base-only"
    lines = [
        "### Model & config",
        f"- **Preset**: {preset_label}",
        f"- **Profile**: {_config_profile_label(config_rel)}",
        f"- **Runtime**: {backend}",
        f"- **Model family**: {base}",
        f"- **Mode**: {mode_line}",
        f"- **Adapter source**: {_adapter_resolution_summary(adapter_resolution, adapter is not None)}",
        f"- **max_new_tokens**: {cfg['inference']['max_new_tokens']} | **max_input_length**: {cfg['inference']['max_input_length']}",
    ]
    return "\n".join(lines)


def unload_runtime() -> None:
    global _RUNTIME
    if _RUNTIME is None:
        return
    try:
        backend = str(_RUNTIME.config.get("backend", "transformers")).lower()
        if backend == "vllm":
            llm = _RUNTIME.bundle[0]
            del llm
        else:
            model = _RUNTIME.bundle
            del model
    except Exception:
        LOGGER.exception("unload_runtime cleanup failed")
    _RUNTIME = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_runtime(preset_label: str) -> Tuple[str, str]:
    """Load or switch weights. Returns (markdown status, error string or empty)."""
    global _RUNTIME
    mapping = dict(MODEL_PRESETS)
    config_rel = mapping.get(preset_label)
    if not config_rel:
        return "### Load status\nUnknown preset.", "unknown_preset"

    unload_runtime()
    try:
        cfg = _prepare_config_for_demo(config_rel)
        backend = str(cfg.get("backend", "transformers")).lower()
        if backend == "vllm":
            llm, tokenizer, sp, lr = load_model_and_tokenizer_vllm(cfg)
            _RUNTIME = LoadedDemoModel(preset_label, cfg, (llm, sp, lr), tokenizer)
        else:
            model, tokenizer = load_model_and_tokenizer_transformers(cfg)
            _RUNTIME = LoadedDemoModel(preset_label, cfg, model, tokenizer)
        body = format_model_status_md(cfg, preset_label, config_rel)
        return body + "\n\n**Load status**: ready.", ""
    except Exception as exc:
        LOGGER.exception("Model load failed for %s", preset_label)
        unload_runtime()
        return f"### Load failed\n```\n{exc}\n```", str(exc)


def ensure_runtime(preset_label: str) -> str:
    if _RUNTIME is not None and _RUNTIME.preset_label == preset_label:
        return ""
    _, err = load_runtime(preset_label)
    return err


def gold_markdown(gold: Sequence[Dict[str, Any]]) -> str:
    if not gold:
        return "### Ground truth\nNo reference labels (custom input or no example selected)."
    rows = []
    for item in normalize_relation_list(gold):
        rows.append(
            f"| {item['head_entity']} | {item['tail_entity']} | `{item['relation_type']}` |"
        )
    header = "| Head | Tail | Type |\n| --- | --- | --- |"
    return "### Ground truth\n" + header + "\n" + "\n".join(rows)


def highlight_text(text: str, relations: Sequence[Dict[str, str]]) -> str:
    """Highlight head/tail spans (first match, case-insensitive)."""
    if not text:
        return "<p><em>(empty)</em></p>"
    entities: List[Tuple[int, int, str]] = []
    lower_full = text.casefold()
    for rel in relations:
        for key in ("head_entity", "tail_entity"):
            ent = str(rel.get(key, "")).strip()
            if not ent:
                continue
            pos = lower_full.find(ent.casefold())
            if pos >= 0:
                entities.append((pos, pos + len(ent), ent))
    entities.sort(key=lambda x: x[0])
    merged: List[Tuple[int, int, str]] = []
    for span in entities:
        if not merged or span[0] >= merged[-1][1]:
            merged.append(span)
    escaped = html.escape(text)
    if not merged:
        return f"<div style='line-height:1.6'>{escaped}</div>"
    pieces: List[str] = []
    cursor = 0
    for start, end, ent in merged:
        pieces.append(html.escape(text[cursor:start]))
        piece = html.escape(text[start:end])
        pieces.append(f"<mark style='background:#ffe08a;padding:0 2px' title='{html.escape(ent)}'>{piece}</mark>")
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "<div style='line-height:1.6'>" + "".join(pieces) + "</div>"


def predictions_table(
    predicted: Sequence[Dict[str, Any]],
    gold: Sequence[Dict[str, Any]],
) -> Tuple[List[List[str]], str]:
    gold_s = relation_set(gold)
    rows_out: List[List[str]] = []
    json_lines: List[str] = []
    for rel in normalize_relation_list(predicted):
        key = (rel["relation_type"], rel["head_entity"].casefold(), rel["tail_entity"].casefold())
        flag = "OK vs gold" if key in gold_s else "Not in gold"
        if not gold_s:
            flag = "- (no gold)"
        rows_out.append([rel["head_entity"], rel["tail_entity"], rel["relation_type"], flag])
        json_lines.append(json.dumps(rel, ensure_ascii=False))
    if not rows_out:
        rows_out = [["(none)", "(none)", "(none)", "-"]]
    pretty = "[\n  " + ",\n  ".join(json_lines) + "\n]" if json_lines else "[]"
    return rows_out, pretty


def error_attribution_markdown(
    gold: Sequence[Dict[str, Any]],
    predicted: Sequence[Dict[str, Any]],
    parse_status: str,
    parse_reason: Optional[str],
    raw_output: str,
) -> str:
    """Qualitative error buckets: parse failure / FN / FP / label confusion."""
    lines: List[str] = ["### Error analysis", ""]

    if parse_status != "parsed":
        lines.append(f"- **Parse failure** (`{parse_status}`): {parse_reason or 'unknown'}")
        lines.append("- **Hint**: model may not have returned a JSON array, or generation was truncated.")
        if raw_output.strip():
            snippet = raw_output[:2000].replace("```", "`\u200b``")
            lines.append("")
            lines.append("Raw output excerpt:\n\n```text\n" + snippet + "\n```")
        return "\n".join(lines)

    g_rel = normalize_relation_list(gold)
    p_rel = normalize_relation_list(predicted)
    g_set = relation_set(g_rel)
    p_set = relation_set(p_rel)
    tp = g_set & p_set
    fp = p_set - g_set
    fn = g_set - p_set

    lines.append(
        f"- **Set overlap** (same as `parser.evaluate_prediction_rows`): TP={len(tp)}, FP={len(fp)}, FN={len(fn)}"
    )

    if not g_set:
        lines.append("- **Empty gold**: only evaluate FP / hallucination unless you selected a labeled example.")
        if fp:
            lines.append(f"- **Extra predictions vs empty gold**: {len(fp)} flags.")
        elif not p_set:
            lines.append("- **Empty prediction**: may match empty gold.")
        return "\n".join(lines)

    if not fp and not fn:
        lines.append("- **Sample-level**: predicted relation set matches gold (exact set match).")
        return "\n".join(lines)

    if fn:
        lines.append("- **False negatives** (in gold, missing in pred):")
        for lbl, h, t in sorted(fn):
            lines.append(f"  - `{lbl}` | {h} -> {t}")

    if fp:
        lines.append("- **False positives** (in pred, not in gold):")
        for lbl, h, t in sorted(fp):
            lines.append(f"  - `{lbl}` | {h} -> {t}")

    gold_map: Dict[Tuple[str, str], str] = {}
    pred_map: Dict[Tuple[str, str], str] = {}
    for lbl, h, t in g_set:
        gold_map[(h, t)] = lbl
    for lbl, h, t in p_set:
        pred_map[(h, t)] = lbl
    for span, gl in gold_map.items():
        if span in pred_map and pred_map[span] != gl:
            lines.append(
                f"- **Label confusion** for `{span[0]}` / `{span[1]}`: gold `{gl}`, pred `{pred_map[span]}`."
            )

    if parse_status == "parsed" and not p_set and g_set:
        lines.append("- **Missed extraction**: parse OK but empty relation list while gold is non-empty.")

    return "\n".join(lines)


def run_one_inference(
    user_text: str,
    preset_label: str,
    task_mode: str,
    system_prompt_override: str,
    example_value: Any,
    catalog: List[Dict[str, Any]],
) -> Tuple[Any, ...]:
    """Run one forward pass; return Gradio update tuple."""
    user_text = (user_text or "").strip()
    if not user_text:
        return (
            "",
            "<p><em>Enter medical text to run inference.</em></p>",
            gr.update(value=[]),
            "",
            gold_markdown([]),
            "### Timing\n(not run)",
            gr.update(value=""),
            "### Error analysis\n(not run)",
        )

    err = ensure_runtime(preset_label)
    if err:
        return (
            "",
            f"<p>Load failed: {html.escape(err)}</p>",
            gr.update(value=[]),
            "",
            gold_markdown([]),
            "### Timing\n(load failed)",
            gr.update(value=""),
            f"### Error analysis\nLoad failed: {html.escape(err)}",
        )

    assert _RUNTIME is not None
    cfg = _RUNTIME.config
    system_prompt = resolve_demo_system_prompt(task_mode, system_prompt_override, cfg)

    gold: List[Dict[str, str]] = []
    if task_mode == EXTRACTION_MODE and isinstance(example_value, int) and 0 <= example_value < len(catalog):
        gold = list(catalog[example_value]["gold"])

    example = DatasetExample(
        sample_id="gradio_single",
        split="gradio",
        system_prompt=system_prompt,
        user_text=user_text,
        gold_relations=gold,
    )

    t0 = time.perf_counter()
    rows = generate_predictions(_RUNTIME.bundle, _RUNTIME.tokenizer, [example], cfg)
    elapsed = time.perf_counter() - t0

    row = rows[0]
    predicted = row.get("predicted_relations") or []
    raw_out = str(row.get("raw_output", ""))
    parse_status = str(row.get("parse_status", ""))
    parse_reason = row.get("parse_failure_reason")
    assistant_response = raw_out

    if task_mode == EXTRACTION_MODE:
        viz = highlight_text(user_text, predicted)
        table, json_pretty = predictions_table(predicted, gold)
        stats_lines = [
            "### Timing",
            f"- **Wall time**: {elapsed * 1000.0:.1f} ms",
            f"- **Mode**: `{task_mode}`",
            f"- **Parse status**: `{parse_status}`",
        ]
        err_md = error_attribution_markdown(gold, predicted, parse_status, parse_reason, raw_out)
    else:
        viz = highlight_text(user_text, [])
        table = [["(n/a)", "(n/a)", "(n/a)", "raw chat mode"]]
        json_pretty = ""
        stats_lines = [
            "### Timing",
            f"- **Wall time**: {elapsed * 1000.0:.1f} ms",
            f"- **Mode**: `{task_mode}`",
            "- **Structured parse**: disabled in raw chat mode",
        ]
        err_md = (
            "### Response mode\n"
            "Raw chat mode does not require a dataset or gold labels. "
            "The textbox above is the direct model output."
        )
    stats_md = "\n".join(stats_lines)
    return (
        assistant_response,
        viz,
        gr.update(value=table),
        json_pretty,
        gold_markdown(gold),
        stats_md,
        gr.update(value=raw_out),
        err_md,
    )


def build_demo(catalog: List[Dict[str, Any]]) -> gr.Blocks:
    example_choices: List[Tuple[str, int]] = [("(custom input)", -1)]
    for i, item in enumerate(catalog):
        snippet = item["text"].replace("\n", " ")
        if len(snippet) > 72:
            snippet = snippet[:72] + "..."
        example_choices.append((f"Example {i + 1}: {snippet}", i))

    default_preset = MODEL_PRESETS[0][0]

    with gr.Blocks(title="ADE/DDI Extraction Demo") as demo:
        gr.Markdown(
            """
            <div class="demo-hero">
              <p class="demo-eyebrow">COMP6713 | Medical NLP</p>
              <h1 class="demo-title">Medical Relation Extraction Studio</h1>
              <p class="demo-subtitle">
                Explore base and LoRA checkpoints for ADE and DDI extraction from free-form medical text.
                Use the examples as quick starters, then inspect the structured output, highlighted spans,
                and raw generation in one place.
              </p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=7, min_width=680):
                gr.Markdown(
                    '<p class="demo-side-title">Input</p>'
                    '<p class="demo-section-note">Choose an example or paste your own text below.</p>'
                )

                if catalog:
                    with gr.Accordion("Examples", open=True):
                        example_dropdown = gr.Dropdown(
                            label="Quick examples",
                            choices=[c[0] for c in example_choices],
                            value=example_choices[0][0],
                        )
                        rand_btn = gr.Button("Random example")
                else:
                    example_dropdown = gr.Dropdown(
                        label="Quick examples",
                        choices=[c[0] for c in example_choices],
                        value=example_choices[0][0],
                        visible=False,
                    )
                    rand_btn = gr.Button("Random example", visible=False)

                input_box = gr.Textbox(
                    label="Medical text",
                    placeholder="Paste English text mentioning drugs, adverse events, or interactions...",
                    lines=9,
                )

                assistant_box = gr.Textbox(
                    label="Assistant response",
                    placeholder="Model output will appear here.",
                    lines=10,
                )

                with gr.Tabs():
                    with gr.Tab("Source highlights"):
                        viz_html = gr.HTML()
                    with gr.Tab("Structured parse"):
                        pred_table = gr.Dataframe(
                            headers=["Head", "Tail", "Type", "vs gold"],
                            row_count=(8, "dynamic"),
                            column_count=(4, "fixed"),
                            label="Predictions (normalized)",
                        )
                        pred_json = gr.Textbox(label="JSON", lines=14)
                    with gr.Tab("Gold & timing"):
                        gold_md = gr.Markdown("### Ground truth\nSelect a dataset example to show reference relations.")
                        stats_md = gr.Markdown("### Timing\n(not run)")
                    with gr.Tab("Debug"):
                        raw_box = gr.Textbox(label="Raw model output", lines=16)
                        error_md = gr.Markdown("### Error analysis\nRuns after inference.")

            with gr.Column(scale=4, min_width=320):
                gr.Markdown('<p class="demo-side-title">Controls</p>')

                model_select = gr.Dropdown(
                    label="Model preset (loads on change)",
                    choices=[x[0] for x in MODEL_PRESETS],
                    value=default_preset,
                    elem_id="demo_model_preset",
                )
                task_mode = gr.Radio(
                    label="Run mode",
                    choices=[RAW_CHAT_MODE, EXTRACTION_MODE],
                    value=EXTRACTION_MODE,
                )
                model_status = gr.Markdown(
                    "### Model & config\n"
                    "- **Preset**: loading selected preset\n"
                    "- **Profile**: preparing model summary\n"
                    "- **Runtime**: Transformers\n"
                    "- **Status**: loading on page open"
                )

                with gr.Group(visible=False) as system_prompt_group:
                    system_prompt_box = gr.Textbox(
                        label="System prompt (optional)",
                        placeholder=(
                            "Leave blank to use the mode default: "
                            "'You are a helpful assistant.' for Free chat, "
                            "or the repo extraction prompt for ADE/DDI extraction."
                        ),
                        lines=4,
                    )

                run_btn = gr.Button("Run inference", variant="primary", interactive=False)
                with gr.Row():
                    clear_btn = gr.Button("Clear input")
                    reset_btn = gr.Button("Reset UI")

        label_to_index = {label: idx for label, idx in example_choices}

        def on_model_change(preset: str):
            md, err = load_runtime(preset)
            if err:
                return md
            return md

        def on_example_pick(choice: str):
            idx = label_to_index.get(choice, -1)
            if idx < 0:
                return gr.update(value=""), gold_markdown([]), gr.update(interactive=False)
            row = catalog[idx]
            return gr.update(value=row["text"]), gold_markdown(row["gold"]), gr.update(interactive=True)

        def sync_run_button_state(text: str):
            return gr.update(interactive=bool((text or "").strip()))

        def sync_system_prompt_visibility(mode: str):
            return gr.update(visible=(mode == RAW_CHAT_MODE))

        def do_infer(text: str, preset: str, mode: str, system_prompt_text: str, choice: str):
            idx = label_to_index.get(choice, -1)
            assistant_text, viz, table_upd, json_pretty, gmd, stats, raw_upd, err_m = run_one_inference(
                text,
                preset,
                mode,
                system_prompt_text,
                idx,
                catalog,
            )
            return assistant_text, viz, table_upd, json_pretty, gmd, stats, raw_upd, err_m

        def random_example():
            if not catalog:
                return (
                    gr.update(),
                    gr.update(),
                    "### Ground truth\nNo `merged_chatml_test.jsonl` found under expected paths.",
                    gr.update(interactive=False),
                )
            idx = random.randint(0, len(catalog) - 1)
            label = example_choices[idx + 1][0]
            text_upd, gold_part, run_btn_state = on_example_pick(label)
            return gr.update(value=label), text_upd, gold_part, run_btn_state

        def clear_inputs():
            return (
                gr.update(value=""),
                gr.update(value=example_choices[0][0]),
                gr.update(value=""),
                gold_markdown([]),
                gr.update(interactive=False),
            )

        def reset_all():
            empty = (
                "",
                "<p><em>Reset.</em></p>",
                gr.update(value=[["(none)", "(none)", "(none)", "-"]]),
                "",
                gold_markdown([]),
                "### Timing\n(not run)",
                gr.update(value=""),
                "### Error analysis\n(not run)",
            )
            return (
                gr.update(value=""),
                gr.update(value=example_choices[0][0]),
                gr.update(value=""),
                gr.update(interactive=False),
            ) + empty

        model_select.change(on_model_change, inputs=[model_select], outputs=[model_status])

        example_dropdown.change(
            on_example_pick,
            inputs=[example_dropdown],
            outputs=[input_box, gold_md, run_btn],
        )

        rand_btn.click(random_example, outputs=[example_dropdown, input_box, gold_md, run_btn])

        input_box.input(sync_run_button_state, inputs=[input_box], outputs=[run_btn])
        input_box.change(sync_run_button_state, inputs=[input_box], outputs=[run_btn])
        task_mode.change(sync_system_prompt_visibility, inputs=[task_mode], outputs=[system_prompt_group])

        clear_btn.click(
            clear_inputs,
            outputs=[input_box, example_dropdown, system_prompt_box, gold_md, run_btn],
        )

        reset_btn.click(
            reset_all,
            outputs=[
                input_box,
                example_dropdown,
                system_prompt_box,
                run_btn,
                assistant_box,
                viz_html,
                pred_table,
                pred_json,
                gold_md,
                stats_md,
                raw_box,
                error_md,
            ],
        )

        run_btn.click(
            do_infer,
            inputs=[input_box, model_select, task_mode, system_prompt_box, example_dropdown],
            outputs=[assistant_box, viz_html, pred_table, pred_json, gold_md, stats_md, raw_box, error_md],
        )

        demo.load(on_model_change, inputs=[model_select], outputs=[model_status])

    return demo


def _first_free_listen_port(*, host: str, start_port: int, max_attempts: int) -> int:
    """First TCP port in [start_port, start_port + max_attempts) that accepts bind."""
    bind_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
            except OSError:
                continue
            return port
    raise OSError(
        f"No free port in {start_port}-{start_port + max_attempts - 1}; "
        "free the listener or use --port / --port-retries."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio ADE/DDI relation demo (English UI).")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument(
        "--port-retries",
        type=int,
        default=24,
        help="If the start port is busy, try this many successive ports (default: 24).",
    )
    p.add_argument("--share", action="store_true", help="Create a temporary public gradio.app link.")
    p.add_argument("--example-limit", type=int, default=128, help="Max examples to load from JSONL.")
    return p.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    global MODEL_PRESETS
    MODEL_PRESETS = available_model_presets()
    if not MODEL_PRESETS:
        raise RuntimeError(
            "No runnable model presets found for the Gradio demo.\n"
            "Download the base model first with:\n"
            "bash scripts/setup/download_base_model.sh"
        )
    catalog = build_example_catalog(limit=args.example_limit)

    demo = build_demo(catalog)
    demo.queue()

    port = _first_free_listen_port(
        host=args.host,
        start_port=args.port,
        max_attempts=max(1, args.port_retries),
    )
    if port != args.port:
        LOGGER.warning("Port %s in use; using %s instead.", args.port, port)

    launch_kw: Dict[str, Any] = dict(
        server_name=args.host,
        server_port=port,
        share=args.share,
        css=DEMO_CUSTOM_CSS,
    )
    try:
        demo.launch(theme=gr.themes.Soft(), **launch_kw)
    except TypeError:
        demo.launch(**launch_kw)


if __name__ == "__main__":
    main()
