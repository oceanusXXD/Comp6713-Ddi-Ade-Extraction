"""Gradio demo for ADE/DDI relation extraction (English UI).

Inference stack matches `scripts/inference/predict.py`:
- **Base model**: causal LM (e.g. Qwen3-8B), loaded via `AutoModelForCausalLM.from_pretrained`.
- **Optional LoRA**: if `adapter_path` is set in YAML, `PeftModel.from_pretrained` wraps the frozen
  base with trainable adapter weights; if `adapter_path` is null, inference is **base-only**—same code path,
  no separate “download LoRA as a full model” step.

Run from repo root with conda env (e.g. `comp6713`). Base resolution: `../models/Qwen3-8B`, in-repo
`models/Qwen3-8B`, or with `allow_remote_model_source: true` fallback to `Qwen/Qwen3-8B`. Override with
`COMP6713_BASE_MODEL` (local path or HF id).
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import random
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

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

# (dropdown label, path to infer YAML). Base-only first; LoRA presets attach an optional PEFT adapter.
MODEL_PRESETS: List[Tuple[str, str]] = [
    ("Base only (Qwen3-8B)", "configs/infer_gradio_base.yaml"),
    ("+ LoRA: EXP-04", "configs/infer_exp04.yaml"),
    ("+ LoRA: Final", "configs/infer_gradio_final_aug_e4.yaml"),
    ("+ LoRA: Balanced E3", "configs/infer_gradio_balanced_e3.yaml"),
]

DEMO_CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600&display=swap');
.gradio-container {
  font-family: "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
  font-size: 15px;
  -webkit-font-smoothing: antialiased;
}
.gradio-container .gr-form label,
.gradio-container .label-wrap {
  font-weight: 500;
  letter-spacing: 0.02em;
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
    """Build example list from default test JSONL for dropdown / gold display."""
    candidates = [
        PROJECT_ROOT / "data" / "merged_chatml_test.jsonl",
        PROJECT_ROOT / "data" / "processed" / "Comp6713-Ddi-Ade-Extraction_latest_raw_clean" / "merged_chatml_test.jsonl",
        PROJECT_ROOT / "data" / "processed" / "Comp6713-Ddi-Ade-Extraction_final_augment" / "merged_chatml_test.jsonl",
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
    cfg["inference"]["batch_size"] = 1
    env_base = os.environ.get("COMP6713_BASE_MODEL", "").strip()
    if env_base:
        cfg["allow_remote_model_source"] = True
        cfg["model"]["base_model_name_or_path"] = resolve_model_source(
            env_base,
            allow_remote=True,
        )
    return cfg


def format_model_status_md(cfg: Dict[str, Any], preset_label: str, config_rel: str) -> str:
    adapter = cfg["model"].get("adapter_path")
    base = cfg["model"]["base_model_name_or_path"]
    backend = str(cfg.get("backend", "transformers"))
    adapter_line = f"`{adapter}`" if adapter else "*none — base-only inference*"
    lines = [
        "### Model & config",
        f"- **Preset**: {preset_label}",
        f"- **Config file**: `{config_rel}`",
        f"- **Backend**: `{backend}`",
        f"- **Base checkpoint**: `{base}`",
        f"- **PEFT adapter (optional LoRA)**: {adapter_line}",
        f"- **max_new_tokens**: {cfg['inference']['max_new_tokens']} · **max_input_length**: {cfg['inference']['max_input_length']}",
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
            flag = "— (no gold)"
        rows_out.append([rel["head_entity"], rel["tail_entity"], rel["relation_type"], flag])
        json_lines.append(json.dumps(rel, ensure_ascii=False))
    if not rows_out:
        rows_out = [["(none)", "(none)", "(none)", "—"]]
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
            lines.append(f"  - `{lbl}` · {h} → {t}")

    if fp:
        lines.append("- **False positives** (in pred, not in gold):")
        for lbl, h, t in sorted(fp):
            lines.append(f"  - `{lbl}` · {h} → {t}")

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
    example_value: Any,
    catalog: List[Dict[str, Any]],
) -> Tuple[Any, ...]:
    """Run one forward pass; return Gradio update tuple."""
    user_text = (user_text or "").strip()
    if not user_text:
        return (
            "<p><em>Enter medical text to run inference.</em></p>",
            gr.update(value=[]),
            "",
            "### Timing\n(not run)",
            gr.update(value=""),
            "### Error analysis\n(not run)",
        )

    err = ensure_runtime(preset_label)
    if err:
        return (
            f"<p>Load failed: {html.escape(err)}</p>",
            gr.update(value=[]),
            "",
            "### Timing\n(load failed)",
            gr.update(value=""),
            f"### Error analysis\nLoad failed: {html.escape(err)}",
        )

    assert _RUNTIME is not None
    cfg = _RUNTIME.config
    system_prompt = load_system_prompt(
        str(cfg["system_prompt_path"]) if cfg.get("system_prompt_path") is not None else None
    )

    gold: List[Dict[str, str]] = []
    if isinstance(example_value, int) and example_value >= 0 and example_value < len(catalog):
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

    viz = highlight_text(user_text, predicted)
    table, json_pretty = predictions_table(predicted, gold)
    stats_md = "\n".join(
        [
            "### Timing",
            f"- **Wall time**: {elapsed * 1000.0:.1f} ms",
            f"- **Parse status**: `{parse_status}`",
        ]
    )
    err_md = error_attribution_markdown(gold, predicted, parse_status, parse_reason, raw_out)
    return (
        viz,
        gr.update(value=table),
        json_pretty,
        stats_md,
        gr.update(value=raw_out),
        err_md,
    )


def build_demo(catalog: List[Dict[str, Any]]) -> gr.Blocks:
    example_choices: List[Tuple[str, int]] = [("(custom input)", -1)]
    for i, item in enumerate(catalog):
        snippet = item["text"].replace("\n", " ")
        if len(snippet) > 72:
            snippet = snippet[:72] + "…"
        example_choices.append((f"Example {i + 1}: {snippet}", i))

    default_preset = MODEL_PRESETS[0][0]

    with gr.Blocks(title="ADE/DDI Extraction Demo", css=DEMO_CUSTOM_CSS) as demo:
        gr.Markdown(
            "## ADE / DDI relation extraction\n"
            "Enter English medical text. The **base** causal LM runs always; an optional **PEFT (LoRA) adapter** "
            "is loaded only when you pick a `+ LoRA:*` preset. Same code path as `scripts/inference/predict.py`."
        )

        model_select = gr.Dropdown(
            label="Model preset (loads on change)",
            choices=[x[0] for x in MODEL_PRESETS],
            value=default_preset,
            elem_id="demo_model_preset",
        )
        model_status = gr.Markdown(
            "### Model & config\nOpen the app or change preset to load weights. Start with **Base only** to verify the foundation model."
        )

        with gr.Row():
            example_dropdown = gr.Dropdown(
                label="Examples",
                choices=[c[0] for c in example_choices],
                value=example_choices[0][0],
            )
            rand_btn = gr.Button("Random example")
            clear_btn = gr.Button("Clear input")
            reset_btn = gr.Button("Reset UI")

        input_box = gr.Textbox(
            label="Medical text",
            placeholder="Paste English text mentioning drugs, adverse events, or interactions…",
            lines=6,
        )

        with gr.Row():
            run_btn = gr.Button("Run inference", variant="primary")

        with gr.Tabs():
            with gr.Tab("Highlights"):
                viz_html = gr.HTML()
            with gr.Tab("Table / JSON"):
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
            with gr.Tab("Raw generation"):
                raw_box = gr.Textbox(label="Raw model output", lines=16)
            with gr.Tab("Error analysis"):
                error_md = gr.Markdown("### Error analysis\nRuns after inference.")

        label_to_index = {label: idx for label, idx in example_choices}

        def on_model_change(preset: str):
            md, err = load_runtime(preset)
            if err:
                return md
            return md

        def on_example_pick(choice: str):
            idx = label_to_index.get(choice, -1)
            if idx < 0:
                return gr.update(value=""), gold_markdown([])
            row = catalog[idx]
            return gr.update(value=row["text"]), gold_markdown(row["gold"])

        def do_infer(text: str, preset: str, choice: str):
            idx = label_to_index.get(choice, -1)
            viz, table_upd, json_pretty, stats, raw_upd, err_m = run_one_inference(text, preset, idx, catalog)
            if idx >= 0:
                gmd = gold_markdown(catalog[idx]["gold"])
            else:
                gmd = gold_markdown([])
            return viz, table_upd, json_pretty, gmd, stats, raw_upd, err_m

        def random_example():
            if not catalog:
                return (
                    gr.update(),
                    gr.update(),
                    "### Ground truth\nNo `merged_chatml_test.jsonl` found under expected paths.",
                )
            idx = random.randint(0, len(catalog) - 1)
            label = example_choices[idx + 1][0]
            text_upd, gold_part = on_example_pick(label)
            return gr.update(value=label), text_upd, gold_part

        def clear_inputs():
            return (
                gr.update(value=""),
                gr.update(value=example_choices[0][0]),
                gold_markdown([]),
            )

        def reset_all():
            empty = (
                "<p><em>Reset.</em></p>",
                gr.update(value=[["(none)", "(none)", "(none)", "—"]]),
                "",
                gold_markdown([]),
                "### Timing\n(not run)",
                gr.update(value=""),
                "### Error analysis\n(not run)",
            )
            return (
                gr.update(value=""),
                gr.update(value=example_choices[0][0]),
            ) + empty

        model_select.change(on_model_change, inputs=[model_select], outputs=[model_status])

        example_dropdown.change(
            on_example_pick,
            inputs=[example_dropdown],
            outputs=[input_box, gold_md],
        )

        rand_btn.click(random_example, outputs=[example_dropdown, input_box, gold_md])

        clear_btn.click(
            clear_inputs,
            outputs=[input_box, example_dropdown, gold_md],
        )

        reset_btn.click(
            reset_all,
            outputs=[
                input_box,
                example_dropdown,
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
            inputs=[input_box, model_select, example_dropdown],
            outputs=[viz_html, pred_table, pred_json, gold_md, stats_md, raw_box, error_md],
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
        f"No free port in {start_port}–{start_port + max_attempts - 1}; "
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
    catalog = build_example_catalog(limit=args.example_limit)
    if not catalog:
        LOGGER.warning("Example JSONL not found; Examples dropdown will be empty.")

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
    )
    try:
        demo.launch(theme=gr.themes.Soft(), **launch_kw)
    except TypeError:
        demo.launch(**launch_kw)


if __name__ == "__main__":
    main()
