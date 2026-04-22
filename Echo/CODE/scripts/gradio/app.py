"""Minimal Gradio app for Echo single-text inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_backends import (
    generate_predictions,
    load_model_and_tokenizer_transformers,
    load_model_and_tokenizer_vllm,
)
from src.inference_config import apply_cli_overrides, load_inference_config
from src.parser import DatasetExample
from src.prompting import load_system_prompt

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Launch the Echo Gradio demo.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/infer_qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml",
        help="Path to the inference YAML config.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("transformers", "vllm"),
        default=None,
        help="Optional backend override.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Inline system prompt override.",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=str,
        default=None,
        help="Optional system prompt file override.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional max_new_tokens override.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature override. Any value > 0 enables sampling.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override the base model path expected by the config.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Override the LoRA adapter path.",
    )
    parser.add_argument(
        "--disable-adapter",
        action="store_true",
        help="Ignore the configured adapter and run the base model only.",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Gradio server bind address.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Gradio server port.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio's share link.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def configure_logging(*, debug: bool) -> None:
    """Configure logging."""

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load the YAML config and apply relevant CLI overrides."""

    config = load_inference_config(args.config)
    config = apply_cli_overrides(
        config,
        argparse.Namespace(
            backend=args.backend,
            split=None,
            input_path=None,
            system_prompt_path=args.system_prompt_path,
            limit=1,
            batch_size=1,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            output_path=None,
            metrics_path=None,
            metrics_json_path=None,
        ),
    )

    if args.disable_adapter:
        config["model"]["adapter_path"] = None

    config["data"]["split"] = "single"
    config["data"]["input_path"] = None
    config["data"]["max_samples"] = 1
    config["output"]["predictions_path"] = None
    config["output"]["metrics_path"] = None
    config["output"]["metrics_json_path"] = None
    return config


def resolve_system_prompt(config: Dict[str, Any], args: argparse.Namespace) -> str:
    """Resolve the prompt text used for the demo."""

    if args.system_prompt is not None:
        return args.system_prompt.strip()

    prompt_path = None
    if config.get("system_prompt_path") is not None:
        prompt_path = str(config["system_prompt_path"])
    return load_system_prompt(prompt_path)


def load_generation_bundle(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """Load the selected backend bundle and tokenizer."""

    backend = str(config.get("backend", "transformers")).lower()
    if backend == "vllm":
        llm, tokenizer, sampling_params_class, lora_request_class = load_model_and_tokenizer_vllm(config)
        return (llm, sampling_params_class, lora_request_class), tokenizer

    model, tokenizer = load_model_and_tokenizer_transformers(config)
    return model, tokenizer


def build_demo(
    *,
    config: Dict[str, Any],
    system_prompt: str,
    model_bundle: Any,
    tokenizer: Any,
) -> Any:
    """Build the Gradio Blocks UI."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "Gradio is not installed in the active environment. Install requirements.txt first."
        ) from exc

    backend = str(config.get("backend", "transformers")).lower()
    adapter_path = config["model"].get("adapter_path")
    adapter_label = "disabled (base model only)" if adapter_path is None else str(adapter_path)

    def infer(text: str) -> Tuple[str, Any, str]:
        stripped = (text or "").strip()
        if not stripped:
            return "Input text is empty.", [], ""

        example = DatasetExample(
            sample_id="single_0000",
            split="single",
            system_prompt=system_prompt,
            user_text=stripped,
            gold_relations=None,
        )
        rows = generate_predictions(model_bundle, tokenizer, [example], config)
        row = rows[0]
        predicted_relations = row.get("predicted_relations") or []
        status_line = (
            f"parse_status={row.get('parse_status')} | "
            f"failure_reason={row.get('parse_failure_reason')} | "
            f"relations={len(predicted_relations)}"
        )
        return status_line, predicted_relations, str(row.get("raw_output", ""))

    with gr.Blocks(title="Echo ADE/DDI Demo") as demo:
        gr.Markdown(
            "\n".join(
                [
                    "# Echo ADE/DDI Extraction Demo",
                    f"- Backend: `{backend}`",
                    f"- Base model: `{config['model']['base_model_name_or_path']}`",
                    f"- Adapter: `{adapter_label}`",
                ]
            )
        )
        with gr.Row():
            input_box = gr.Textbox(
                label="Medical Text",
                lines=10,
                placeholder="Paste one sentence or paragraph of medical text here.",
            )
        run_button = gr.Button("Run Inference", variant="primary")
        status_box = gr.Textbox(label="Status", interactive=False)
        relations_box = gr.JSON(label="Predicted Relations")
        raw_output_box = gr.Textbox(label="Raw Model Output", lines=8, interactive=False)

        run_button.click(
            infer,
            inputs=input_box,
            outputs=[status_box, relations_box, raw_output_box],
        )

    return demo


def main() -> None:
    """Launch the Gradio demo server."""

    args = parse_args()
    configure_logging(debug=args.debug)

    config = build_runtime_config(args)
    system_prompt = resolve_system_prompt(config, args)
    LOGGER.info(
        "Launching Gradio with backend=%s base_model=%s adapter=%s",
        config.get("backend"),
        config["model"].get("base_model_name_or_path"),
        config["model"].get("adapter_path"),
    )
    model_bundle, tokenizer = load_generation_bundle(config)
    demo = build_demo(config=config, system_prompt=system_prompt, model_bundle=model_bundle, tokenizer=tokenizer)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
