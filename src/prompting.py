import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_SYSTEM_PROMPT = """You are a medical relation extraction assistant.

Task:
Extract all valid medical relations from the input medical text.

Allowed relation types:
- ADE
- DDI-MECHANISM
- DDI-EFFECT
- DDI-ADVISE
- DDI-INT

Output rules:
- Output valid JSON only.
- Output a JSON list.
- Do not output explanations.
- Do not output reasoning or <think> content.
- Do not output markdown.
- Each item must follow this exact format:

[
  {
    "head_entity": "string",
    "tail_entity": "string",
    "relation_type": "ADE or one of the DDI subtypes"
  }
]

- relation_type must be exactly one of the allowed relation types above.
- If there is no relation, output:
[]
"""


def extract_message_content(messages: List[Dict[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            return str(message.get("content", "")).strip()
    return ""


def load_system_prompt(system_prompt_path: Optional[str] = None) -> str:
    if system_prompt_path:
        return Path(system_prompt_path).read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM_PROMPT.strip()


def build_messages(system_prompt: str, user_text: str, assistant_text: Optional[str] = None) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_text.strip()},
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text.strip()})
    return messages


def supports_chat_template(tokenizer: Any) -> bool:
    return callable(getattr(tokenizer, "apply_chat_template", None))


def supports_assistant_tokens_mask(tokenizer: Any) -> bool:
    chat_template = getattr(tokenizer, "chat_template", None)
    return isinstance(chat_template, str) and "{% generation" in chat_template


def should_try_enable_thinking(tokenizer: Any, enable_thinking: Optional[bool]) -> bool:
    if enable_thinking is None:
        return False

    name_or_path = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "qwen3" in name_or_path:
        return True

    try:
        signature = inspect.signature(tokenizer.apply_chat_template)
    except (TypeError, ValueError):
        return False

    return "enable_thinking" in signature.parameters


def apply_chat_template(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    truncation: bool = False,
    max_length: Optional[int] = None,
    return_dict: bool = False,
    return_assistant_tokens_mask: bool = False,
    enable_thinking: Optional[bool] = None,
) -> Any:
    if not supports_chat_template(tokenizer):
        raise ValueError("Tokenizer does not expose apply_chat_template().")

    kwargs: Dict[str, Any] = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
        "truncation": truncation,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    if return_dict:
        kwargs["return_dict"] = True
    if return_assistant_tokens_mask:
        kwargs["return_assistant_tokens_mask"] = True

    if should_try_enable_thinking(tokenizer, enable_thinking):
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                **kwargs,
            )
        except TypeError:
            pass

    return tokenizer.apply_chat_template(messages, **kwargs)
