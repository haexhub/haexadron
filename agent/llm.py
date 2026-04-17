"""LLM abstraction — unified structured output for all models.

Models like GPT-4.1 support native structured output via response_format.
Models like Gemma need JSON mode + manual Pydantic parsing.
This module auto-detects the model and picks the right path.
"""

import json
import re

from openai import OpenAI
from pydantic import BaseModel


# Models that support native structured output (response_format with Pydantic)
STRUCTURED_OUTPUT_MODELS = {"openai/", "anthropic/"}


def _supports_structured_output(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in STRUCTURED_OUTPUT_MODELS)


def _extract_json(text: str) -> str:
    """Extract JSON from model response, handling code fences and thinking tags."""
    # Strip thinking blocks (Qwen, MiMo, DeepSeek, etc.)
    text = re.sub(r"<(?:think|thinking|reasoning|thought)>.*?</(?:think|thinking|reasoning|thought)>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<\|think_start\|>.*?<\|think_end\|>", "", text, flags=re.DOTALL).strip()

    # Try to find JSON in code fences first
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try to find a JSON object directly (first { to last })
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0).strip()

    # Otherwise assume the whole response is JSON
    return text.strip()


def parse_structured(
    client: OpenAI,
    model: str,
    response_format: type[BaseModel],
    messages: list[dict],
    max_completion_tokens: int = 16384,
    temperature: float = 0,
) -> BaseModel | None:
    """Request structured output from any model.

    For models with native structured output support, uses response_format directly.
    For other models (Gemma), uses JSON mode and parses manually with Pydantic.

    Returns:
        Parsed Pydantic model instance, or None if parsing fails.
    """
    if _supports_structured_output(model):
        resp = client.beta.chat.completions.parse(
            model=model,
            response_format=response_format,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.parsed

    # JSON mode path for models without native structured output
    # Inject schema info into the system message
    schema_json = json.dumps(response_format.model_json_schema(), indent=2)

    # Prepend schema instruction to the first system message
    patched_messages = []
    schema_injected = False
    for msg in messages:
        if msg["role"] == "system" and not schema_injected:
            patched_messages.append({
                "role": "system",
                "content": (
                    f"{msg['content']}\n\n"
                    f"IMPORTANT: Respond with ONLY valid JSON matching this exact schema (no markdown, no explanation):\n"
                    f"{schema_json}"
                ),
            })
            schema_injected = True
        else:
            patched_messages.append(msg)

    resp = client.chat.completions.create(
        model=model,
        messages=patched_messages,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content
    if not raw:
        return None

    try:
        json_str = _extract_json(raw)
        return response_format.model_validate_json(json_str)
    except Exception as exc:
        print(f"\x1B[33mPARSE FAIL ({model}): {exc}\nRAW: {raw[:300]}\x1B[0m")
        return None
