"""OpenAI-compatible JSON LLM client with a mock/offline mode."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class LLMConfig(BaseModel):
    provider: str = "openai_compatible"
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_retries: int = 3
    timeout_seconds: int = 60
    enabled: bool = True


def _expand_env(value: str | None) -> str | None:
    if not value:
        return value
    if value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1])
    return os.path.expandvars(value)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("'\"")
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        data[key.strip()] = parsed
    return data


def load_llm_config(path: str | Path | None = None) -> LLMConfig:
    data: dict[str, Any] = {}
    if path:
        config_path = Path(path)
        if config_path.exists():
            text = config_path.read_text(encoding="utf-8")
            data = json.loads(text) if config_path.suffix.lower() == ".json" else _parse_simple_yaml(text)
    env_data = {
        "base_url": os.getenv("LLM_BASE_URL"),
        "api_key": os.getenv("LLM_API_KEY"),
        "model": os.getenv("LLM_MODEL"),
    }
    merged = {**data, **{k: v for k, v in env_data.items() if v}}
    for key in ("base_url", "api_key", "model"):
        if key in merged:
            expanded = _expand_env(str(merged[key])) if merged[key] is not None else None
            if expanded in {None, ""} and key in {"base_url", "model"}:
                merged.pop(key, None)
            else:
                merged[key] = expanded
    return LLMConfig.model_validate(merged)


class LLMClient:
    """Small OpenAI-compatible client that validates JSON into Pydantic schemas."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or load_llm_config()

    @property
    def available(self) -> bool:
        return bool(self.config.enabled and self.config.api_key and self.config.model)

    def generate_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        response_schema: type[T],
        tool_context: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> T:
        if not self.available:
            raise RuntimeError("LLMClient is not configured. Set LLM_API_KEY and LLM_MODEL, or use deterministic fallback.")

        schema_json = response_schema.model_json_schema()
        prompt_payload = {
            "input": user_payload,
            "tool_context": tool_context or {},
            "required_json_schema": schema_json,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Return only valid JSON matching required_json_schema. "
                    "Do not include Markdown fences.\n"
                    + json.dumps(prompt_payload, ensure_ascii=False)
                ),
            },
        ]
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "response_format": {"type": "json_object"},
        }
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                request = urllib.request.Request(
                    url=url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                return response_schema.model_validate(json.loads(content))
            except (urllib.error.URLError, KeyError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"LLM JSON generation failed after {self.config.max_retries} attempts: {last_error}")
