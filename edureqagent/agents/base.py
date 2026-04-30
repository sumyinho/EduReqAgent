"""Base agent abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from ..llm_client import LLMClient
from ..tools import ToolRegistry

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class AgentModule:
    name: str
    description: str


class BaseAgent(Generic[T]):
    name: str = "BaseAgent"
    system_prompt: str = ""
    modules: list[AgentModule] = []
    tools: ToolRegistry
    output_schema: type[T]

    def __init__(self, llm_client: LLMClient | None = None, use_llm: bool = True) -> None:
        self.llm_client = llm_client or LLMClient()
        self.use_llm = use_llm
        self.tools = self.build_tools()

    def build_tools(self) -> ToolRegistry:
        return ToolRegistry()

    def build_tool_context(self, input_payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    def deterministic_run(self, input_payload: dict[str, Any], tool_context: dict[str, Any]) -> T:
        raise NotImplementedError

    def post_validate(self, output: T, input_payload: dict[str, Any], tool_context: dict[str, Any]) -> T:
        return output

    def run(self, input_payload: dict[str, Any]) -> T:
        tool_context = self.build_tool_context(input_payload)
        if self.use_llm and self.llm_client.available:
            output = self.llm_client.generate_json(
                system_prompt=self.system_prompt,
                user_payload=input_payload,
                response_schema=self.output_schema,
                tool_context=tool_context,
            )
        else:
            output = self.deterministic_run(input_payload, tool_context)
        return self.post_validate(output, input_payload, tool_context)
