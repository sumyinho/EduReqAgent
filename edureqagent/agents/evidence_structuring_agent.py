"""Evidence Structuring Agent."""

from __future__ import annotations

from typing import Any

from .base import AgentModule, BaseAgent
from ..prompts import EVIDENCE_STRUCTURING_PROMPT
from ..schemas import EvidenceStructuringOutput, ProofInput
from ..tools import (
    EmptyInput,
    EmptyOutput,
    ToolRegistry,
    ToolSpec,
    build_evidence_unit,
    normalize_rubric,
    search_span_by_patterns,
    split_proof_steps,
    validate_evidence_units,
)


class EvidenceStructuringAgent(BaseAgent[EvidenceStructuringOutput]):
    name = "Evidence Structuring Agent"
    system_prompt = EVIDENCE_STRUCTURING_PROMPT
    output_schema = EvidenceStructuringOutput
    modules = [
        AgentModule("ProofStepSegmentationModule", "Segment proof text into pedagogically meaningful proof steps."),
        AgentModule("EvidenceSpanLocatorModule", "Locate source spans for each rubric dimension."),
        AgentModule("RubricMappingModule", "Map proof segments to normalized rubric dimensions."),
        AgentModule("EvidenceStrengthEstimatorModule", "Estimate evidence status and strength."),
        AgentModule("EvidenceLedgerBuilderModule", "Build a complete evidence ledger."),
        AgentModule("EvidenceConsistencyCheckModule", "Check consistency between span, rubric, and descriptions."),
    ]

    def build_tools(self) -> ToolRegistry:
        return ToolRegistry(
            [
                ToolSpec("split_proof_steps", "Split proof text into typed proof segments.", EmptyInput, EmptyOutput, split_proof_steps),
                ToolSpec("normalize_rubric", "Normalize rubric dimensions and scores.", EmptyInput, EmptyOutput, normalize_rubric),
                ToolSpec(
                    "search_span_by_patterns",
                    "Locate likely source spans by rubric-specific patterns.",
                    EmptyInput,
                    EmptyOutput,
                    search_span_by_patterns,
                ),
                ToolSpec("build_evidence_unit", "Build one EvidenceUnit.", EmptyInput, EmptyOutput, build_evidence_unit),
                ToolSpec("validate_evidence_units", "Validate evidence units.", EmptyInput, EmptyOutput, validate_evidence_units),
            ]
        )

    def build_tool_context(self, input_payload: dict[str, Any]) -> dict[str, Any]:
        proof_input = ProofInput.model_validate(input_payload)
        segments = self.tools.call("split_proof_steps", proof_input.student_response)
        rubric_items = self.tools.call("normalize_rubric", proof_input.rubric)
        span_candidates = {
            item.dimension: self.tools.call("search_span_by_patterns", proof_input.student_response, item.dimension)
            for item in rubric_items
        }
        return {
            "segments": [segment.model_dump() for segment in segments],
            "normalized_rubric": [item.model_dump() for item in rubric_items],
            "span_candidates": span_candidates,
            "tools": self.tools.describe(),
        }

    def deterministic_run(
        self,
        input_payload: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> EvidenceStructuringOutput:
        proof_input = ProofInput.model_validate(input_payload)
        rubric_items = self.tools.call("normalize_rubric", proof_input.rubric)
        evidence_units = [
            self.tools.call("build_evidence_unit", index, proof_input.student_response, item)
            for index, item in enumerate(rubric_items, start=1)
        ]
        return EvidenceStructuringOutput(evidence_units=evidence_units)

    def post_validate(
        self,
        output: EvidenceStructuringOutput,
        input_payload: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> EvidenceStructuringOutput:
        errors = self.tools.call("validate_evidence_units", output.evidence_units)
        if errors:
            raise ValueError(f"Evidence validation failed: {errors}")
        return output
