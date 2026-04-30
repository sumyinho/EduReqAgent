"""Learning Requirement Generation Agent."""

from __future__ import annotations

from typing import Any

from .base import AgentModule, BaseAgent
from ..prompts import LR_GENERATION_PROMPT
from ..schemas import EvidenceUnit, LearningRequirement, LearningRequirementGenerationOutput
from ..tools import (
    EmptyInput,
    EmptyOutput,
    ToolRegistry,
    ToolSpec,
    check_lr_not_feedback_or_hint,
    compose_requirement_id,
    lookup_requirement_template,
    select_relevant_evidence,
    validate_learning_requirements,
)


class LearningRequirementGenerationAgent(BaseAgent[LearningRequirementGenerationOutput]):
    name = "Learning Requirement Generation Agent"
    system_prompt = LR_GENERATION_PROMPT
    output_schema = LearningRequirementGenerationOutput
    modules = [
        AgentModule("DifficultyInferenceModule", "Infer learning difficulties from structured evidence."),
        AgentModule("TargetUnderstandingModule", "Generate target understanding states."),
        AgentModule("SupportNeedConstructorModule", "Construct learning support needs without producing feedback."),
        AgentModule("PedagogicalSupportSelectorModule", "Select a pedagogical support type."),
        AgentModule("LRSchemaFillerModule", "Fill the structured Learning Requirement schema."),
        AgentModule("GroundingCheckModule", "Ensure each requirement is grounded in evidence."),
    ]

    def build_tools(self) -> ToolRegistry:
        return ToolRegistry(
            [
                ToolSpec("select_relevant_evidence", "Select evidence units that require LR generation.", EmptyInput, EmptyOutput, select_relevant_evidence),
                ToolSpec("lookup_requirement_template", "Lookup dimension-specific LR templates.", EmptyInput, EmptyOutput, lookup_requirement_template),
                ToolSpec("compose_requirement_id", "Compose stable requirement ids.", EmptyInput, EmptyOutput, compose_requirement_id),
                ToolSpec("check_lr_not_feedback_or_hint", "Detect feedback/hint-like LR text.", EmptyInput, EmptyOutput, check_lr_not_feedback_or_hint),
                ToolSpec("validate_learning_requirements", "Validate LR grounding and schema consistency.", EmptyInput, EmptyOutput, validate_learning_requirements),
            ]
        )

    def build_tool_context(self, input_payload: dict[str, Any]) -> dict[str, Any]:
        evidence_units = [EvidenceUnit.model_validate(item) for item in input_payload.get("evidence_units", [])]
        relevant = self.tools.call("select_relevant_evidence", evidence_units)
        templates = {
            unit.evidence_id: self.tools.call("lookup_requirement_template", unit.rubric_dimension, unit.evidence_type)
            for unit in relevant
        }
        return {
            "relevant_evidence": [unit.model_dump() for unit in relevant],
            "templates": templates,
            "tools": self.tools.describe(),
        }

    def deterministic_run(
        self,
        input_payload: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> LearningRequirementGenerationOutput:
        evidence_units = [EvidenceUnit.model_validate(item) for item in input_payload.get("evidence_units", [])]
        relevant = self.tools.call("select_relevant_evidence", evidence_units)
        requirements: list[LearningRequirement] = []
        for index, unit in enumerate(relevant, start=1):
            template = self.tools.call("lookup_requirement_template", unit.rubric_dimension, unit.evidence_type)
            evidence_support = "partial" if unit.evidence_type in {"partial", "missing", "unclear"} else "strong"
            uncertainty = "high" if unit.evidence_type == "missing" else "medium"
            over_inference = "high" if unit.evidence_type == "missing" else "medium" if unit.evidence_type == "unclear" else "low"
            probe_needed = unit.evidence_type in {"missing", "unclear", "contradictory"}
            requirements.append(
                LearningRequirement(
                    requirement_id=self.tools.call("compose_requirement_id", index),
                    rubric_dimension=unit.rubric_dimension,
                    observed_evidence=[unit],
                    diagnosed_difficulty=template["difficulty"],
                    learning_requirement=template["requirement"],
                    target_understanding=template["target"],
                    pedagogical_support_type=template["support"],
                    evidence_support_level=evidence_support,
                    uncertainty_level=uncertainty,
                    over_inference_risk=over_inference,
                    actionability="high",
                    verification_condition=template["verification"],
                    probe_needed=probe_needed,
                    diagnostic_probe=None,
                    metadata={"generation_strategy": "deterministic_template"},
                )
            )
        return LearningRequirementGenerationOutput(candidate_requirements=requirements)

    def post_validate(
        self,
        output: LearningRequirementGenerationOutput,
        input_payload: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> LearningRequirementGenerationOutput:
        evidence_units = [EvidenceUnit.model_validate(item) for item in input_payload.get("evidence_units", [])]
        errors = self.tools.call("validate_learning_requirements", output.candidate_requirements, evidence_units)
        if errors:
            raise ValueError(f"Learning requirement validation failed: {errors}")
        return output
