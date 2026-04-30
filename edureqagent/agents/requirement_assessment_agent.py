"""Requirement Assessment Agent."""

from __future__ import annotations

from typing import Any

from .base import AgentModule, BaseAgent
from ..prompts import REQUIREMENT_ASSESSMENT_PROMPT
from ..schemas import EvidenceUnit, LearningRequirement, RequirementAssessment, RequirementAssessmentOutput
from ..tools import (
    EmptyInput,
    EmptyOutput,
    ToolRegistry,
    ToolSpec,
    detect_over_inference,
    make_assessment_decision,
    score_actionability,
    score_specificity,
    trace_lr_to_evidence,
    validate_assessment,
)


class RequirementAssessmentAgent(BaseAgent[RequirementAssessmentOutput]):
    name = "Requirement Assessment Agent"
    system_prompt = REQUIREMENT_ASSESSMENT_PROMPT
    output_schema = RequirementAssessmentOutput
    modules = [
        AgentModule("EvidenceFaithfulnessCheckerModule", "Check whether each LR is supported by referenced evidence."),
        AgentModule("SpecificityCheckerModule", "Check whether each LR is specific enough."),
        AgentModule("ActionabilityCheckerModule", "Check whether each LR can guide pedagogical support."),
        AgentModule("OverInferenceDetectorModule", "Detect claims that exceed the available evidence."),
        AgentModule("UncertaintyCalibratorModule", "Calibrate uncertainty level."),
        AgentModule("DecisionRouterModule", "Route each LR to accept, revise, reject, or probe."),
    ]

    def build_tools(self) -> ToolRegistry:
        return ToolRegistry(
            [
                ToolSpec("trace_lr_to_evidence", "Trace LR references to evidence units.", EmptyInput, EmptyOutput, trace_lr_to_evidence),
                ToolSpec("score_specificity", "Score LR specificity.", EmptyInput, EmptyOutput, score_specificity),
                ToolSpec("score_actionability", "Score LR actionability.", EmptyInput, EmptyOutput, score_actionability),
                ToolSpec("detect_over_inference", "Detect over-inference risk.", EmptyInput, EmptyOutput, detect_over_inference),
                ToolSpec("make_assessment_decision", "Make accept/revise/reject/probe decision.", EmptyInput, EmptyOutput, make_assessment_decision),
                ToolSpec("validate_assessment", "Validate assessment consistency.", EmptyInput, EmptyOutput, validate_assessment),
            ]
        )

    def build_tool_context(self, input_payload: dict[str, Any]) -> dict[str, Any]:
        evidence_units = [EvidenceUnit.model_validate(item) for item in input_payload.get("evidence_units", [])]
        requirements = [
            LearningRequirement.model_validate(item) for item in input_payload.get("candidate_requirements", [])
        ]
        contexts = []
        for requirement in requirements:
            trace = self.tools.call("trace_lr_to_evidence", requirement, evidence_units)
            specificity = self.tools.call("score_specificity", requirement)
            actionability = self.tools.call("score_actionability", requirement)
            over_inference = self.tools.call("detect_over_inference", requirement, evidence_units)
            contexts.append(
                {
                    "requirement_id": requirement.requirement_id,
                    "trace": {
                        "support": trace["support"],
                        "missing_refs": trace["missing_refs"],
                        "referenced_ids": [unit.evidence_id for unit in trace["referenced"]],
                    },
                    "specificity": specificity,
                    "actionability": actionability,
                    "over_inference_risk": over_inference,
                }
            )
        return {"assessment_contexts": contexts, "tools": self.tools.describe()}

    def deterministic_run(
        self,
        input_payload: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> RequirementAssessmentOutput:
        evidence_units = [EvidenceUnit.model_validate(item) for item in input_payload.get("evidence_units", [])]
        requirements = [
            LearningRequirement.model_validate(item) for item in input_payload.get("candidate_requirements", [])
        ]
        assessments: list[RequirementAssessment] = []
        for requirement in requirements:
            trace = self.tools.call("trace_lr_to_evidence", requirement, evidence_units)
            specificity = self.tools.call("score_specificity", requirement)
            actionability = self.tools.call("score_actionability", requirement)
            over_inference = self.tools.call("detect_over_inference", requirement, evidence_units)
            uncertainty = "high" if over_inference == "high" else requirement.uncertainty_level
            scores = {
                "evidence_support": trace["support"],
                "specificity": specificity,
                "actionability": actionability,
                "over_inference_risk": over_inference,
                "uncertainty_level": uncertainty,
            }
            decision = self.tools.call("make_assessment_decision", scores)
            missing_evidence = None
            if decision == "probe":
                missing_evidence = requirement.verification_condition
            reason = (
                f"Evidence support is {trace['support']}; specificity is {specificity}; "
                f"actionability is {actionability}; over-inference risk is {over_inference}."
            )
            suggestions = []
            if decision == "revise":
                suggestions.append("Make the requirement more specific and explicitly tied to the cited evidence.")
            assessments.append(
                RequirementAssessment(
                    requirement_id=requirement.requirement_id,
                    decision=decision,
                    evidence_support_level=trace["support"],
                    uncertainty_level=uncertainty,
                    over_inference_risk=over_inference,
                    reason=reason,
                    required_missing_evidence=missing_evidence,
                    suggestions=suggestions,
                    metadata={"assessment_strategy": "deterministic_rules"},
                )
            )
        return RequirementAssessmentOutput(assessments=assessments)

    def post_validate(
        self,
        output: RequirementAssessmentOutput,
        input_payload: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> RequirementAssessmentOutput:
        errors: list[str] = []
        seen: set[str] = set()
        for assessment in output.assessments:
            if assessment.requirement_id in seen:
                errors.append(f"duplicate assessment for {assessment.requirement_id}")
            seen.add(assessment.requirement_id)
            errors.extend(self.tools.call("validate_assessment", assessment))
        if errors:
            raise ValueError(f"Assessment validation failed: {errors}")
        return output
