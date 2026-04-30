"""Diagnostic Probe Agent."""

from __future__ import annotations

from typing import Any

from .base import AgentModule, BaseAgent
from ..prompts import DIAGNOSTIC_PROBE_PROMPT
from ..schemas import (
    DiagnosticProbe,
    DiagnosticProbeOutput,
    EvidenceUnit,
    LearningRequirement,
    RequirementAssessment,
)
from ..tools import (
    EmptyInput,
    EmptyOutput,
    ToolRegistry,
    ToolSpec,
    check_probe_discriminativeness,
    check_probe_non_leakage,
    default_probe_text,
    extract_missing_evidence,
    generate_competing_hypotheses,
    select_probe_type,
    validate_diagnostic_probe,
)


class DiagnosticProbeAgent(BaseAgent[DiagnosticProbeOutput]):
    name = "Diagnostic Probe Agent"
    system_prompt = DIAGNOSTIC_PROBE_PROMPT
    output_schema = DiagnosticProbeOutput
    modules = [
        AgentModule("MissingEvidenceAnalyzerModule", "Identify missing evidence for uncertain requirements."),
        AgentModule("CompetingHypothesesModule", "Generate competing explanations."),
        AgentModule("ProbeTypeSelectorModule", "Select the probe type."),
        AgentModule("MinimalProbeGeneratorModule", "Generate a minimal diagnostic question."),
        AgentModule("ProbeQualityCheckerModule", "Check targetedness, minimality, discriminativeness, and non-leakage."),
    ]

    def build_tools(self) -> ToolRegistry:
        return ToolRegistry(
            [
                ToolSpec("extract_missing_evidence", "Extract missing evidence from assessment.", EmptyInput, EmptyOutput, extract_missing_evidence),
                ToolSpec("generate_competing_hypotheses", "Generate competing hypotheses.", EmptyInput, EmptyOutput, generate_competing_hypotheses),
                ToolSpec("select_probe_type", "Select probe type.", EmptyInput, EmptyOutput, select_probe_type),
                ToolSpec("check_probe_non_leakage", "Check whether probe leaks answer.", EmptyInput, EmptyOutput, check_probe_non_leakage),
                ToolSpec("check_probe_discriminativeness", "Check whether probe distinguishes hypotheses.", EmptyInput, EmptyOutput, check_probe_discriminativeness),
                ToolSpec("validate_diagnostic_probe", "Validate diagnostic probe.", EmptyInput, EmptyOutput, validate_diagnostic_probe),
            ]
        )

    def build_tool_context(self, input_payload: dict[str, Any]) -> dict[str, Any]:
        requirements = [
            LearningRequirement.model_validate(item) for item in input_payload.get("abstained_requirements", [])
        ]
        assessments = [RequirementAssessment.model_validate(item) for item in input_payload.get("assessments", [])]
        evidence_units = [EvidenceUnit.model_validate(item) for item in input_payload.get("evidence_units", [])]
        assessments_by_id = {assessment.requirement_id: assessment for assessment in assessments}
        contexts = []
        for requirement in requirements:
            assessment = assessments_by_id.get(requirement.requirement_id)
            if not assessment:
                continue
            missing = self.tools.call("extract_missing_evidence", assessment)
            hypotheses = self.tools.call("generate_competing_hypotheses", requirement, evidence_units)
            contexts.append(
                {
                    "requirement_id": requirement.requirement_id,
                    "missing_evidence": missing,
                    "competing_hypotheses": hypotheses,
                    "probe_type": self.tools.call("select_probe_type", missing, requirement.rubric_dimension),
                }
            )
        return {"probe_contexts": contexts, "tools": self.tools.describe()}

    def deterministic_run(
        self,
        input_payload: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> DiagnosticProbeOutput:
        requirements = [
            LearningRequirement.model_validate(item) for item in input_payload.get("abstained_requirements", [])
        ]
        assessments = [RequirementAssessment.model_validate(item) for item in input_payload.get("assessments", [])]
        evidence_units = [EvidenceUnit.model_validate(item) for item in input_payload.get("evidence_units", [])]
        assessments_by_id = {assessment.requirement_id: assessment for assessment in assessments}
        probes: list[DiagnosticProbe] = []
        for requirement in requirements:
            assessment = assessments_by_id.get(requirement.requirement_id)
            if not assessment:
                continue
            missing = self.tools.call("extract_missing_evidence", assessment)
            hypotheses = self.tools.call("generate_competing_hypotheses", requirement, evidence_units)
            probe_text = default_probe_text(requirement, missing)
            probes.append(
                DiagnosticProbe(
                    requirement_id=requirement.requirement_id,
                    probe=probe_text,
                    missing_evidence=missing,
                    competing_hypotheses=hypotheses,
                    probe_type=self.tools.call("select_probe_type", missing, requirement.rubric_dimension),
                    metadata={"generation_strategy": "deterministic_template"},
                )
            )
        return DiagnosticProbeOutput(diagnostic_probes=probes)

    def post_validate(
        self,
        output: DiagnosticProbeOutput,
        input_payload: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> DiagnosticProbeOutput:
        requirement_ids = {
            item["requirement_id"] for item in input_payload.get("abstained_requirements", []) if "requirement_id" in item
        }
        errors: list[str] = []
        for probe in output.diagnostic_probes:
            errors.extend(self.tools.call("validate_diagnostic_probe", probe, requirement_ids))
        if errors:
            raise ValueError(f"Diagnostic probe validation failed: {errors}")
        return output
