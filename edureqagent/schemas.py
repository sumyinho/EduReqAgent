"""Pydantic schemas shared by the EduReqAgent workflow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


EvidenceType = Literal["present", "partial", "missing", "contradictory", "unclear"]
EvidenceStrength = Literal[
    "strong_positive",
    "weak_positive",
    "weak_negative",
    "strong_negative",
    "contradictory",
    "unknown",
]
Level = Literal["low", "medium", "high"]
Decision = Literal["accept", "revise", "reject", "probe"]
ProbeType = Literal["explanation", "micro-proof", "contrastive judgment", "rewrite", "other"]


class ProofInput(BaseModel):
    """Single proof-response input for cold-start learning requirement generation."""

    question: str
    student_response: str
    rubric: dict[str, Any] = Field(default_factory=dict)
    standard_proof: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProofSegment(BaseModel):
    segment_id: str
    step_type: Literal["base_case", "hypothesis", "inductive_step", "conclusion", "unclear"]
    char_start: int
    char_end: int
    text: str


class NormalizedRubricItem(BaseModel):
    dimension: str
    display_name: str
    raw_score: Any = None
    normalized_score: Literal["present", "partial", "missing", "unknown"]


class EvidenceUnit(BaseModel):
    evidence_id: str
    rubric_dimension: str
    evidence_span: str
    evidence_type: EvidenceType
    description: str
    evidence_strength: EvidenceStrength
    source_text: str = ""
    rubric_score: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def missing_evidence_must_not_have_source_text(self) -> "EvidenceUnit":
        if self.evidence_type == "missing" and self.source_text.strip():
            raise ValueError("missing evidence cannot include source_text")
        return self


class LearningRequirement(BaseModel):
    requirement_id: str
    target_concept: str = "Mathematical Induction"
    rubric_dimension: str
    observed_evidence: list[EvidenceUnit]
    diagnosed_difficulty: str
    learning_requirement: str
    target_understanding: str
    pedagogical_support_type: str
    evidence_support_level: Literal["none", "partial", "strong"]
    uncertainty_level: Level
    over_inference_risk: Level
    actionability: Level
    verification_condition: str
    probe_needed: bool
    diagnostic_probe: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def must_reference_evidence(self) -> "LearningRequirement":
        if not self.observed_evidence:
            raise ValueError("LearningRequirement must reference at least one evidence unit")
        return self


class RequirementAssessment(BaseModel):
    requirement_id: str
    decision: Decision
    evidence_support_level: Literal["none", "partial", "strong"]
    uncertainty_level: Level
    over_inference_risk: Level
    reason: str
    required_missing_evidence: str | None = None
    suggestions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def probe_requires_missing_evidence(self) -> "RequirementAssessment":
        if self.decision == "probe" and not self.required_missing_evidence:
            raise ValueError("probe decisions require required_missing_evidence")
        return self


class DiagnosticProbe(BaseModel):
    requirement_id: str
    probe: str
    missing_evidence: str
    competing_hypotheses: list[str]
    probe_type: ProbeType
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def needs_competing_hypotheses(self) -> "DiagnosticProbe":
        if len(self.competing_hypotheses) < 2:
            raise ValueError("DiagnosticProbe requires at least two competing hypotheses")
        return self


class EvidenceStructuringOutput(BaseModel):
    evidence_units: list[EvidenceUnit]


class LearningRequirementGenerationOutput(BaseModel):
    candidate_requirements: list[LearningRequirement]


class RequirementAssessmentOutput(BaseModel):
    assessments: list[RequirementAssessment]


class DiagnosticProbeOutput(BaseModel):
    diagnostic_probes: list[DiagnosticProbe]


class EduReqAgentResult(BaseModel):
    input: ProofInput
    evidence_units: list[EvidenceUnit]
    candidate_requirements: list[LearningRequirement]
    assessments: list[RequirementAssessment]
    final_requirements: list[LearningRequirement]
    abstained_requirements: list[LearningRequirement]
    rejected_requirements: list[LearningRequirement]
    diagnostic_probes: list[DiagnosticProbe]
    metadata: dict[str, Any] = Field(default_factory=dict)
