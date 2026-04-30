"""EduReqAgent four-agent workflow."""

from __future__ import annotations

from typing import Any

from .agents import (
    DiagnosticProbeAgent,
    EvidenceStructuringAgent,
    LearningRequirementGenerationAgent,
    RequirementAssessmentAgent,
)
from .llm_client import LLMClient
from .schemas import EduReqAgentResult, LearningRequirement, ProofInput, RequirementAssessment


class EduReqAgentWorkflow:
    """Directed, inspectable four-agent workflow."""

    def __init__(self, llm_client: LLMClient | None = None, use_llm: bool = True) -> None:
        self.llm_client = llm_client or LLMClient()
        self.evidence_agent = EvidenceStructuringAgent(self.llm_client, use_llm=use_llm)
        self.lr_agent = LearningRequirementGenerationAgent(self.llm_client, use_llm=use_llm)
        self.assessment_agent = RequirementAssessmentAgent(self.llm_client, use_llm=use_llm)
        self.probe_agent = DiagnosticProbeAgent(self.llm_client, use_llm=use_llm)

    def run(self, proof_input: ProofInput | dict[str, Any]) -> EduReqAgentResult:
        parsed_input = ProofInput.model_validate(proof_input)
        evidence_output = self.evidence_agent.run(parsed_input.model_dump())
        lr_output = self.lr_agent.run({"evidence_units": [unit.model_dump() for unit in evidence_output.evidence_units]})
        assessment_output = self.assessment_agent.run(
            {
                "evidence_units": [unit.model_dump() for unit in evidence_output.evidence_units],
                "candidate_requirements": [req.model_dump() for req in lr_output.candidate_requirements],
                "proof_input": parsed_input.model_dump(),
            }
        )

        requirements_by_id = {req.requirement_id: req for req in lr_output.candidate_requirements}
        final_requirements: list[LearningRequirement] = []
        abstained_requirements: list[LearningRequirement] = []
        rejected_requirements: list[LearningRequirement] = []
        final_assessments: list[RequirementAssessment] = []

        for assessment in assessment_output.assessments:
            requirement = requirements_by_id.get(assessment.requirement_id)
            if not requirement:
                continue
            if assessment.decision == "accept":
                final_requirements.append(requirement)
                final_assessments.append(assessment)
            elif assessment.decision == "reject":
                rejected_requirements.append(requirement)
                final_assessments.append(assessment)
            elif assessment.decision == "probe":
                abstained_requirements.append(requirement)
                final_assessments.append(assessment)
            elif assessment.decision == "revise":
                revised = self._revise_requirement(requirement, assessment)
                revised_output = self.assessment_agent.run(
                    {
                        "evidence_units": [unit.model_dump() for unit in evidence_output.evidence_units],
                        "candidate_requirements": [revised.model_dump()],
                        "proof_input": parsed_input.model_dump(),
                    }
                )
                revised_assessment = revised_output.assessments[0]
                final_assessments.append(revised_assessment)
                if revised_assessment.decision == "accept":
                    final_requirements.append(revised)
                elif revised_assessment.decision == "probe":
                    abstained_requirements.append(revised)
                else:
                    rejected_requirements.append(revised)

        probe_output = self.probe_agent.run(
            {
                "evidence_units": [unit.model_dump() for unit in evidence_output.evidence_units],
                "assessments": [assessment.model_dump() for assessment in final_assessments],
                "abstained_requirements": [req.model_dump() for req in abstained_requirements],
            }
        )

        return EduReqAgentResult(
            input=parsed_input,
            evidence_units=evidence_output.evidence_units,
            candidate_requirements=lr_output.candidate_requirements,
            assessments=final_assessments,
            final_requirements=final_requirements,
            abstained_requirements=abstained_requirements,
            rejected_requirements=rejected_requirements,
            diagnostic_probes=probe_output.diagnostic_probes,
            metadata={
                "workflow": "four_agent_directed",
                "llm_available": self.llm_client.available,
            },
        )

    def _revise_requirement(
        self,
        requirement: LearningRequirement,
        assessment: RequirementAssessment,
    ) -> LearningRequirement:
        revised = requirement.model_copy(deep=True)
        revised.learning_requirement = (
            f"{requirement.learning_requirement} This need should be interpreted only within "
            f"the cited evidence for {requirement.rubric_dimension}."
        )
        revised.metadata["revision_reason"] = assessment.reason
        revised.metadata["revision_strategy"] = "single_pass_scoping"
        return revised
