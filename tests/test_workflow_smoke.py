from edureqagent import EduReqAgentWorkflow, ProofInput


def test_four_agent_workflow_smoke_runs_without_llm():
    proof_input = ProofInput(
        question="Prove by induction that sum_{i=0}^n i = n(n+1)/2.",
        student_response=(
            "Base case n = 1 works.\n\n"
            "Assume P(k). We need prove P(k+1).\n\n"
            "I do some algebra and conclude it is true."
        ),
        rubric={
            "Identify.Base.Case": 2,
            "Prove.Base.Case": 1,
            "Hypothesis.is.stated": 2,
            "Hypothesis.is.given.some.bound": 0,
            "Goal.is.Clear": 2,
            "Expression.of.Size.k.1.is.decomposed.into.expression.of.size.k": 0,
            "Inductive.Hypothesis.is.applied": 0,
        },
    )
    result = EduReqAgentWorkflow(use_llm=False).run(proof_input)
    assert result.evidence_units
    assert result.candidate_requirements
    assert result.assessments
    assert result.abstained_requirements
    assert result.diagnostic_probes
    assert all(probe.probe.endswith("?") for probe in result.diagnostic_probes)
