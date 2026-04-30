"""Shared tool interfaces and deterministic tools for EduReqAgent agents."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .schemas import (
    DiagnosticProbe,
    EvidenceUnit,
    LearningRequirement,
    NormalizedRubricItem,
    ProofSegment,
    RequirementAssessment,
)


class EmptyInput(BaseModel):
    pass


class EmptyOutput(BaseModel):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    handler: Callable[..., Any]


class ToolRegistry:
    """Named tool registry used by agent wrappers."""

    def __init__(self, specs: list[ToolSpec] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self._tools[name].handler(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, str]]:
        return [{"name": spec.name, "description": spec.description} for spec in self._tools.values()]


RUBRIC_DISPLAY_NAMES: dict[str, str] = {
    "Identify.Base.Case": "Identify Base Case",
    "Prove.Base.Case": "Prove Base Case",
    "Hypothesis.is.stated": "Hypothesis is stated",
    "Hypothesis.is.given.some.bound": "Hypothesis is given some bound",
    "Goal.is.Clear": "Goal is clear",
    "Expression.of.Size.k.1.is.decomposed.into.expression.of.size.k": (
        "Expression of size k+1 is decomposed into expression of size k"
    ),
    "Inductive.Hypothesis.is.applied": "Inductive Hypothesis is applied",
}


RUBRIC_PATTERNS: dict[str, list[str]] = {
    "Identify.Base.Case": [
        r"\bbase\s*case\b[^\n]*",
        r"\bfor\s+n\s*=\s*(0|1)\b[^\n]*",
        r"\bP\s*\(\s*(0|1)\s*\)[^\n]*",
    ],
    "Prove.Base.Case": [
        r"\bbase\s*case\b[\s\S]{0,240}?(true|holds|equals|=)",
        r"\bP\s*\(\s*(0|1)\s*\)[\s\S]{0,180}?(true|holds|=)",
    ],
    "Hypothesis.is.stated": [
        r"\b(assume|suppose|assuming)\b[\s\S]{0,220}?\b(k|n)\b",
        r"\bP\s*\(\s*k\s*\)[^\n]*",
        r"\binductive\s+hypothesis\b[^\n]*",
    ],
    "Hypothesis.is.given.some.bound": [
        r"\bfor\s+(all|some|any)\s+k\s*(>=|\\ge|≥|>|\\in|in)[^\n]*",
        r"\bk\s*(>=|\\ge|≥)\s*(0|1)[^\n]*",
    ],
    "Goal.is.Clear": [
        r"\b(prove|show|need\s+to\s+prove|goal)\b[\s\S]{0,180}?(k\s*\+\s*1|P\s*\(\s*k\s*\+\s*1\s*\))",
        r"\bP\s*\(\s*k\s*\+\s*1\s*\)[^\n]*",
    ],
    "Expression.of.Size.k.1.is.decomposed.into.expression.of.size.k": [
        r"(k\s*\+\s*1)[\s\S]{0,240}?(\+\s*\(?\s*k\s*\+\s*1|\+\s*\(k\s*\+\s*1\))",
        r"\\sum_\{?.*?\}?\^\{?k\s*\+\s*1\}?[\s\S]{0,240}?\\sum_\{?.*?\}?\^\{?k\}?",
    ],
    "Inductive.Hypothesis.is.applied": [
        r"\b(using|by|from)\s+(the\s+)?(above\s+)?(assumption|hypothesis|IH)\b[^\n]*",
        r"\binductive\s+hypothesis\b[\s\S]{0,160}?(apply|use|gives|therefore)",
    ],
}


def normalize_dimension_name(name: str) -> str:
    compact = name.strip()
    if compact in RUBRIC_DISPLAY_NAMES:
        return compact
    lowered = re.sub(r"[^a-z0-9]+", "", compact.lower())
    for canonical, display in RUBRIC_DISPLAY_NAMES.items():
        candidates = {canonical, display, canonical.replace(".", " ")}
        if lowered in {re.sub(r"[^a-z0-9]+", "", c.lower()) for c in candidates}:
            return canonical
    return compact


def normalize_score(score: Any) -> str:
    if score is None:
        return "unknown"
    text = str(score).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return "unknown"
    if text in {"2", "2.0", "c", "correct", "present", "yes", "true"}:
        return "present"
    if text in {"1", "1.0", "p", "partial", "partially present"}:
        return "partial"
    if text in {"0", "0.0", "n", "no", "not present", "missing", "false"}:
        return "missing"
    return "unknown"


def split_proof_steps(proof_text: str) -> list[ProofSegment]:
    """Segment a proof by paragraphs/lines and classify each segment coarsely."""

    segments: list[ProofSegment] = []
    for index, match in enumerate(re.finditer(r"\S(?:.*?)(?=\n\s*\n|\Z)", proof_text, flags=re.S), start=1):
        text = match.group(0).strip()
        lowered = text.lower()
        if re.search(r"base\s*case|n\s*=\s*(0|1)|p\s*\(\s*(0|1)\s*\)", lowered):
            step_type = "base_case"
        elif re.search(r"assume|suppose|inductive hypothesis|p\s*\(\s*k\s*\)", lowered):
            step_type = "hypothesis"
        elif re.search(r"k\s*\+\s*1|p\s*\(\s*k\s*\+\s*1\s*\)|inductive step", lowered):
            step_type = "inductive_step"
        elif re.search(r"hence|therefore|qed|proved|thus", lowered):
            step_type = "conclusion"
        else:
            step_type = "unclear"
        segments.append(
            ProofSegment(
                segment_id=f"S{index}",
                step_type=step_type,
                char_start=match.start(),
                char_end=match.end(),
                text=text,
            )
        )
    if not segments and proof_text.strip():
        segments.append(
            ProofSegment(segment_id="S1", step_type="unclear", char_start=0, char_end=len(proof_text), text=proof_text)
        )
    return segments


def normalize_rubric(rubric: dict[str, Any]) -> list[NormalizedRubricItem]:
    items: list[NormalizedRubricItem] = []
    for name, score in rubric.items():
        canonical = normalize_dimension_name(str(name))
        display = RUBRIC_DISPLAY_NAMES.get(canonical, str(name).replace(".", " "))
        items.append(
            NormalizedRubricItem(
                dimension=canonical,
                display_name=display,
                raw_score=score,
                normalized_score=normalize_score(score),
            )
        )
    return items


def search_span_by_patterns(proof_text: str, rubric_dimension: str) -> dict[str, Any]:
    patterns = RUBRIC_PATTERNS.get(normalize_dimension_name(rubric_dimension), [])
    for pattern in patterns:
        match = re.search(pattern, proof_text, flags=re.I | re.S)
        if match:
            text = re.sub(r"\s+", " ", match.group(0)).strip()
            return {
                "found": True,
                "char_start": match.start(),
                "char_end": match.end(),
                "source_text": text,
                "evidence_span": f"chars {match.start()}-{match.end()}",
                "pattern": pattern,
            }
    return {"found": False, "source_text": "", "evidence_span": "not located", "pattern": None}


def _evidence_type_from_score(score: str, span_found: bool) -> str:
    if score == "present":
        return "present" if span_found else "unclear"
    if score == "partial":
        return "partial"
    if score == "missing":
        return "missing"
    return "unclear"


def _evidence_strength_from_type(evidence_type: str, span_found: bool) -> str:
    if evidence_type == "present" and span_found:
        return "strong_positive"
    if evidence_type in {"present", "partial", "unclear"}:
        return "weak_positive" if evidence_type != "unclear" else "unknown"
    if evidence_type == "missing":
        return "strong_negative"
    if evidence_type == "contradictory":
        return "contradictory"
    return "unknown"


def build_evidence_unit(
    index: int,
    proof_text: str,
    rubric_item: NormalizedRubricItem,
) -> EvidenceUnit:
    span = search_span_by_patterns(proof_text, rubric_item.dimension)
    span_found = bool(span["found"])
    evidence_type = _evidence_type_from_score(rubric_item.normalized_score, span_found)
    source_text = span["source_text"] if evidence_type != "missing" else ""
    if evidence_type == "missing":
        evidence_span = "missing from proof text"
        description = f"No clear evidence for '{rubric_item.display_name}' is present in the response."
    elif span_found:
        evidence_span = span["evidence_span"]
        qualifier = "partial evidence for" if evidence_type == "partial" else "evidence for"
        description = f"The response provides {qualifier} '{rubric_item.display_name}'."
    else:
        evidence_span = "rubric label only"
        description = (
            f"Rubric score suggests '{rubric_item.display_name}', but no reliable local span was found."
        )
    return EvidenceUnit(
        evidence_id=f"E{index}",
        rubric_dimension=rubric_item.dimension,
        evidence_span=evidence_span,
        evidence_type=evidence_type,
        description=description,
        evidence_strength=_evidence_strength_from_type(evidence_type, span_found),
        source_text=source_text,
        rubric_score=str(rubric_item.raw_score) if rubric_item.raw_score is not None else None,
        metadata={"rubric_display_name": rubric_item.display_name, "span_pattern": span.get("pattern")},
    )


def validate_evidence_units(evidence_units: list[EvidenceUnit]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for unit in evidence_units:
        if unit.evidence_id in seen:
            errors.append(f"duplicate evidence_id: {unit.evidence_id}")
        seen.add(unit.evidence_id)
        if unit.evidence_type == "missing" and unit.source_text.strip():
            errors.append(f"{unit.evidence_id} is missing but has source_text")
        if unit.evidence_type != "missing" and unit.evidence_span == "missing from proof text":
            errors.append(f"{unit.evidence_id} has inconsistent evidence_span")
    return errors


REQUIREMENT_TEMPLATES: dict[str, dict[str, str]] = {
    "Identify.Base.Case": {
        "difficulty": "The learner does not clearly identify the base case needed to start the induction proof.",
        "requirement": "The learner needs support in selecting and naming the correct base case before the inductive step.",
        "target": "The learner can state the initial value and explain why induction begins there.",
        "support": "base-case comparison scaffold",
        "verification": "Ask the learner to identify the starting value and justify why it is the base case.",
    },
    "Prove.Base.Case": {
        "difficulty": "The learner does not clearly verify that the proposition holds at the base case.",
        "requirement": "The learner needs support in checking both sides or all required parts of the proposition at the base case.",
        "target": "The learner can substitute the base value and verify the statement rather than only naming it.",
        "support": "worked base-case example",
        "verification": "Ask the learner to show the calculation that verifies the base case.",
    },
    "Hypothesis.is.stated": {
        "difficulty": "The learner does not clearly state the inductive hypothesis as an assumption.",
        "requirement": "The learner needs support in formulating P(k) as the assumption used later in the proof.",
        "target": "The learner can write the proposition for an arbitrary k before attempting the k+1 case.",
        "support": "contrastive example",
        "verification": "Ask the learner to write the exact statement assumed true for k.",
    },
    "Hypothesis.is.given.some.bound": {
        "difficulty": "The learner does not clearly attach the inductive hypothesis to the correct domain or lower bound.",
        "requirement": "The learner needs support in specifying the range of k values for which the hypothesis is assumed.",
        "target": "The learner can state the allowed values of k and connect them to the base case.",
        "support": "domain-focused prompt",
        "verification": "Ask the learner to state for which values of k the inductive hypothesis is assumed.",
    },
    "Goal.is.Clear": {
        "difficulty": "The learner does not clearly express the proposition to be proved for k+1.",
        "requirement": "The learner needs support in translating the original proposition into the k+1 goal.",
        "target": "The learner can write P(k+1) correctly and use it to organize the inductive step.",
        "support": "goal-setting scaffold",
        "verification": "Ask the learner to write the target statement for k+1 before manipulating expressions.",
    },
    "Expression.of.Size.k.1.is.decomposed.into.expression.of.size.k": {
        "difficulty": "The learner does not clearly decompose the k+1 expression into a part where P(k) can be used.",
        "requirement": "The learner needs support in separating the k+1 case into the k case plus the new term or structure.",
        "target": "The learner can transform the k+1 expression so the inductive hypothesis becomes applicable.",
        "support": "worked-example comparison",
        "verification": "Ask the learner to show where the k-size expression appears inside the k+1 case.",
    },
    "Inductive.Hypothesis.is.applied": {
        "difficulty": "The learner does not clearly use the inductive hypothesis as a reasoning resource in the inductive step.",
        "requirement": "The learner needs support in connecting the assumed P(k) to the proof of P(k+1).",
        "target": "The learner can identify the line where P(k) is invoked and explain how it advances the proof.",
        "support": "evidence-linked explanation prompt",
        "verification": "Ask the learner to explain which step uses P(k) and why that use is valid.",
    },
}


def select_relevant_evidence(evidence_units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    return [
        unit
        for unit in evidence_units
        if unit.evidence_type in {"partial", "missing", "contradictory", "unclear"}
        or unit.evidence_strength in {"weak_negative", "strong_negative", "contradictory", "unknown"}
    ]


def lookup_requirement_template(rubric_dimension: str, evidence_type: str) -> dict[str, str]:
    template = REQUIREMENT_TEMPLATES.get(normalize_dimension_name(rubric_dimension))
    if template:
        return template
    display = rubric_dimension.replace(".", " ")
    return {
        "difficulty": f"The learner's response leaves uncertainty around {display}.",
        "requirement": f"The learner needs targeted support related to {display}.",
        "target": f"The learner can demonstrate the expected understanding for {display}.",
        "support": "targeted clarification scaffold",
        "verification": f"Ask the learner to clarify their reasoning for {display}.",
    }


def compose_requirement_id(index: int) -> str:
    return f"LR-{index:03d}"


def check_lr_not_feedback_or_hint(text: str) -> bool:
    lowered = text.strip().lower()
    direct_address = lowered.startswith(("you ", "try ", "please ", "next, ", "first, "))
    imperative = bool(re.search(r"\b(try|remember|calculate|write|use)\b", lowered[:80]))
    return not (direct_address or imperative)


def validate_learning_requirements(
    requirements: list[LearningRequirement],
    evidence_units: list[EvidenceUnit],
) -> list[str]:
    errors: list[str] = []
    evidence_ids = {unit.evidence_id for unit in evidence_units}
    for req in requirements:
        if not check_lr_not_feedback_or_hint(req.learning_requirement):
            errors.append(f"{req.requirement_id} appears to be feedback or a hint")
        for observed in req.observed_evidence:
            if observed.evidence_id not in evidence_ids:
                errors.append(f"{req.requirement_id} references unknown evidence {observed.evidence_id}")
    return errors


def trace_lr_to_evidence(requirement: LearningRequirement, evidence_units: list[EvidenceUnit]) -> dict[str, Any]:
    evidence_by_id = {unit.evidence_id: unit for unit in evidence_units}
    refs = [evidence_by_id.get(unit.evidence_id) for unit in requirement.observed_evidence]
    missing_refs = [unit.evidence_id for unit in requirement.observed_evidence if unit.evidence_id not in evidence_by_id]
    supporting = [unit for unit in refs if unit and unit.evidence_type != "missing"]
    negative = [unit for unit in refs if unit and unit.evidence_type == "missing"]
    if missing_refs:
        support = "none"
    elif supporting and not negative:
        support = "strong" if all(unit.evidence_strength == "strong_positive" for unit in supporting) else "partial"
    elif supporting or negative:
        support = "partial"
    else:
        support = "none"
    return {"support": support, "missing_refs": missing_refs, "referenced": [u for u in refs if u is not None]}


def score_specificity(requirement: LearningRequirement) -> str:
    text = f"{requirement.learning_requirement} {requirement.target_understanding}".lower()
    broad_phrases = ["relearn", "study more", "understand induction", "learn mathematical induction", "重新学习"]
    if any(phrase in text for phrase in broad_phrases):
        return "low"
    if len(requirement.learning_requirement.split()) >= 8 and requirement.rubric_dimension:
        return "high"
    return "medium"


def score_actionability(requirement: LearningRequirement) -> str:
    if requirement.verification_condition and requirement.pedagogical_support_type:
        return "high"
    if requirement.verification_condition or requirement.pedagogical_support_type:
        return "medium"
    return "low"


def detect_over_inference(requirement: LearningRequirement, evidence_units: list[EvidenceUnit]) -> str:
    trace = trace_lr_to_evidence(requirement, evidence_units)
    evidence_types = {unit.evidence_type for unit in trace["referenced"]}
    if trace["support"] == "none":
        return "high"
    if "missing" in evidence_types or requirement.uncertainty_level == "high":
        return "high"
    if "partial" in evidence_types or "unclear" in evidence_types:
        return "medium"
    return "low"


def make_assessment_decision(scores: dict[str, str]) -> str:
    if scores["evidence_support"] == "none":
        return "reject"
    if scores["over_inference_risk"] == "high" and scores["uncertainty_level"] in {"medium", "high"}:
        return "probe"
    if scores["specificity"] == "low" or scores["actionability"] == "low":
        return "revise"
    return "accept"


def validate_assessment(assessment: RequirementAssessment) -> list[str]:
    if assessment.decision == "probe" and not assessment.required_missing_evidence:
        return [f"{assessment.requirement_id}: probe decision needs required_missing_evidence"]
    return []


def extract_missing_evidence(assessment: RequirementAssessment) -> str:
    return assessment.required_missing_evidence or "The learner's reasoning is under-specified."


def generate_competing_hypotheses(requirement: LearningRequirement, evidence_units: list[EvidenceUnit]) -> list[str]:
    display = requirement.rubric_dimension.replace(".", " ")
    return [
        f"The learner understands {display} but left the reasoning implicit.",
        f"The learner has not yet developed the understanding needed for {display}.",
    ]


def select_probe_type(missing_evidence: str, rubric_dimension: str) -> str:
    lowered = f"{missing_evidence} {rubric_dimension}".lower()
    if "write" in lowered or "state" in lowered:
        return "rewrite"
    if "which step" in lowered or "explain" in lowered or "how" in lowered:
        return "explanation"
    if "decompose" in lowered or "prove" in lowered:
        return "micro-proof"
    return "explanation"


def default_probe_text(requirement: LearningRequirement, missing_evidence: str) -> str:
    dim = requirement.rubric_dimension
    if dim == "Hypothesis.is.stated":
        return "What exact statement are you assuming is true for k before proving the k+1 case?"
    if dim == "Hypothesis.is.given.some.bound":
        return "For which values of k is your inductive hypothesis assumed to hold?"
    if dim == "Goal.is.Clear":
        return "What statement do you need to prove for the k+1 case?"
    if dim == "Expression.of.Size.k.1.is.decomposed.into.expression.of.size.k":
        return "Where does the k-size expression appear inside your k+1 expression?"
    if dim == "Inductive.Hypothesis.is.applied":
        return "Which line in your k+1 proof uses the statement assumed for k?"
    if dim == "Prove.Base.Case":
        return "Can you show the calculation that verifies the base case?"
    if dim == "Identify.Base.Case":
        return "What is the starting value for this induction proof, and why?"
    return f"Can you clarify the missing reasoning about {missing_evidence}?"


def check_probe_non_leakage(probe: str) -> bool:
    leaked_phrases = ["the answer is", "you should use", "therefore use p(k)", "equals the target"]
    lowered = probe.lower()
    return not any(phrase in lowered for phrase in leaked_phrases)


def check_probe_discriminativeness(probe: str, competing_hypotheses: list[str]) -> bool:
    return bool(probe.strip().endswith("?") and len(competing_hypotheses) >= 2 and len(probe.split()) >= 6)


def validate_diagnostic_probe(probe: DiagnosticProbe, requirement_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if probe.requirement_id not in requirement_ids:
        errors.append(f"probe references unknown requirement {probe.requirement_id}")
    if not check_probe_non_leakage(probe.probe):
        errors.append(f"{probe.requirement_id}: probe may leak the answer")
    if not check_probe_discriminativeness(probe.probe, probe.competing_hypotheses):
        errors.append(f"{probe.requirement_id}: probe may not distinguish hypotheses")
    return errors
