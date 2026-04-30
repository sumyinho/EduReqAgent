"""Command-line entry points for EduReqAgent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .llm_client import LLMClient, load_llm_config
from .schemas import ProofInput
from .workflow import EduReqAgentWorkflow


def _read_text(path: str | None) -> str | None:
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8")


def _read_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the EduReqAgent four-agent workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run workflow for one proof response.")
    run.add_argument("--question", required=True, help="Path to a question text file.")
    run.add_argument("--response", required=True, help="Path to a student response text file.")
    run.add_argument("--rubric", required=True, help="Path to a rubric JSON file.")
    run.add_argument("--standard-proof", default=None, help="Optional standard proof text file.")
    run.add_argument("--metadata", default=None, help="Optional metadata JSON file.")
    run.add_argument("--config", default="configs/llm.yaml", help="LLM config path.")
    run.add_argument("--no-llm", action="store_true", help="Use deterministic fallback even if an API key is configured.")
    run.add_argument("--out", default=None, help="Output JSON path. Prints to stdout if omitted.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = load_llm_config(args.config)
        workflow = EduReqAgentWorkflow(LLMClient(config), use_llm=not args.no_llm)
        proof_input = ProofInput(
            question=_read_text(args.question) or "",
            student_response=_read_text(args.response) or "",
            rubric=_read_json(args.rubric),
            standard_proof=_read_text(args.standard_proof),
            metadata=_read_json(args.metadata),
        )
        result = workflow.run(proof_input)
        text = result.model_dump_json(indent=2)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
