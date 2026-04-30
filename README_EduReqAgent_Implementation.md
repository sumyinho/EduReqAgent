# EduReqAgent Method Framework Implementation

This package implements the four-agent method framework only:

```text
Evidence Structuring Agent
→ Learning Requirement Generation Agent
→ Requirement Assessment Agent
→ Diagnostic Probe Agent
→ Structured JSON output
```

It does not implement baselines, gold-set construction, metric evaluation, or batch experiments.

## LLM Configuration

The workflow uses an OpenAI-compatible JSON API through `LLMClient`.
Set these environment variables when you want real LLM calls:

```powershell
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_API_KEY="..."
$env:LLM_MODEL="gpt-4o-mini"
```

Without these variables, pass `--no-llm` to run the deterministic tool-backed fallback.

## Run One Example

```powershell
python -m edureqagent.cli run `
  --question examples/question.txt `
  --response examples/response.txt `
  --rubric examples/rubric.json `
  --no-llm `
  --out output/example_edureqagent_result.json
```

The output keeps every intermediate artifact: evidence units, candidate requirements, assessments, accepted requirements, abstained requirements, rejected requirements, and diagnostic probes.
