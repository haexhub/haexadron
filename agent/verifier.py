"""Two-stage Verifier — Completeness (Gemma) then Correctness (GPT-4.1-mini).

Stage 1 (Completeness): Did the agent explore all relevant sources?
  → Uses Gemma (cheap, fast, good at pattern matching)
Stage 2 (Correctness): Is the answer logically consistent with the data?
  → Uses GPT-4.1-mini (better reasoning, only runs if Stage 1 passes)

This division of labor catches both "didn't look enough" and "looked but got
the wrong answer" — two different failure modes that need different skills.
"""

from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    verdict: Literal["COMPLETE", "INCOMPLETE"] = Field(
        ...,
        description="COMPLETE: check passed. INCOMPLETE: issue found.",
    )
    feedback: str = Field(
        ...,
        description="If INCOMPLETE: specific guidance. If COMPLETE: brief confirmation.",
    )


COMPLETENESS_PROMPT = """\
You are a completeness checker for an autonomous file-based agent.
Did the agent explore ENOUGH sources before answering?

You receive: the task, the file tree, which files the agent read, and its answer.

## Check for:
- **Missing folders**: Relevant folders in the tree the agent never explored?
- **Shallow search**: Only checked one file when multiple exist?
- **Wrong location**: Looked in the wrong place? (e.g., Telegram data in docs/channels/, not accounts/)
- **Counting tasks**: CRITICAL — if the task asks "how many" or "count", the agent MUST have \
READ the full relevant file (not just searched it). The search tool returns max 20 results \
and CANNOT be used for accurate counting. If you see a counting task and the agent only \
used search but did not fully read the data file, flag INCOMPLETE with: \
"Read the full file to count accurately — search is limited to 20 results."

## Rules:
- Only flag INCOMPLETE if you see a SPECIFIC missing folder or file.
- Be brief: "Read docs/channels/Telegram.txt fully instead of searching" not "explore more".
- If exploration looks reasonable for the task, say COMPLETE.
- For counting tasks, ALWAYS check that the agent read the full data file.
- For file-writing tasks (emails, invoices, records, file creation/editing): \
if the agent's files_read list shows it read the relevant README/rules and wrote the file, \
that is COMPLETE. Do NOT flag INCOMPLETE just because the agent didn't explore unrelated folders.

Respond with EXACTLY:
VERDICT: COMPLETE or INCOMPLETE
FEEDBACK: <specific feedback>
"""

CORRECTNESS_PROMPT = """\
You are a correctness checker for an autonomous file-based agent.
Is the agent's answer logically consistent with the task and data?

You receive: the task, which files the agent read, and its proposed answer.

## CRITICAL: Understand the answer field
The "Message" field is a SUMMARY of what the agent did, NOT the file content.
- For file-writing tasks (emails, invoices, records): a message like "OUTCOME_OK", \
a file path, or a brief confirmation is CORRECT — the actual work was writing the file. \
Do NOT flag INCOMPLETE because the message doesn't contain the file content.
- For query tasks: the message IS the answer and should be checked for correctness.

## Check for:
- **Numeric plausibility**: For counting tasks, does the number make sense? \
If the data source is a large file with hundreds of entries, an answer of "5" is suspicious.
- **Date arithmetic**: Is "X days ago from Y" calculated correctly?
- **Listing completeness**: Did the agent find ALL matching items, not just the first?
- **Answer format**: Does the answer match what the task asked for? (e.g., "only the number", \
"one per line, sorted alphabetically")
- **Logic errors**: Does the conclusion follow from the data the agent read?

## Rules:
- Only flag INCOMPLETE if you see a SPECIFIC logical issue with the ANSWER CONTENT.
- Do NOT flag INCOMPLETE for file-writing tasks where the agent reports success and a file path.
- Be brief and actionable: "Re-count blacklisted entries by reading the full file" not "the count might be wrong".
- If the answer seems reasonable, say COMPLETE.

Respond with EXACTLY:
VERDICT: COMPLETE or INCOMPLETE
FEEDBACK: <specific feedback>
"""


def _parse_plain_text_result(text: str) -> VerificationResult:
    """Parse plain text response into VerificationResult."""
    lines = text.strip().split("\n")
    verdict = "COMPLETE"
    feedback = text

    for line in lines:
        upper = line.upper().strip()
        if upper.startswith("VERDICT:"):
            v = upper.split(":", 1)[1].strip()
            if "INCOMPLETE" in v:
                verdict = "INCOMPLETE"
            else:
                verdict = "COMPLETE"
        elif upper.startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    return VerificationResult(verdict=verdict, feedback=feedback)


def _run_single_check(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_msg: str,
) -> VerificationResult:
    """Run a single verification check, auto-detecting model capabilities.

    Uses a 30s timeout. If the model hangs (e.g. Qwen loop), falls through as COMPLETE.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    try:
        if not any(model.startswith(p) for p in ("openai/", "anthropic/")):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=512,
                temperature=0,
                timeout=30,
            )
            return _parse_plain_text_result(resp.choices[0].message.content or "")

        resp = client.beta.chat.completions.parse(
            model=model,
            response_format=VerificationResult,
            messages=messages,
            max_completion_tokens=512,
            temperature=0,
            timeout=30,
        )
        result = resp.choices[0].message.parsed
        if result is None:
            return VerificationResult(verdict="COMPLETE", feedback="Parse failed, allowing.")
        return result
    except Exception:
        return VerificationResult(verdict="COMPLETE", feedback="Verifier timeout, allowing.")


def verify_completion(
    client: OpenAI,
    completeness_model: str,
    correctness_model: str,
    task_text: str,
    tree_text: str,
    files_read: list[str],
    proposed_answer: str,
    proposed_outcome: str,
) -> VerificationResult:
    """Two-stage verification: completeness (cheap) then correctness (smart).

    Stage 1 runs always. Stage 2 only runs if Stage 1 passes.
    Returns the first INCOMPLETE result, or COMPLETE if both pass.
    """
    if proposed_outcome != "OUTCOME_OK":
        return VerificationResult(
            verdict="COMPLETE",
            feedback="Non-OK outcome, verification skipped.",
        )

    files_summary = "\n".join(f"- {f}" for f in (files_read or [])) if files_read else "(none)"

    completeness_msg = (
        f"## Task\n{task_text}\n\n"
        f"## File Tree\n{tree_text}\n\n"
        f"## Files Agent Read\n{files_summary}\n\n"
        f"## Agent's Answer\nOutcome: {proposed_outcome}\nMessage: {proposed_answer}"
    )

    # Stage 1: Completeness (Gemma — cheap, fast)
    stage1 = _run_single_check(client, completeness_model, COMPLETENESS_PROMPT, completeness_msg)
    if stage1.verdict == "INCOMPLETE":
        return stage1

    # Stage 2: Correctness (GPT-4.1-mini — better reasoning)
    correctness_msg = (
        f"## Task\n{task_text}\n\n"
        f"## Files Agent Read\n{files_summary}\n\n"
        f"## Agent's Answer\nOutcome: {proposed_outcome}\nMessage: {proposed_answer}"
    )

    stage2 = _run_single_check(client, correctness_model, CORRECTNESS_PROMPT, correctness_msg)
    return stage2
