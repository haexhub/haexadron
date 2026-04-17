# Haexadron

A two-phase, multi-model agent for the [BitGN](https://bitgn.com) **PAC1 Personal-Assistant Competition**.
Haexadron splits each task into an **early classification pass** and a **targeted tool-loop execution pass**, routed across different LLMs via [OpenRouter](https://openrouter.ai).

> Best scores so far: **~79 %** on `bitgn/pac1-dev`, **48 %** on `bitgn/pac1-prod`.

---

## Why two phases?

The BitGN harness scores outcomes in four discrete buckets:

| Outcome                        | When                                                                          |
| ------------------------------ | ----------------------------------------------------------------------------- |
| `OUTCOME_OK`                   | Task completed successfully                                                   |
| `OUTCOME_DENIED_SECURITY`      | Prompt injection or social engineering detected                               |
| `OUTCOME_NONE_CLARIFICATION`   | Information genuinely missing and cannot be inferred                          |
| `OUTCOME_NONE_UNSUPPORTED`     | Capability not available in the sandbox (e.g. real-time web access)           |

A single monolithic agent tends to either over-execute (burning steps on impossible tasks) or over-refuse (declining tasks it could solve). Haexadron separates the two concerns:

1. **Classifier** (`agent/classifier.py`) — sees the task and the bootstrap tree, decides up-front whether to `EXECUTE`, flag `SECURITY`, request `NEEDS_INFO`, or mark `UNSUPPORTED`. A wrong `UNSUPPORTED` call loses the task, so the classifier is biased towards `EXECUTE`.
2. **Executor** (`agent/pac_agent.py`) — the actual tool loop. Reads files, edits files, dispatches commands, and reports a final outcome.

Each phase can use a different model (see [Configuration](#configuration)).

---

## Architecture

```
                       ┌──────────────────────┐
                       │  BitGN harness       │
                       │  (trial + scoring)   │
                       └──────────┬───────────┘
                                  │  instruction + harness_url
                                  ▼
                       ┌──────────────────────┐
                       │  main.py             │
                       │  (optional parallel  │
                       │   ThreadPoolExecutor)│
                       └──────────┬───────────┘
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │                                                 │
        │     ┌────────────────────────────────┐          │
        │     │  Phase 1: Classifier           │          │
        │     │  → EXECUTE / SECURITY / …      │          │
        │     └──────────────┬─────────────────┘          │
        │                    │                            │
        │            EXECUTE │  non-EXECUTE → short-circuit│
        │                    ▼                            │
        │     ┌────────────────────────────────┐          │
        │     │  Phase 2: Executor loop        │          │
        │     │  • list / read / write / …     │          │
        │     │  • inbox cleanup on OK         │          │
        │     │  • security advisor per msg    │          │
        │     └──────────────┬─────────────────┘          │
        │                    ▼                            │
        │     ┌────────────────────────────────┐          │
        │     │  Verifier (completeness +      │          │
        │     │  correctness, 30s timeout)     │          │
        │     └──────────────┬─────────────────┘          │
        │                    ▼                            │
        │     ┌────────────────────────────────┐          │
        │     │  Fallback model retry          │          │
        │     │  (only if INCOMPLETE)          │          │
        │     └────────────────────────────────┘          │
        │                                                 │
        └─────────────────────────────────────────────────┘
```

Failure modes are covered by **defensive fallbacks rather than hard stops** — a hanging verifier falls through as `COMPLETE`, a step-limit-exhausted executor still emits a last-resort report, an `UNSUPPORTED` classification is nudged toward `EXECUTE` when in doubt. In a scored benchmark, a degraded answer beats no answer.

---

## Components

| File                          | Responsibility                                                                    |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `agent/main.py`               | BitGN harness driver; supports parallel trial execution                           |
| `agent/classifier.py`         | Phase-1 outcome classifier with `EXECUTE`-preferring exceptions                   |
| `agent/pac_agent.py`          | Phase-2 executor tool loop (INBOX / EMAIL / QUERY / CAPTURE prompt variants)      |
| `agent/inbox_analyzer.py`     | Specialized inbox-message semantics analyzer                                      |
| `agent/verifier.py`           | Dual completeness + correctness check with timeout fallback                       |
| `agent/security.py`           | Layer-3 tiered security policy (allow / warn / deny)                              |
| `agent/security_advisor.py`   | Nuanced per-message security judgement for inbox content                          |
| `agent/llm.py`                | OpenAI-compatible wrapper with thinking-tag stripping (Qwen / DeepSeek / MiMo)    |
| `agent/vm_dispatch.py`        | PCM runtime dispatch (byte-exact write normalization)                             |
| `agent/models.py`             | Pydantic schemas for tool calls                                                   |
| `agent/sandbox_runner.py`     | Local sandbox mode for development without the live harness                       |
| `config/deerflow/`            | deer-flow gateway + extensions configuration (Docker compose profile)             |
| `vendor/`                     | OpenClaw + deer-flow submodules (not tracked here)                                |

---

## Setup

Requirements:

- Python ≥ 3.14
- [`uv`](https://github.com/astral-sh/uv)
- An [OpenRouter](https://openrouter.ai) API key
- A [BitGN](https://bitgn.com) API key for the PAC benchmark

```bash
git clone https://github.com/haexhub/haexadron.git
cd haexadron
uv sync

cp .env.example .env
# then edit .env and fill in the two keys
```

`.env` is gitignored — never commit real keys.

---

## Running

### Against the live BitGN benchmark

```bash
# Sequential (default) — one trial at a time
uv run python -m agent.main

# Parallel — N workers (use with care; burns OpenRouter credits quickly)
uv run python -m agent.main 4
```

### Local sandbox mode

For developing prompts and tool flows without hitting the live harness:

```bash
uv run python -m agent.sandbox_runner
```

### Docker compose (OpenClaw + deer-flow profile)

```bash
docker compose up -d
```

This is only relevant for the OpenClaw/deer-flow orchestration layer. The PAC competition runner (`agent/main.py`) does **not** need the compose stack.

---

## Configuration

All configuration goes through environment variables (or `.env`). Defaults live in [`agent/config.py`](agent/config.py).

| Variable                     | Default                          | Purpose                                            |
| ---------------------------- | -------------------------------- | -------------------------------------------------- |
| `OPENROUTER_API_KEY`         | —                                | Required. OpenRouter credential.                   |
| `BITGN_API_KEY`              | —                                | Required. BitGN credential.                        |
| `BITGN_HOST`                 | `https://api.bitgn.com`          | BitGN gRPC host.                                   |
| `BENCH_ID`                   | `bitgn/pac1-dev`                 | Which benchmark to run.                            |
| `MODEL_ID`                   | `openai/gpt-4.1`                 | Main executor model.                               |
| `CLASSIFIER_MODEL_ID`        | `openai/gpt-4.1`                 | Phase-1 classifier.                                |
| `INSPECTOR_MODEL_ID`         | `google/gemma-4-31b-it`          | Tool-level inspector (currently disabled).         |
| `COMPLETENESS_MODEL_ID`      | `qwen/qwen3.6-plus`              | Verifier: did the agent read enough?               |
| `CORRECTNESS_MODEL_ID`       | `qwen/qwen3.6-plus`              | Verifier: is the answer logically consistent?      |
| `INBOX_ANALYZER_MODEL_ID`    | `qwen/qwen3.6-plus`              | Inbox message semantics.                           |
| `FALLBACK_MODEL_ID`          | `qwen/qwen3.6-plus`              | Retry model when primary returns `INCOMPLETE`.     |
| `MAX_AGENT_STEPS`            | `30`                             | Executor loop cap per trial.                       |

Swap any of these to experiment with different model mixes.

---

## Key design decisions

- **Bias toward EXECUTE.** Pre-classifying `UNSUPPORTED` loses the task; executing and then realizing it's unsupported still has a chance to score.
- **Fail-open verifier.** Verification is advisory. A hanging verifier shouldn't stall the whole trial, so timeouts and parse failures resolve to `COMPLETE`. Regressions here would be expensive — see the `Verifier semantics` note if tuning.
- **Byte-exact writes.** LLMs love to add trailing newlines. [`vm_dispatch.py`](agent/vm_dispatch.py) strips a single trailing `\n` on whole-file writes so content matches the scoring templates exactly.
- **Inbox auto-cleanup.** A successfully processed inbox message is deleted automatically — the task isn't "done" until the source file is gone.
- **Thinking-tag tolerance.** The response parser strips `<think>`, `<thinking>`, `<reasoning>`, `<thought>`, and `<|think_start|>` blocks so reasoning-heavy open models (Qwen, DeepSeek, MiMo) round-trip cleanly.

---

## Project status

- **Phase:** active development for BitGN PAC1 (competition day: April 21, 2026)
- **Benchmarks:** `bitgn/pac1-dev` (development) and `bitgn/pac1-prod` (evaluation)
- **Best runs:** ~79 % on dev, 48 % on prod

Open tasks and regression notes live in [docs/plans/](docs/plans/).

---

## License

Unlicensed / all rights reserved. This repository exists primarily as a competition submission.
