"""Haexadron PAC1 competition runner — connects to BitGN and runs the agent."""

import os
import sys
import textwrap

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import (
    EndTrialRequest,
    EvalPolicy,
    GetBenchmarkRequest,
    StartRunRequest,
    StartTrialRequest,
    StatusRequest,
    SubmitRunRequest,
)
from connectrpc.errors import ConnectError
from openai import OpenAI

from agent.config import (
    BENCH_ID,
    BITGN_API_KEY,
    BITGN_HOST,
    MAX_AGENT_STEPS,
    MODEL_ID,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from agent.pac_agent import run_agent

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_BLUE = "\x1B[34m"
CLI_CLR = "\x1B[0m"


def main() -> None:
    if not OPENROUTER_API_KEY:
        print(f"{CLI_RED}ERROR: OPENROUTER_API_KEY not set{CLI_CLR}")
        sys.exit(1)

    # OpenRouter is OpenAI-compatible — just change base_url
    llm_client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    task_filter = sys.argv[1:]

    print(f"Model: {MODEL_ID}")
    print(f"Benchmark: {BENCH_ID}")
    print(f"Max steps: {MAX_AGENT_STEPS}")

    scores: list[tuple[str, float]] = []
    try:
        client = HarnessServiceClientSync(BITGN_HOST)
        print("Connecting to BitGN...", client.status(StatusRequest()))

        res = client.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCH_ID))
        print(
            f"{EvalPolicy.Name(res.policy)} benchmark: {res.benchmark_id} "
            f"with {len(res.tasks)} tasks.\n{CLI_GREEN}{res.description}{CLI_CLR}"
        )

        run = client.start_run(
            StartRunRequest(
                name="Haexadron PAC Agent",
                benchmark_id=BENCH_ID,
                api_key=BITGN_API_KEY,
            )
        )

        try:
            for trial_id in run.trial_ids:
                trial = client.start_trial(StartTrialRequest(trial_id=trial_id))

                if task_filter and trial.task_id not in task_filter:
                    continue

                print(f"\n{'=' * 30} Task: {trial.task_id} {'=' * 30}")
                print(f"{CLI_BLUE}{trial.instruction}{CLI_CLR}\n{'-' * 80}")

                try:
                    run_agent(
                        client=llm_client,
                        model=MODEL_ID,
                        harness_url=trial.harness_url,
                        task_text=trial.instruction,
                        enable_inspector=False,  # Tool-level inspector off (false positives), task-level check handles security
                    )
                except Exception as exc:
                    print(f"{CLI_RED}AGENT ERROR: {exc}{CLI_CLR}")

                result = client.end_trial(EndTrialRequest(trial_id=trial.trial_id))
                if result.score >= 0:
                    scores.append((trial.task_id, result.score))
                    style = CLI_GREEN if result.score == 1 else CLI_RED
                    explain = textwrap.indent("\n".join(result.score_detail), "  ")
                    print(f"\n{style}Score: {result.score:0.2f}\n{explain}\n{CLI_CLR}")

        finally:
            client.submit_run(SubmitRunRequest(run_id=run.run_id, force=True))

    except ConnectError as exc:
        print(f"{CLI_RED}{exc.code}: {exc.message}{CLI_CLR}")
    except KeyboardInterrupt:
        print(f"\n{CLI_RED}Interrupted{CLI_CLR}")

    if scores:
        print(f"\n{'=' * 60}")
        for task_id, score in scores:
            style = CLI_GREEN if score == 1 else CLI_RED
            print(f"  {task_id}: {style}{score:0.2f}{CLI_CLR}")

        total = sum(score for _, score in scores) / len(scores) * 100.0
        print(f"\n  FINAL: {total:0.2f}%")


if __name__ == "__main__":
    main()
