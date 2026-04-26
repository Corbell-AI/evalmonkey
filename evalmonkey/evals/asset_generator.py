"""
evalmonkey.evals.asset_generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Persists failing benchmark traces and generates LLM-synthesised improvement
eval scenarios so developers can iteratively fix their agents.

Flow
----
1. During a benchmark/chaos run each failing trace is recorded via
   ``EvalAssetGenerator.record_failure()``.
2. After the run, if any failures exist, ``generate_improvement_evals()``
   asks the judge LLM to synthesise N new test cases that specifically target
   the identified weaknesses.
3. ``save(output_dir)`` writes three files:
       traces.json           — raw failing trace records
       evals.json            — new eval scenarios (Langfuse-compatible shape)
       improvement_prompt.md — a copy-pasteable prompt to hand to any coding agent
4. ``export_to_langfuse(dataset_name)`` optionally pushes the evals to a
   Langfuse dataset when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set.
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from evalmonkey.utils.llm import call_llm

# ─── Score threshold below which a trace is considered a failure ───────────
DEFAULT_FAILURE_THRESHOLD = int(os.getenv("EVAL_SCORE_THRESHOLD", "70"))


@dataclass
class FailingTrace:
    """One recorded failure from a benchmark or chaos run."""
    scenario: str
    eval_id: str
    input_payload: dict
    agent_output: str
    expected_rubric: str
    score: int
    reasoning: str
    chaos_profile: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "eval_id": self.eval_id,
            "input_payload": self.input_payload,
            "agent_output": self.agent_output,
            "expected_rubric": self.expected_rubric,
            "score": self.score,
            "reasoning": self.reasoning,
            "chaos_profile": self.chaos_profile,
            "timestamp": self.timestamp,
        }


class EvalAssetGenerator:
    """
    Accumulates failing traces during a benchmark session and generates
    targeted improvement eval scenarios using the configured LLM.
    """

    def __init__(
        self,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        model_name: Optional[str] = None,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.model_name = model_name or os.getenv("EVAL_MODEL", "gpt-4o")
        self._failures: List[FailingTrace] = []

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def record_failure(self, trace: FailingTrace) -> None:
        """Add a trace to the failure log if it is below the threshold."""
        if trace.score < self.failure_threshold:
            self._failures.append(trace)

    @property
    def has_failures(self) -> bool:
        return len(self._failures) > 0

    @property
    def failure_count(self) -> int:
        return len(self._failures)

    def generate_improvement_evals(self, n: int = 5) -> List[dict]:
        """
        Ask the configured LLM to synthesise ``n`` new test-case scenarios
        that specifically target the agent weaknesses identified in the
        failing traces.

        Returns a list of dicts with keys:
            id, description, input_payload, expected_behavior_rubric
        """
        if not self._failures:
            return []

        # Build a compact summary of failures for the LLM
        failure_summary = self._build_failure_summary()

        prompt = textwrap.dedent(f"""
            You are an expert agent quality engineer.

            The following agent benchmark traces FAILED (score < {self.failure_threshold}/100).
            Each entry shows: the question sent, the agent's actual response, what was expected, and the judge's reasoning.

            FAILING TRACES:
            {failure_summary}

            Your task: Generate exactly {n} NEW evaluation scenarios that specifically target the
            weaknesses revealed by these failures. Each scenario should probe a slightly different
            angle of the same weakness so that fixing the agent will require genuine improvement,
            not just memorising one answer.

            Return a JSON array of exactly {n} objects. Each object MUST have these keys:
            - "id": a short snake_case string (e.g. "math_carry_error_1")
            - "description": one sentence explaining what this test checks
            - "input_payload": dict with a single "question" key containing the test question
            - "expected_behavior_rubric": a one-sentence rubric the agent must satisfy to pass

            Return ONLY the JSON array, no other text.
        """).strip()

        try:
            response = call_llm(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            # LLM sometimes wraps the array in {"evals": [...]}
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
            # unwrap any top-level wrapper key
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            return []
        except Exception as e:
            # Graceful fallback: return a minimal stub so the rest of the
            # pipeline still works.
            return [
                {
                    "id": f"generated_eval_{i}",
                    "description": f"Generated eval targeting failure #{i}",
                    "input_payload": {"question": t.input_payload.get("question", "")},
                    "expected_behavior_rubric": t.expected_rubric,
                }
                for i, t in enumerate(self._failures[:n])
            ]

    def save(self, output_dir: str) -> str:
        """
        Persist failing traces, generated improvement evals, and a coding-agent
        improvement prompt to *output_dir*.

        Returns the absolute path of the output directory.
        """
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        # 1. Raw failing traces
        traces_path = path / "traces.json"
        traces_path.write_text(
            json.dumps([t.to_dict() for t in self._failures], indent=2),
            encoding="utf-8",
        )

        # 2. Synthesised improvement evals (Langfuse-compatible shape)
        improvement_evals = self.generate_improvement_evals()
        evals_path = path / "evals.json"
        evals_path.write_text(
            json.dumps(improvement_evals, indent=2),
            encoding="utf-8",
        )

        # 3. Coding-agent improvement prompt
        prompt_md = self._build_improvement_prompt_md(improvement_evals)
        prompt_path = path / "improvement_prompt.md"
        prompt_path.write_text(prompt_md, encoding="utf-8")

        return str(path.resolve())

    def export_to_langfuse(self, dataset_name: str) -> bool:
        """
        Push generated evals to a Langfuse dataset via their REST API.
        Silently skips when credentials are not configured.

        Returns True if export succeeded, False otherwise.
        """
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not (public_key and secret_key):
            return False  # Langfuse not configured — skip silently

        try:
            import requests  # included in litellm's deps

            evals = self.generate_improvement_evals()
            if not evals:
                return False

            auth = (public_key, secret_key)

            # Ensure the dataset exists
            requests.post(
                f"{host}/api/public/datasets",
                json={"name": dataset_name},
                auth=auth,
                timeout=15,
            )

            # Push each eval as a dataset item
            for ev in evals:
                requests.post(
                    f"{host}/api/public/dataset-items",
                    json={
                        "datasetName": dataset_name,
                        "input": ev.get("input_payload", {}),
                        "expectedOutput": ev.get("expected_behavior_rubric", ""),
                        "metadata": {"id": ev.get("id", ""), "description": ev.get("description", "")},
                    },
                    auth=auth,
                    timeout=15,
                )
            return True
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_failure_summary(self) -> str:
        lines = []
        for i, t in enumerate(self._failures, 1):
            question = t.input_payload.get("question", str(t.input_payload))[:300]
            output_snippet = t.agent_output[:300]
            lines.append(
                f"[Failure {i}] Score={t.score}/100 | Chaos={t.chaos_profile or 'none'}\n"
                f"  Question  : {question}\n"
                f"  Agent said: {output_snippet}\n"
                f"  Expected  : {t.expected_rubric[:200]}\n"
                f"  Reasoning : {t.reasoning[:200]}\n"
            )
        return "\n".join(lines)

    def _build_improvement_prompt_md(self, improvement_evals: List[dict]) -> str:
        evals_block = json.dumps(improvement_evals, indent=2)
        n_failures = len(self._failures)
        scenarios = {t.scenario for t in self._failures}
        chaoses = {t.chaos_profile for t in self._failures if t.chaos_profile}

        return textwrap.dedent(f"""
            # EvalMonkey Agent Improvement Brief

            ## Summary
            - **{n_failures} failing trace(s)** captured during benchmarking
            - **Scenarios tested**: {', '.join(scenarios)}
            - **Chaos profiles that caused failures**: {', '.join(chaoses) if chaoses else 'none'}

            ## What went wrong
            {self._build_failure_summary()}

            ## New Evaluation Scenarios to Fix
            The following {len(improvement_evals)} targeted test cases were synthesised to
            specifically probe the weaknesses above. Your agent must pass all of them.

            ```json
            {evals_block}
            ```

            ## Next steps for your coding agent
            Copy and paste the following instruction to Claude Code, Cursor, or any AI coding assistant:

            ---
            **Coding Agent Prompt:**

            I have an AI agent that is failing the following benchmark evaluation tests.
            Please analyse the failures and improve my agent's code so it passes all of them.

            Failing patterns:
            {self._build_failure_summary()}

            New test cases to pass (in evalmonkey evals.json format):
            ```json
            {evals_block}
            ```

            To verify your fix locally, run:
            ```bash
            evalmonkey run-benchmark --eval-file output/evals.json --scenario <your_scenario_id>
            ```
            ---
        """).strip()


def build_output_dir(scenario: str, base: str = "output") -> str:
    """Return a time-stamped output directory path for a given scenario."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, f"{scenario}_{ts}")
