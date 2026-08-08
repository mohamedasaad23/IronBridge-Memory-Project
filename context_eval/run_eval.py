"""
Run: python -m context_eval.run_eval

Scores all four context-management strategies against the fixed test
suite in test_transcripts/generate.py. For each (strategy, transcript)
pair:
  - recall: does the critical fact survive in the pruned output?
    (checked via the exact critical_marker string OR the
    expected_answer_contains keyword — the marker proves literal
    survival, the keyword proves paraphrased survival counts too)
  - input tokens: approx. word count of the pruned messages sent back in
  - output tokens: approx. word count only summarization.py actually
    produces (an LLM call) — the other three strategies are 0 here,
    which is the whole point of the comparison
  - latency: wall-clock time of the strategy call itself

This table is what the README cites to justify the strategy actually
shipped — per the lab's guardrail, don't change the test suite after
you've started scoring.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .strategies import sliding_window, masking, summarization, zone_pruning
from .test_transcripts.generate import build_test_suite

STRATEGIES: dict[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] = {
    "sliding_window": lambda msgs: sliding_window.apply(msgs, keep_turns=10),
    "observation_masking": lambda msgs: masking.apply(msgs, keep_recent_tool_outputs=3),
    "recursive_summarization": lambda msgs: summarization.apply(msgs, keep_recent=8),
    "zone_based_pruning": lambda msgs: zone_pruning.apply(msgs),
}


def _approx_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(len(m["content"].split()) for m in messages)


def _recall_hit(pruned: list[dict[str, Any]], marker: str, keyword: str) -> bool:
    blob = " ".join(m["content"] for m in pruned).lower()
    return marker.lower() in blob or keyword.lower() in blob


def run() -> list[dict[str, Any]]:
    suite = build_test_suite()
    results = []

    for strategy_name, fn in STRATEGIES.items():
        hits = 0
        total_in_tokens = 0
        total_out_tokens = 0
        total_latency = 0.0

        for case in suite:
            before_tokens = _approx_tokens(case["messages"])
            t0 = time.perf_counter()
            pruned = fn(case["messages"])
            latency = time.perf_counter() - t0

            after_tokens = _approx_tokens(pruned)
            # For summarization, the "output" is the newly generated
            # summary text — everything else strategies produce 0 new
            # tokens, they only select/mask existing ones.
            out_tokens = 0
            if strategy_name == "recursive_summarization":
                new_system_lines = [
                    m["content"] for m in pruned if m["content"].startswith("Earlier context:")
                ]
                out_tokens = sum(len(t.split()) for t in new_system_lines)

            hit = _recall_hit(pruned, case["critical_marker"], case["expected_answer_contains"])
            hits += int(hit)
            total_in_tokens += after_tokens
            total_out_tokens += out_tokens
            total_latency += latency

        n = len(suite)
        results.append(
            {
                "strategy": strategy_name,
                "recall": f"{hits}/{n}",
                "avg_input_tokens": round(total_in_tokens / n),
                "avg_output_tokens": round(total_out_tokens / n),
                "avg_latency_ms": round((total_latency / n) * 1000, 1),
            }
        )
    return results


def print_table(results: list[dict[str, Any]]) -> None:
    header = f"{'Strategy':<26}{'Recall':<10}{'AvgInTok':<12}{'AvgOutTok':<12}{'AvgLatency(ms)':<15}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['strategy']:<26}{r['recall']:<10}{r['avg_input_tokens']:<12}"
            f"{r['avg_output_tokens']:<12}{r['avg_latency_ms']:<15}"
        )


if __name__ == "__main__":
    results = run()
    print_table(results)