#!/usr/bin/env python3
"""Compare chunking-strategy evaluation results from checkpoint files.

Reads checkpoint_fixed.json, checkpoint_structure.json, and
checkpoint_semantic.json from data/eval/. For each strategy it displays:
  - N (number of evaluations completed)
  - per-metric averages (retrieval_relevance, answer_correctness, faithfulness, citation_accuracy)
  - partial result warning if N < expected subset size

Computes paired wins/losses per metric across strategies for questions that
appear in all checkpoints. Refuses to declare an overall winner if N is
mismatched across strategies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "data/eval"
STRATEGIES = ("fixed", "structure", "semantic")
SCORE_KEYS = (
    "retrieval_relevance",
    "answer_correctness",
    "faithfulness",
    "citation_accuracy",
)
EXPECTED_N = 30


def load_checkpoint(strategy: str) -> Dict[str, Dict[str, int]]:
    path = CHECKPOINT_DIR / f"checkpoint_{strategy}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Dict[str, int]] = {}
    for key, value in data.items():
        if isinstance(value, dict) and all(k in value for k in SCORE_KEYS):
            out[key] = {k: int(value[k]) for k in SCORE_KEYS}
    return out


def average(scores: List[int]) -> float:
    return sum(scores) / len(scores) * 100 if scores else 0.0


def paired_comparison(
    checkpoints: Dict[str, Dict[str, Dict[str, int]]],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """For each pair of strategies, count wins/losses/ties per metric on
    questions present in both checkpoints."""
    pairs: Dict[str, Dict[str, Dict[str, int]]] = {}
    strategies = list(checkpoints.keys())
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i + 1 :]:
            c1 = checkpoints[s1]
            c2 = checkpoints[s2]
            common_qids = set(c1.keys()) & set(c2.keys())
            pair_result: Dict[str, Dict[str, int]] = {}
            for key in SCORE_KEYS:
                wins = losses = ties = 0
                for qid in common_qids:
                    v1 = c1[qid][key]
                    v2 = c2[qid][key]
                    if v1 > v2:
                        wins += 1
                    elif v1 < v2:
                        losses += 1
                    else:
                        ties += 1
                pair_result[key] = {
                    f"{s1}_wins": wins,
                    f"{s2}_wins": losses,
                    "ties": ties,
                    "n_common": len(common_qids),
                }
            pairs[f"{s1}_vs_{s2}"] = pair_result
    return pairs


def main() -> int:
    checkpoints: Dict[str, Dict[str, Dict[str, int]]] = {}
    for strategy in STRATEGIES:
        checkpoints[strategy] = load_checkpoint(strategy)

    ns = {s: len(checkpoints[s]) for s in STRATEGIES}
    all_equal = len(set(ns.values())) == 1
    all_full = all(n >= EXPECTED_N for n in ns.values())

    print("=== Chunking Strategy Comparison ===")
    print()
    print(f"{'Strategy':<14} {'N':>4} {'Retrieval':>10} {'Correctness':>12} {'Faithfulness':>12} {'Citation':>10}")
    print("-" * 70)
    for s in STRATEGIES:
        cp = checkpoints[s]
        n = ns[s]
        if n == 0:
            print(f"{s:<14} {n:>4} {'--':>10} {'--':>12} {'--':>12} {'--':>10}  (no data)")
            continue
        avg = {k: average([r[k] for r in cp.values()]) for k in SCORE_KEYS}
        partial_mark = " (PARTIAL)" if n < EXPECTED_N else ""
        print(
            f"{s:<14} {n:>4} {avg['retrieval_relevance']:>9.1f}% "
            f"{avg['answer_correctness']:>11.1f}% {avg['faithfulness']:>11.1f}% "
            f"{avg['citation_accuracy']:>9.1f}%{partial_mark}"
        )
    print()

    # Paired comparison
    print("=== Paired Wins/Losses (questions in common) ===")
    pairs = paired_comparison(checkpoints)
    for pair_name, metrics in pairs.items():
        s1, s2 = pair_name.split("_vs_")
        n_common = metrics["retrieval_relevance"]["n_common"]
        print(f"\n{s1} vs {s2} (common questions: {n_common})")
        for key in SCORE_KEYS:
            m = metrics[key]
            print(
                f"  {key:<20} {s1}: {m[f'{s1}_wins']:>3}  {s2}: {m[f'{s2}_wins']:>3}  ties: {m['ties']:>3}"
            )
    print()

    # Winner declaration logic
    if not all_equal:
        print("RESULT: Cannot declare overall winner — N values are mismatched:")
        for s in STRATEGIES:
            print(f"  {s}: N={ns[s]}")
    elif not all_full:
        print("RESULT: Partial results — not all strategies have reached N=30:")
        for s in STRATEGIES:
            mark = "FULL" if ns[s] >= EXPECTED_N else f"PARTIAL (N={ns[s]})"
            print(f"  {s}: {mark}")
    else:
        # Declare winner by majority of metrics won
        wins: Dict[str, int] = {s: 0 for s in STRATEGIES}
        for metrics in pairs.values():
            for key in SCORE_KEYS:
                m = metrics[key]
                s1_key = f"{list(STRATEGIES)[0]}_wins"  # placeholder, replaced below
        # Recompute wins properly
        pair_list = list(pairs.items())
        for pair_name, metrics in pair_list:
            s1, s2 = pair_name.split("_vs_")
            for key in SCORE_KEYS:
                m = metrics[key]
                if m[f"{s1}_wins"] > m[f"{s2}_wins"]:
                    wins[s1] += 1
                elif m[f"{s2}_wins"] > m[f"{s1}_wins"]:
                    wins[s2] += 1
        best = max(wins, key=wins.get)
        print(f"RESULT: Overall winner: {best} (won {wins[best]}/{len(SCORE_KEYS)*len(pair_list)} metric-pair matchups)")
        for s in STRATEGIES:
            print(f"  {s}: {wins[s]} metric wins")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
