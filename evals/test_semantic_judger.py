from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import pytest
from core.composition_root import build_orchestrator
from core.orchestrator import Orchestrator
from core.types.dataclasses import JudgeIssue
from .cases import GOLDEN_CASES, METAMORPHIC_PAIRS, EvalCase

pytestmark = pytest.mark.eval

RESULTS_DIR = Path(__file__).parent / "results"
CONSISTENCY_REPEATS = 5

# Quality bars. Tune these once you have a baseline; they exist so a
# regression in judge quality fails the eval instead of going unnoticed.
MIN_RECALL = 0.8
MAX_FALSE_POSITIVE_RATE = 0.2


@pytest.fixture(scope="session")
def orchestrator() -> Orchestrator:
    """Build the real orchestrator via the composition root, so this eval
    stays in sync with production wiring automatically."""
    return build_orchestrator()


def _run_case(orchestrator: Orchestrator, case: EvalCase) -> list[JudgeIssue]:
    """Run a single case and return the full list of issues the judge raised
    (not just their categories), so reports can preserve the "reason" text
    -- essential for understanding *why* a category was or wasn't flagged
    without having to re-read raw stdout from Judger's own print calls.

    Mirrors Orchestrator.generate_dialogue's exact client-selection
    sequence rather than using one fixed client for the whole eval:
    LLMRouter.select_model can route to a different model depending on
    game_context/npc_context, so replicating that call per case keeps the
    eval representative of what a real request for that scenario would
    actually use, instead of testing every case against an arbitrarily
    chosen single model.
    """
    client = orchestrator.llm_router.select_model(
        game_context=case.game_context, npc_context=case.npc_context
    )
    orchestrator.judger.set_client(client)

    return orchestrator.judger.judge_dialogue(case.composed_dialogue, case.npc_context, case.game_context)


def _issues_to_report(issues: list[JudgeIssue]) -> list[dict]:
    """Serialize issues as {category, reason} dicts for JSON reports."""
    return [{"category": i.category, "reason": i.issue} for i in issues]


def _write_report(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{timestamp}_{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def test_semantic_recall_and_precision(orchestrator):
    """Core eval: recall per category on violation cases, false-positive
    rate per category on clean cases."""
    hits: dict[str, int] = defaultdict(int)
    expected_totals: dict[str, int] = defaultdict(int)
    false_positives: dict[str, int] = defaultdict(int)
    clean_totals = 0
    per_case_results = []

    for case in GOLDEN_CASES:
        issues = _run_case(orchestrator, case)
        flagged = {issue.category for issue in issues}
        per_case_results.append({
            "id": case.id,
            "expected": sorted(case.expected_categories),
            "flagged": _issues_to_report(issues),
        })

        if not case.expected_categories:
            clean_totals += 1
            for category in flagged:
                false_positives[category] += 1
        else:
            for category in case.expected_categories:
                expected_totals[category] += 1
                if category in flagged:
                    hits[category] += 1

    recall = {
        category: hits[category] / expected_totals[category]
        for category in expected_totals
    }
    false_positive_rate = {
        category: count / clean_totals if clean_totals else 0.0
        for category, count in false_positives.items()
    }

    report_path = _write_report("recall_precision", {
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "per_case": per_case_results,
    })

    print(f"\nFull report written to: {report_path}")
    print("\n=== Semantic judge recall per category ===")
    for category in sorted(expected_totals):
        print(f"  {category}: {recall[category]:.0%} ({hits[category]}/{expected_totals[category]})")

    print("\n=== False positive rate per category (on clean cases) ===")
    if not false_positive_rate:
        print("  (none — no clean case was ever flagged)")
    for category, value in sorted(false_positive_rate.items()):
        print(f"  {category}: {value:.0%}")

    low_recall = {c: v for c, v in recall.items() if v < MIN_RECALL}
    high_fp = {c: v for c, v in false_positive_rate.items() if v > MAX_FALSE_POSITIVE_RATE}

    assert not low_recall, f"Categories below minimum recall ({MIN_RECALL:.0%}): {low_recall}"
    assert not high_fp, f"Categories above max false-positive rate ({MAX_FALSE_POSITIVE_RATE:.0%}): {high_fp}"


def test_semantic_metamorphic_pairs(orchestrator):
    """For each (clean, mutated) pair: the clean version must pass, and
    the mutated version must specifically fail the expected category."""
    failures: list[str] = []
    per_pair_results = []

    for pair in METAMORPHIC_PAIRS:
        clean_issues = _run_case(orchestrator, pair.clean_case)
        mutated_issues = _run_case(orchestrator, pair.mutated_case)
        clean_flagged = {issue.category for issue in clean_issues}
        mutated_flagged = {issue.category for issue in mutated_issues}

        per_pair_results.append({
            "id": pair.id,
            "expected_category": pair.expected_category,
            "clean_flagged": _issues_to_report(clean_issues),
            "mutated_flagged": _issues_to_report(mutated_issues),
        })

        if clean_flagged:
            failures.append(f"[{pair.id}] clean case unexpectedly flagged: {sorted(clean_flagged)}")
        if pair.expected_category not in mutated_flagged:
            failures.append(
                f"[{pair.id}] mutation did not trigger '{pair.expected_category}', "
                f"got: {sorted(mutated_flagged)}"
            )

    _write_report("metamorphic", {"pairs": per_pair_results, "failures": failures})

    assert not failures, "Metamorphic test failures:\n" + "\n".join(failures)


def test_semantic_consistency(orchestrator):
    """Run a single case N times and check the per-category verdict
    doesn't flip-flop under the configured judge temperature.

    Note: if LLMRouter.select_model has any internal state (e.g.
    load-balancing across equivalent models), repeated calls with the
    identical game_context/npc_context could in principle route to
    different underlying models across runs, adding a second source of
    variance on top of sampling temperature. If instability shows up
    here, check which model was actually selected each time before
    concluding the question wording itself is ambiguous.
    """
    case = next((c for c in GOLDEN_CASES if c.id == "consistency_probe"), None)
    if case is None:
        pytest.skip("No case with id 'consistency_probe' in the dataset.")

    runs = [_run_case(orchestrator, case) for _ in range(CONSISTENCY_REPEATS)]
    run_categories = [{issue.category for issue in run} for run in runs]

    all_categories = set().union(*run_categories) if run_categories else set()
    instability = {}
    for category in all_categories:
        votes = [category in run for run in run_categories]
        agreement = max(votes.count(True), votes.count(False)) / len(votes)
        if agreement < 1.0:
            instability[category] = agreement

    _write_report("consistency", {
        "case_id": case.id,
        "runs": [_issues_to_report(run) for run in runs],
        "instability": instability,
    })

    print(f"\n=== Consistency across {CONSISTENCY_REPEATS} runs on '{case.id}' ===")
    if not instability:
        print("  Fully stable: every category agreed across all runs.")
    for category, agreement in sorted(instability.items()):
        print(f"  {category}: {agreement:.0%} agreement (unstable)")

    # Soft check by design: flag instability for human review rather than
    # hard-failing the build, since some wobble on borderline cases is
    # expected. Tighten this into a hard assert once you have a baseline
    # for how much instability is tolerable.
    if instability:
        print(
            "\nNOTE: some categories are not stable across repeated runs at "
            "the configured temperature. Consider tightening the wording of "
            "the corresponding question(s) in Judger._build_questions."
        )