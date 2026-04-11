"""
Eval – Runner
Loads eval/dataset.json, runs each FunctionSlice through evaluate_slice(),
then scores every verdict with the LLM-as-a-judge metrics.

Exit codes:
    0  all thresholds passed
    1  one or more quality gates failed (use as CI gate)

Usage:
    python eval/eval_runner.py
    python eval/eval_runner.py --model anthropic/claude-3-5-sonnet-20241022
    python eval/eval_runner.py --verdict-accuracy-threshold 0.9
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ast_extractor import FunctionSlice
from llm_evaluator import SecurityVerdict, _build_user_message, evaluate_slice
from eval.metrics import (
    BlockJustificationMetric,
    HallucinationMetric,
    VulnClassificationMetric,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "dataset.json"

# ---------------------------------------------------------------------------
# Thresholds — override via CLI flags or env vars
# ---------------------------------------------------------------------------

DEFAULT_VERDICT_ACCURACY_THRESHOLD = 0.85  # 85% binary BLOCK/PASS accuracy
DEFAULT_PRECISION_THRESHOLD = 0.80         # TP / (TP + FP) — are BLOCK calls correct?
DEFAULT_RECALL_THRESHOLD = 0.90            # TP / (TP + FN) — stricter: missing vulns is worse
DEFAULT_HALLUCINATION_THRESHOLD = 0.6      # mean judge score
DEFAULT_BLOCK_JUSTIFICATION_THRESHOLD = 0.6   # mean judge: are BLOCK calls justified?
DEFAULT_VULN_CLASSIFICATION_THRESHOLD = 0.7   # mean judge: did model identify the correct vulnerability class?


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    expected_verdict: str
    actual_verdict: str
    expected_vulnerability_type: str | None  # human label from dataset
    correct: bool
    reasoning: str
    vulnerability_type: str | None           # model's predicted type
    confidence: float
    llm_test_case: LLMTestCase

    @property
    def is_tp(self) -> bool:
        return self.expected_verdict == "BLOCK" and self.actual_verdict == "BLOCK"

    @property
    def is_fp(self) -> bool:
        return self.expected_verdict == "PASS" and self.actual_verdict == "BLOCK"

    @property
    def is_fn(self) -> bool:
        return self.expected_verdict == "BLOCK" and self.actual_verdict == "PASS"

    @property
    def vuln_type_correct(self) -> bool | None:
        """None for non-BLOCK cases. True/False for TP cases only."""
        if not self.is_tp:
            return None
        if self.expected_vulnerability_type is None or self.vulnerability_type is None:
            return None
        return self.expected_vulnerability_type.lower() == self.vulnerability_type.lower()


@dataclass
class ConfusionMatrix:
    """
    Binary confusion matrix for the BLOCK/PASS classification task.
    Positive class = BLOCK (a real vulnerability).

        predicted BLOCK  predicted PASS
    label BLOCK     TP               FN
    label PASS      FP               TN
    """
    tp: int = field(default=0)  # label BLOCK, predicted BLOCK
    fp: int = field(default=0)  # label PASS,  predicted BLOCK
    tn: int = field(default=0)  # label PASS,  predicted PASS
    fn: int = field(default=0)  # label BLOCK, predicted PASS

    @classmethod
    def from_results(cls, results: list[CaseResult]) -> ConfusionMatrix:
        cm = cls()
        for r in results:
            if r.expected_verdict == "BLOCK" and r.actual_verdict == "BLOCK":
                cm.tp += 1
            elif r.expected_verdict == "PASS" and r.actual_verdict == "BLOCK":
                cm.fp += 1
            elif r.expected_verdict == "PASS" and r.actual_verdict == "PASS":
                cm.tn += 1
            else:  # expected BLOCK, predicted PASS
                cm.fn += 1
        return cm

    @property
    def precision(self) -> float | None:
        """TP / (TP + FP) — of all predicted BLOCKs, how many were real?"""
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        """TP / (TP + FN) — of all real BLOCKs, how many did we catch?"""
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def f1(self) -> float | None:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def _build_actual_output(verdict: SecurityVerdict) -> str:
    """Combine verdict label and reasoning into a single string for the judge."""
    label = f"Verdict: {verdict.verdict}"
    vuln = f"Vulnerability: {verdict.vulnerability_type}" if verdict.vulnerability_type else ""
    parts = [p for p in [label, vuln, verdict.reasoning] if p]
    return "\n".join(parts)


def _run_production_model(cases: list[dict], model: str | None) -> list[CaseResult]:
    results: list[CaseResult] = []

    for case in cases:
        case_id = case["id"]
        logger.info("Running production model on case %s …", case_id)

        fn_slice = FunctionSlice(**case["input"])
        kwargs = {"model": model} if model else {}

        try:
            verdict = evaluate_slice(fn_slice, **kwargs)
        except Exception as exc:
            logger.error("Case %s failed: %s", case_id, exc)
            continue

        prompt = _build_user_message(fn_slice)
        actual_output = _build_actual_output(verdict)

        results.append(CaseResult(
            case_id=case_id,
            expected_verdict=case["expected_verdict"],
            actual_verdict=verdict.verdict,
            expected_vulnerability_type=case.get("expected_vulnerability_type"),
            correct=verdict.verdict == case["expected_verdict"],
            reasoning=verdict.reasoning,
            vulnerability_type=verdict.vulnerability_type,
            confidence=verdict.confidence,
            llm_test_case=LLMTestCase(
                input=prompt,
                actual_output=actual_output,
                context=[fn_slice.raw_source],
                expected_output=case["expected_verdict"],
            ),
        ))

    return results


def _print_verdict_report(results: list[CaseResult]) -> tuple[float, ConfusionMatrix]:
    correct = sum(r.correct for r in results)
    accuracy = correct / len(results) if results else 0.0
    cm = ConfusionMatrix.from_results(results)

    def _fmt(v: float | None) -> str:
        return f"{v:.1%}" if v is not None else "n/a"

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(f"  Total cases : {len(results)}")
    print(f"  Accuracy    : {accuracy:.1%}  ({correct}/{len(results)})")
    print()
    print("  Confusion matrix (positive class = BLOCK):")
    print(f"    TP={cm.tp}  FP={cm.fp}  TN={cm.tn}  FN={cm.fn}")
    print()
    print(f"  Precision   : {_fmt(cm.precision)}  (TP / predicted BLOCK)")
    print(f"  Recall      : {_fmt(cm.recall)}  (TP / labeled BLOCK)")
    print(f"  F1          : {_fmt(cm.f1)}")
    print()

    # Vulnerability type accuracy — deterministic, only meaningful on TP cases.
    tp_with_type = [r for r in results if r.is_tp and r.vuln_type_correct is not None]
    if tp_with_type:
        type_correct = sum(1 for r in tp_with_type if r.vuln_type_correct)
        type_acc = type_correct / len(tp_with_type)
        print(f"  Vuln type accuracy : {type_acc:.1%}  ({type_correct}/{len(tp_with_type)} TP cases)")
        type_mismatches = [r for r in tp_with_type if not r.vuln_type_correct]
        if type_mismatches:
            for r in type_mismatches:
                print(f"    [WRONG TYPE] [{r.case_id}]  expected={r.expected_vulnerability_type}  "
                      f"got={r.vulnerability_type}")
        print()

    failures = [r for r in results if not r.correct]
    if failures:
        print("  FAILURES:")
        for r in failures:
            label = "FP" if r.is_fp else "FN"
            print(f"    [{label}] [{r.case_id}]  expected={r.expected_verdict}  got={r.actual_verdict}")
            print(f"           reasoning: {r.reasoning[:120]} …")
    print("=" * 60 + "\n")

    return accuracy, cm


def _extract_scores(deepeval_results, metric_name: str) -> list[float]:
    """Pull per-case scores for one metric from a DeepEval results object."""
    scores: list[float] = []
    if deepeval_results is None:
        return scores
    for tc in deepeval_results.test_results:
        for metric_data in tc.metrics_data:
            if metric_data.name == metric_name:
                scores.append(metric_data.score or 0.0)
    return scores


def _check_gates(
    verdict_accuracy: float,
    cm: ConfusionMatrix,
    deepeval_results,        # all-cases results (Hallucination)
    tp_deepeval_results,     # TP-only results (BlockJustification, VulnClassification)
    verdict_threshold: float,
    precision_threshold: float,
    recall_threshold: float,
    hallucination_threshold: float,
    block_justification_threshold: float,
    vuln_classification_threshold: float,
) -> bool:
    passed = True

    # ── Deterministic gates ───────────────────────────────────────────────────
    if verdict_accuracy < verdict_threshold:
        logger.error(
            "GATE FAILED — verdict accuracy %.1f%% < threshold %.1f%%",
            verdict_accuracy * 100, verdict_threshold * 100,
        )
        passed = False
    else:
        logger.info("GATE PASSED — verdict accuracy %.1f%%", verdict_accuracy * 100)

    if cm.precision is not None and cm.precision < precision_threshold:
        logger.error(
            "GATE FAILED — precision %.1f%% < threshold %.1f%% (%d false positives)",
            cm.precision * 100, precision_threshold * 100, cm.fp,
        )
        passed = False
    elif cm.precision is not None:
        logger.info("GATE PASSED — precision %.1f%%", cm.precision * 100)

    if cm.recall is not None and cm.recall < recall_threshold:
        logger.error(
            "GATE FAILED — recall %.1f%% < threshold %.1f%% (%d missed vulnerabilities)",
            cm.recall * 100, recall_threshold * 100, cm.fn,
        )
        passed = False
    elif cm.recall is not None:
        logger.info("GATE PASSED — recall %.1f%%", cm.recall * 100)

    # ── LLM judge gates ───────────────────────────────────────────────────────
    # Hallucination: judge ran on all cases.
    hallucination_scores = _extract_scores(deepeval_results, "Hallucination")
    if hallucination_scores:
        mean_h = sum(hallucination_scores) / len(hallucination_scores)
        if mean_h < hallucination_threshold:
            logger.error("GATE FAILED — mean hallucination score %.2f < threshold %.2f", mean_h, hallucination_threshold)
            passed = False
        else:
            logger.info("GATE PASSED — mean hallucination score %.2f", mean_h)

    # BlockJustification: judge ran only on TP cases; FP cases contribute 0.0.
    tp_bj_scores = _extract_scores(tp_deepeval_results, "BlockJustification")
    bj_scores = tp_bj_scores + [0.0] * cm.fp   # FPs scored deterministically
    if bj_scores:
        mean_b = sum(bj_scores) / len(bj_scores)
        if mean_b < block_justification_threshold:
            logger.error(
                "GATE FAILED — mean block-justification score %.2f < threshold %.2f "
                "(includes %d FP cases scored 0.0)",
                mean_b, block_justification_threshold, cm.fp,
            )
            passed = False
        else:
            logger.info("GATE PASSED — mean block-justification score %.2f", mean_b)

    # VulnClassification: judge ran only on TP cases; FN cases contribute 0.0.
    tp_vc_scores = _extract_scores(tp_deepeval_results, "VulnClassification")
    vc_scores = tp_vc_scores + [0.0] * cm.fn  # FNs scored deterministically
    if vc_scores:
        mean_v = sum(vc_scores) / len(vc_scores)
        if mean_v < vuln_classification_threshold:
            logger.error(
                "GATE FAILED — mean vuln-classification score %.2f < threshold %.2f "
                "(includes %d FN cases scored 0.0)",
                mean_v, vuln_classification_threshold, cm.fn,
            )
            passed = False
        else:
            logger.info("GATE PASSED — mean vuln-classification score %.2f", mean_v)

    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PR Security Agent — eval runner")
    parser.add_argument("--model", default=None, help="litellm model string for the production model")
    parser.add_argument("--verdict-accuracy-threshold", type=float, default=DEFAULT_VERDICT_ACCURACY_THRESHOLD)
    parser.add_argument("--precision-threshold", type=float, default=DEFAULT_PRECISION_THRESHOLD,
                        help="Min precision: TP / (TP + FP)")
    parser.add_argument("--recall-threshold", type=float, default=DEFAULT_RECALL_THRESHOLD,
                        help="Min recall: TP / (TP + FN)")
    parser.add_argument("--hallucination-threshold", type=float, default=DEFAULT_HALLUCINATION_THRESHOLD)
    parser.add_argument(
        "--block-justification-threshold",
        type=float,
        default=DEFAULT_BLOCK_JUSTIFICATION_THRESHOLD,
        help="Mean judge score for BlockJustification metric (are BLOCK calls justified?)",
    )
    parser.add_argument(
        "--vuln-classification-threshold",
        type=float,
        default=DEFAULT_VULN_CLASSIFICATION_THRESHOLD,
        help="Mean judge score for VulnClassification (did model identify the correct vulnerability class?)",
    )
    args = parser.parse_args()

    dataset = _load_dataset()
    logger.info("Loaded %d test cases from %s", len(dataset), DATASET_PATH)

    # Step 1: run production model over all cases
    results = _run_production_model(dataset, args.model)
    if not results:
        logger.error("No results — aborting.")
        sys.exit(1)

    # Step 2: report classification metrics (cheap, deterministic — no judge needed)
    verdict_accuracy, cm = _print_verdict_report(results)

    # Step 3: run LLM-as-a-judge metrics.
    # - HallucinationMetric runs on all cases (reasoning quality is universal).
    # - BlockJustificationMetric and VulnClassificationMetric run only on TP cases
    #   (label=BLOCK, actual=BLOCK). FP and FN scores are assigned deterministically.
    tp_results = [r for r in results if r.is_tp]
    logger.info(
        "Running LLM judge: %d total cases → %d TP cases sent to judge "
        "(%d FP and %d FN scored deterministically)",
        len(results), len(tp_results), cm.fp, cm.fn,
    )
    deepeval_results = evaluate(
        test_cases=[r.llm_test_case for r in results],        # Hallucination: all cases
        metrics=[HallucinationMetric],
        run_async=True,
        show_indicator=True,
    )
    tp_deepeval_results = evaluate(
        test_cases=[r.llm_test_case for r in tp_results],    # TP-only: reasoning quality
        metrics=[BlockJustificationMetric, VulnClassificationMetric],
        run_async=True,
        show_indicator=True,
    ) if tp_results else None

    # Step 4: apply quality gates and exit accordingly
    passed = _check_gates(
        verdict_accuracy=verdict_accuracy,
        cm=cm,
        deepeval_results=deepeval_results,
        tp_deepeval_results=tp_deepeval_results,
        verdict_threshold=args.verdict_accuracy_threshold,
        precision_threshold=args.precision_threshold,
        recall_threshold=args.recall_threshold,
        hallucination_threshold=args.hallucination_threshold,
        block_justification_threshold=args.block_justification_threshold,
        vuln_classification_threshold=args.vuln_classification_threshold,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
