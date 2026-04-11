"""
Eval – Custom DeepEval Metrics
Three GEval metrics that use a capable judge model to score SecurityVerdict outputs.

HallucinationMetric       – is the reasoning grounded in the real code? (explanation quality)
BlockJustificationMetric  – when the model says BLOCK, is that call justified? (over-alert guard)
LabeledThreatCatchMetric  – when the human label is BLOCK, did the model catch the threat well?
                            (missed-vuln / weak-detection guard — not formal recall)

All use Chain-of-Thought reasoning before assigning a 0–1 score, which forces
the judge to justify its decision and reduces positional/length bias.
"""

from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# ---------------------------------------------------------------------------
# Judge model
# ---------------------------------------------------------------------------
# Uses the DEEPEVAL_MODEL env var if set, otherwise falls back to gpt-4o.
# Override via: DEEPEVAL_MODEL=gpt-4o deepeval test run ...

JUDGE_MODEL = "gpt-4o"

# ---------------------------------------------------------------------------
# Hallucination metric
# ---------------------------------------------------------------------------
# Detects when the model's reasoning invents variables, function calls, or
# logic that does not appear in the actual raw_source provided as context.
#
# LLMTestCaseParams mapping:
#   INPUT          → the full prompt sent to the production model
#   ACTUAL_OUTPUT  → SecurityVerdict.reasoning (the text to ground-check)
#   CONTEXT        → [raw_source] (the ground truth the reasoning must be grounded in)

HallucinationMetric = GEval(
    name="Hallucination",
    model=JUDGE_MODEL,
    criteria=(
        "Determine whether the ACTUAL OUTPUT (the model's reasoning) only references "
        "identifiers, variable names, function calls, and logic that are explicitly "
        "present in the CONTEXT (the raw source code of the function). "
        "The reasoning must not invent symbols, call paths, or data flows that do "
        "not exist in the provided code. "
        "A score of 1.0 means the reasoning is fully grounded in the actual code. "
        "A score of 0.0 means the reasoning fabricates at least one identifier or "
        "code path that does not exist in the context."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT,
    ],
    threshold=0.6,
)

# ---------------------------------------------------------------------------
# Block justification metric  (false-positive guard)
# ---------------------------------------------------------------------------
# When the model predicts BLOCK, was that call correct and well-reasoned?
# Uses EXPECTED_OUTPUT (ground truth label) so confirmed false positives
# (BLOCK on a PASS-labeled case) are always penalised, even when the
# reasoning sounds convincing.
#
# Scoring guidance:
#   • ACTUAL_OUTPUT is PASS → neutral 0.7 (nothing to justify).
#   • ACTUAL_OUTPUT is BLOCK, EXPECTED_OUTPUT is BLOCK → score by quality
#     of the exploit reasoning (1.0 specific, 0.5 vague).
#   • ACTUAL_OUTPUT is BLOCK, EXPECTED_OUTPUT is PASS → score 0.0 regardless
#     of how plausible the reasoning sounds — this is a confirmed false positive.
#
# LLMTestCaseParams mapping:
#   INPUT           → the full prompt (includes raw_source and changed_lines)
#   ACTUAL_OUTPUT   → SecurityVerdict.reasoning + verdict label
#   EXPECTED_OUTPUT → human-verified ground truth ("BLOCK" or "PASS")

BlockJustificationMetric = GEval(
    name="BlockJustification",
    model=JUDGE_MODEL,
    criteria=(
        "Evaluate whether a BLOCK verdict from the model was correct and well-reasoned. "
        "The EXPECTED OUTPUT is the human-verified ground truth ('BLOCK' or 'PASS'). "
        "If the ACTUAL OUTPUT verdict is PASS, assign 0.7 — no BLOCK to justify. "
        "If the ACTUAL OUTPUT verdict is BLOCK and the EXPECTED OUTPUT is PASS, "
        "score 0.0 — this is a confirmed false positive regardless of how plausible "
        "the reasoning sounds. "
        "If the ACTUAL OUTPUT verdict is BLOCK and the EXPECTED OUTPUT is BLOCK, "
        "evaluate the quality of the reasoning: score 1.0 if the reasoning identifies "
        "a concrete, exploitable attack path (e.g. user-controlled data reaches a "
        "dangerous sink without sanitisation); score 0.5 if the reasoning is plausible "
        "but vague or missing a clear exploit path; score 0.2 if the reasoning is "
        "misleading or references the wrong vulnerability class."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.6,
)

# ---------------------------------------------------------------------------
# Labeled-threat-catch metric  (missed-vuln / weak-response guard)
# ---------------------------------------------------------------------------
# Uses human ground truth (EXPECTED_OUTPUT). When the label is BLOCK, did the
# model respond appropriately? This is not formal sensitivity/recall; it adds
# a graded judge on top of label agreement.
#
# Scoring guidance:
#   • EXPECTED_OUTPUT is PASS → score 0.8 (neutral; threat label not applicable).
#   • EXPECTED_OUTPUT is BLOCK and ACTUAL_OUTPUT contains BLOCK → 1.0 or 0.6
#     by quality of reasoning vs. the real issue.
#   • EXPECTED_OUTPUT is BLOCK and ACTUAL_OUTPUT contains PASS → 0.0 (missed threat).
#
# LLMTestCaseParams mapping:
#   INPUT           → the full prompt (includes raw_source and changed_lines)
#   ACTUAL_OUTPUT   → SecurityVerdict.reasoning + verdict label
#   EXPECTED_OUTPUT → human verdict ("BLOCK" or "PASS")

LabeledThreatCatchMetric = GEval(
    name="LabeledThreatCatch",
    model=JUDGE_MODEL,
    criteria=(
        "You are evaluating whether a security model correctly responded to a "
        "human-verified label. The EXPECTED OUTPUT is the ground truth verdict "
        "('BLOCK' means experts judged a real vulnerability; 'PASS' means safe). "
        "The ACTUAL OUTPUT is what the model produced. "
        "If the EXPECTED OUTPUT is 'PASS', assign 0.8 — this check does not apply. "
        "If the EXPECTED OUTPUT is 'BLOCK' and the ACTUAL OUTPUT contains 'BLOCK', "
        "score 1.0 if the reasoning correctly identifies the vulnerability class "
        "and relevant code path, or 0.6 if it BLOCKs for a vague or partially "
        "correct reason. "
        "If the EXPECTED OUTPUT is 'BLOCK' and the ACTUAL OUTPUT contains 'PASS', "
        "score 0.0 — the model failed to flag a confirmed vulnerability."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.6,
)
