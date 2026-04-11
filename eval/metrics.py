"""
Eval – Custom DeepEval Metrics
Three GEval metrics that use a capable judge model to score SecurityVerdict outputs.

HallucinationMetric       – is the reasoning grounded in the real code?
BlockJustificationMetric  – for a correct BLOCK (TP), was the exploit reasoning specific?
VulnClassificationMetric  – for a correct BLOCK (TP), was the vulnerability class right?

IMPORTANT — Judge routing strategy:
    Only True Positive cases (expected=BLOCK, actual=BLOCK) are sent to the judge.
    All other outcomes are scored deterministically in eval_runner.py before the
    judge is called, so the LLM never sees hardcoded conditional branching:

        TN (PASS / PASS)   → neutral, skip judge
        FP (PASS / BLOCK)  → BlockJustification score = 0.0  (deterministic)
        FN (BLOCK / PASS)  → VulnClassification score = 0.0  (deterministic)
        TP (BLOCK / BLOCK) → judge scores both metrics on reasoning quality

This keeps criteria prompts simple (no if/else), reduces judge calls by ~50%,
and eliminates the known LLM weakness of following strict conditional logic.
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
# Runs on ALL cases — reasoning quality is independent of verdict correctness.
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
# Block justification metric  — TP cases only
# ---------------------------------------------------------------------------
# Evaluates whether the exploit reasoning for a confirmed BLOCK is specific
# and correct. FP cases (model BLOCKed a safe function) receive score 0.0
# deterministically in eval_runner.py and are never sent here.
#
# LLMTestCaseParams mapping:
#   INPUT          → the full prompt (includes raw_source and changed_lines)
#   ACTUAL_OUTPUT  → SecurityVerdict.reasoning + verdict label

BlockJustificationMetric = GEval(
    name="BlockJustification",
    model=JUDGE_MODEL,
    criteria=(
        "You are reviewing the reasoning for a BLOCK verdict that was confirmed "
        "correct by a human expert. Evaluate the quality of the exploit description: "
        "Score 1.0 if the reasoning identifies a concrete, exploitable attack path "
        "(e.g. user-controlled data reaches a dangerous sink without sanitisation, "
        "with the specific variable and sink named). "
        "Score 0.5 if the reasoning correctly identifies the vulnerability type but "
        "is vague about the specific code path or attack vector. "
        "Score 0.2 if the reasoning references the wrong vulnerability class or "
        "describes an attack path that does not match the actual code."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.6,
)

# ---------------------------------------------------------------------------
# Vulnerability classification metric  — TP cases only
# ---------------------------------------------------------------------------
# Evaluates whether the model correctly identified the vulnerability CLASS for
# a confirmed BLOCK. FN cases (model missed a real vulnerability) receive
# score 0.0 deterministically in eval_runner.py and are never sent here.
#
# LLMTestCaseParams mapping:
#   INPUT           → the full prompt (includes raw_source and changed_lines)
#   ACTUAL_OUTPUT   → SecurityVerdict.reasoning + verdict label + vulnerability type

VulnClassificationMetric = GEval(
    name="VulnClassification",
    model=JUDGE_MODEL,
    criteria=(
        "You are reviewing a confirmed BLOCK verdict. Evaluate whether the model "
        "correctly identified the vulnerability class and the relevant code path. "
        "Score 1.0 if the reasoning names the correct attack type (e.g. SQL Injection, "
        "Path Traversal, Command Injection) and correctly traces the data flow from "
        "the user-controlled source to the dangerous sink. "
        "Score 0.6 if the vulnerability class is correct but the data flow description "
        "is incomplete or partially wrong. "
        "Score 0.2 if the vulnerability class is wrong or the described attack path "
        "does not match the actual code (correct verdict, wrong reason)."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.6,
)
