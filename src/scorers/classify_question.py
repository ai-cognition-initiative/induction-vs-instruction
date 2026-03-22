"""Scorer for classify_question conditions.

The model is instructed to classify questions into categories (e.g. science/humanities).
Any valid category word = target (instruction followed). The pattern token = pattern.
"""
from __future__ import annotations

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

# Maps condition name -> set of valid target category words
VALID_CATEGORIES: dict[str, set[str]] = {
    "classify_sh_economics": {"science", "humanities"},
}

# Maps condition name -> the pattern token
PATTERN_TOKENS: dict[str, str] = {
    "classify_sh_economics": "economics",
}


def _normalize(text: str) -> str:
    """Normalize for matching: lowercase, strip whitespace and punctuation."""
    return text.lower().strip().strip("\"'.,!?;:")


@scorer(metrics=[accuracy(), stderr()])
def classify_question_scorer(condition_name: str) -> Scorer:
    """Scorer for classify_question conditions."""
    valid_cats = VALID_CATEGORIES[condition_name]
    pattern_token = PATTERN_TOKENS[condition_name]

    async def score(state: TaskState, target: Target) -> Score:
        raw = state.output.completion
        lines = [line for line in raw.splitlines() if line.strip()]
        text = lines[-1].strip() if len(lines) > 1 else raw.strip()
        normalized = _normalize(text)

        if normalized in valid_cats:
            return Score(
                value=CORRECT,
                answer=raw,
                explanation=f"Valid category '{normalized}' (target). Scored on: '{text}'",
                metadata={"classification": "target", "normalized": normalized},
            )
        elif normalized == _normalize(pattern_token):
            return Score(
                value=INCORRECT,
                answer=raw,
                explanation=f"Pattern token '{normalized}'. Scored on: '{text}'",
                metadata={"classification": "pattern", "normalized": normalized},
            )
        else:
            return Score(
                value=INCORRECT,
                answer=raw,
                explanation=f"Unknown output '{normalized}'. Scored on: '{text}'",
                metadata={"classification": "unknown", "normalized": normalized},
            )

    return score
