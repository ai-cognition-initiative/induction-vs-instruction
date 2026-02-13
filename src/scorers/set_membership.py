from __future__ import annotations

import json
from pathlib import Path

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

DATA_DIR = Path(__file__).parent.parent.parent / "data"

_set_cache: dict[str, list[str]] = {}


def _load_set(filename: str) -> list[str]:
    if filename not in _set_cache:
        with open(DATA_DIR / "sets" / filename) as f:
            data = json.load(f)
        _set_cache[filename] = data["members"]
    return _set_cache[filename]


def _normalize(text: str) -> str:
    return text.strip().lower()


def _fuzzy_set_match(text: str, members: list[str]) -> str | None:
    """Check if text matches any member of the set. Returns matched member or None."""
    normalized = _normalize(text)
    for member in members:
        if _normalize(member) == normalized:
            return member
        if _normalize(member) in normalized or normalized in _normalize(member):
            return member
    return None


@scorer(metrics=[accuracy(), stderr()])
def set_membership_scorer(
    pattern_set: str,
    target_set: str,
) -> Scorer:
    """Score based on whether output is a member of target set (T) or pattern set (P)."""

    async def score(state: TaskState, target: Target) -> Score:
        output = state.output.completion.strip()
        t_members = _load_set(target_set)
        p_members = _load_set(pattern_set)

        t_match = _fuzzy_set_match(output, t_members)
        p_match = _fuzzy_set_match(output, p_members)

        if t_match and not p_match:
            return Score(
                value=CORRECT,
                answer=output,
                explanation=f"Output matched target set member '{t_match}'",
                metadata={"matched_member": t_match, "matched_set": "target"},
            )
        elif p_match and not t_match:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation=f"Output matched pattern set member '{p_match}'",
                metadata={"matched_member": p_match, "matched_set": "pattern"},
            )
        elif t_match and p_match:
            return Score(
                value=CORRECT,
                answer=output,
                explanation=f"Output matched both sets (target: '{t_match}', pattern: '{p_match}')",
                metadata={"matched_member": t_match, "matched_set": "both"},
            )
        else:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation="Output matched neither set",
                metadata={"matched_member": None, "matched_set": "neither"},
            )

    return score
