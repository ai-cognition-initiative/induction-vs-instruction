from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

from src.scorers.classify import classify_actual


@scorer(metrics=[accuracy(), stderr()])
def behavioral_scorer() -> Scorer:
    """Score Protocol 1 (Behavioral) output.

    Uses the same classification logic as the task's built-in scorers,
    dispatching based on condition type in metadata.
    """

    async def score(state: TaskState, target: Target) -> Score:
        output = state.output.completion.strip()
        metadata = state.metadata

        choice = await classify_actual(output, metadata)
        followed_instruction = choice == "target"

        return Score(
            value=CORRECT if followed_instruction else INCORRECT,
            answer=output,
            explanation=f"Classified as: {choice}",
            metadata={
                "actual": output,
                "choice": choice,
                "condition": metadata.get("condition", ""),
                "condition_type": metadata.get("condition_type", ""),
            },
        )

    return score
