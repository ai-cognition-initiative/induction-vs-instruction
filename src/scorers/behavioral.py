from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Importing the package registers every @scorer-decorated factory so
# inspect-ai's registry can reconstruct the original scorers when reading the
# log during `inspect score`.
import src.scorers  # noqa: F401, E402

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

from src.scorers.classify import classify_actual_multi


@scorer(
    metrics={
        "instruction_following": [accuracy(), stderr()],
        "unknown": [accuracy(), stderr()],
    }
)
def behavioral_scorer() -> Scorer:
    """Score Protocol 1 (Behavioral) output.

    Uses the same classification logic as the task's built-in scorers,
    dispatching based on condition type in metadata. For LLM-judge condition
    types this runs every model in JUDGE_MODELS and majority-votes; per-judge
    votes and agreement statistics are preserved in Score.metadata.
    """

    async def score(state: TaskState, target: Target) -> Score:
        output = state.output.completion.strip()
        metadata = state.metadata

        result = await classify_actual_multi(output, metadata)
        choice = result["classification"]
        followed_instruction = choice == "target"

        return Score(
            value={
                "instruction_following": CORRECT if followed_instruction else INCORRECT,
                "unknown": CORRECT if choice == "unknown" else INCORRECT,
            },
            answer=output,
            explanation=f"Classified as: {choice}",
            metadata={
                "actual": output,
                "choice": choice,
                "condition": metadata.get("condition", ""),
                "condition_type": metadata.get("condition_type", ""),
                "judges": {
                    k: result[k]
                    for k in (
                        "judge_votes",
                        "n_judges",
                        "n_target",
                        "n_pattern",
                        "n_unknown",
                        "agreement_rate",
                        "unanimous",
                    )
                    if k in result
                },
            },
        )

    return score
