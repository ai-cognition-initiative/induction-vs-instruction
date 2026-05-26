from __future__ import annotations

import sys
from pathlib import Path

# When loaded via `inspect score --scorer src/scorers/prediction.py`, inspect
# loads the file outside the normal package context, so `src.*` imports fail.
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

from src.scorers.classify import classify_actual_multi, classify_prediction_multi


@scorer(
    metrics={
        "prediction_accuracy": [accuracy(), stderr()],
        "instruction_following": [accuracy(), stderr()],
        "prediction_instruction": [accuracy(), stderr()],
        "actual_unknown": [accuracy(), stderr()],
        "prediction_unknown": [accuracy(), stderr()],
    }
)
def prediction_scorer() -> Scorer:
    """Score Protocol 2 (Self-Prediction) with multiple metrics.

    Uses the same classification logic as the behavioral scorers for each
    condition type. For LLM-judge condition types this runs every model in
    JUDGE_MODELS and majority-votes; per-judge votes and agreement statistics
    are preserved in Score.metadata.
    """

    async def score(state: TaskState, target: Target) -> Score:
        output = state.output.completion.strip()
        prediction = state.store.get("prediction", "").strip()
        metadata = state.metadata

        actual_result = await classify_actual_multi(output, metadata)
        prediction_result = await classify_prediction_multi(prediction, metadata)

        actual_choice = actual_result["classification"]
        prediction_choice = prediction_result["classification"]

        actual_followed_instruction = actual_choice == "target"
        prediction_followed_instruction = prediction_choice == "target"
        prediction_correct = (
            actual_choice == prediction_choice and actual_choice != "unknown"
        )

        return Score(
            value={
                "prediction_accuracy": CORRECT if prediction_correct else INCORRECT,
                "instruction_following": CORRECT if actual_followed_instruction else INCORRECT,
                "prediction_instruction": CORRECT if prediction_followed_instruction else INCORRECT,
                "actual_unknown": CORRECT if actual_choice == "unknown" else INCORRECT,
                "prediction_unknown": CORRECT if prediction_choice == "unknown" else INCORRECT,
            },
            answer=output,
            explanation=f"Prediction choice: {prediction_choice}, Actual choice: {actual_choice}",
            metadata={
                "prediction": prediction,
                "actual": output,
                "prediction_choice": prediction_choice,
                "actual_choice": actual_choice,
                "condition": metadata.get("condition", ""),
                "condition_type": metadata.get("condition_type", ""),
                "actual_judges": {
                    k: actual_result[k]
                    for k in (
                        "judge_votes",
                        "n_judges",
                        "n_target",
                        "n_pattern",
                        "n_unknown",
                        "agreement_rate",
                        "unanimous",
                    )
                    if k in actual_result
                },
                "prediction_judges": {
                    k: prediction_result[k]
                    for k in (
                        "judge_votes",
                        "n_judges",
                        "n_target",
                        "n_pattern",
                        "n_unknown",
                        "agreement_rate",
                        "unanimous",
                    )
                    if k in prediction_result
                },
            },
        )

    return score
