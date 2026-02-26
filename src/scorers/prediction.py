from __future__ import annotations

import sys
from pathlib import Path

# When loaded via `inspect score --scorer src/scorers/prediction.py`, inspect
# loads the file outside the normal package context, so `src.*` imports fail.
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

from src.scorers.classify import classify_actual, classify_prediction


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
    condition type, so instruction_following is consistent with Protocol 1.

    Returns a dict with three metrics:
    - prediction_accuracy: Did prediction match actual behavior?
    - instruction_following: Did actual output follow instruction (T)?
    - prediction_instruction: Did prediction say it would follow instruction (T)?
    """

    async def score(state: TaskState, target: Target) -> Score:
        output = state.output.completion.strip()
        prediction = state.store.get("prediction", "").strip()
        metadata = state.metadata

        # classify_actual uses the authoritative scorer for each condition type
        # (LLM judge for persona/preference, deterministic otherwise)
        actual_choice = await classify_actual(output, metadata)

        # classify_prediction uses an LLM judge — the prediction prompt names both
        # options explicitly so keyword matching on presence is unreliable
        prediction_choice = await classify_prediction(prediction, metadata)

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
            },
        )

    return score
