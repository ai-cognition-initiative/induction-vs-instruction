from __future__ import annotations

from pathlib import Path

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Load a prompt template from file."""
    with open(PROMPTS_DIR / filename) as f:
        return f.read().strip()


@solver
def behavioral_solver() -> Solver:
    """Solver for Protocol 1 (Behavioral Baseline).

    Simply generates a response - the conversation is already built in the sample.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await generate(state)

    return solve


@solver
def prediction_solver() -> Solver:
    """Solver for Protocol 2 (Self-Prediction).

    1. Insert prediction request using composable template
    2. Generate prediction
    3. Store prediction in state.store
    4. Append final question
    5. Generate actual response
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        target_description = state.metadata["target_description"]
        pattern_description = state.metadata["pattern_description"]
        final_question = state.metadata.get("final_question", "What do you think?")

        prediction_template = _load_prompt("prediction_request.txt")
        prediction_request = prediction_template.format(
            target_description=target_description,
            pattern_description=pattern_description,
        )

        state.messages.append(ChatMessageUser(content=prediction_request))

        prediction_state = await generate(state)

        prediction = prediction_state.output.completion
        prediction_state.store.set("prediction", prediction)

        prediction_state.messages.append(ChatMessageUser(content=final_question))

        final_state = await generate(prediction_state)

        return final_state

    return solve
