from __future__ import annotations

import json
import random
from pathlib import Path

from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)

from src.config import Condition
from src.datasets.questions import get_question_sequence, load_questions

PROMPTS_DIR = Path(__file__).parent.parent.parent / "data" / "prompts"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

_hardcoded_cache: dict[str, dict[str, str]] = {}
_set_cache: dict[str, list[str]] = {}


def _build_metadata(
    condition: Condition,
    n_turns: int,
    instruction_template: str,
    trial_index: int,
) -> dict:
    """Build standard metadata dict for a sample."""
    meta = {
        "condition": condition.name,
        "condition_type": condition.condition_type,
        "pattern": condition.pattern,
        "target": condition.target,
        "target_description": condition.target_description,
        "pattern_description": condition.pattern_description,
        "n_turns": n_turns,
        "instruction_template": instruction_template,
        "trial_index": trial_index,
    }
    if condition.pattern_set:
        meta["pattern_set"] = condition.pattern_set
    if condition.target_set:
        meta["target_set"] = condition.target_set
    return meta


def _load_prompt(filename: str) -> str:
    """Load a prompt template from file."""
    with open(PROMPTS_DIR / filename) as f:
        return f.read().strip()


def _load_set(filename: str) -> list[str]:
    """Load a set membership list from JSON."""
    if filename not in _set_cache:
        with open(DATA_DIR / "sets" / filename) as f:
            data = json.load(f)
        _set_cache[filename] = data["members"]
    return _set_cache[filename]


def _load_hardcoded_responses(data_key: str) -> dict[str, str]:
    """Load pre-generated hardcoded responses from JSON."""
    if data_key not in _hardcoded_cache:
        path = DATA_DIR / "hardcoded_responses" / f"{data_key}.json"
        with open(path) as f:
            data = json.load(f)
        _hardcoded_cache[data_key] = data["responses"]
    return _hardcoded_cache[data_key]


def _get_hardcoded_response(condition: Condition, question: str) -> str:
    """Get the hardcoded P-response for a given question and condition."""
    if condition.condition_type == "static":
        return condition.pattern

    if condition.condition_type == "token_pattern":
        members = _load_set(condition.pattern_set)
        rng = random.Random(hash(question))
        return rng.choice(members)

    if condition.pattern_data_key is not None:
        path = DATA_DIR / "hardcoded_responses" / f"{condition.pattern_data_key}.json"
        if not path.exists():
            raise NotImplementedError(
                f"Condition '{condition.name}' requires pre-generated hardcoded "
                f"responses at {path}. Run scripts/generate_hardcoded_responses.py first."
            )
        responses = _load_hardcoded_responses(condition.pattern_data_key)
        return responses[question]

    raise NotImplementedError(
        f"Condition '{condition.name}' requires pre-generated hardcoded responses "
        f"that have not been created yet. Run scripts/generate_hardcoded_responses.py first."
    )


def _build_conversation(
    condition: Condition,
    n_turns: int,
    questions: list[str],
    instruction_template: str = "instruction_hint",
) -> list[ChatMessage]:
    """Build the conversation history for the sample.

    Args:
        condition: The condition with pattern (P) and target (T).
        n_turns: Number of hardcoded assistant turns.
        questions: List of questions to use.
        instruction_template: Name of instruction template file (without .txt).

    Returns:
        List of ChatMessage objects representing the conversation.
    """
    messages: list[ChatMessage] = []

    system_content = _load_prompt(condition.system_template or "system.txt")
    messages.append(ChatMessageSystem(content=system_content))

    instruction_content = _load_prompt(f"{instruction_template}.txt")
    instruction_content = instruction_content.format(
        target_description=condition.target_description,
        pattern_description=condition.pattern_description,
    )
    # First question is concatenated with the instruction in a single user message
    first_question = questions[0] if questions else "What do you think?"
    messages.append(
        ChatMessageUser(content=f"{instruction_content}\n\n{first_question}")
    )
    messages.append(
        ChatMessageAssistant(content=_get_hardcoded_response(condition, first_question))
    )

    # Remaining hardcoded turns (questions[1] through questions[n_turns-1])
    for question in questions[1:n_turns]:
        messages.append(ChatMessageUser(content=question))
        messages.append(
            ChatMessageAssistant(content=_get_hardcoded_response(condition, question))
        )

    # Final question for free generation
    if len(questions) > n_turns:
        final_question = questions[n_turns]
    else:
        final_question = "What do you think?"
    messages.append(ChatMessageUser(content=final_question))

    return messages


def build_behavioral_sample(
    condition: Condition,
    n_turns: int,
    instruction_template: str = "instruction_hint",
    trial_index: int = 0,
    sample_id: str | None = None,
) -> Sample:
    """Build a Sample for Protocol 1 (Behavioral Baseline).

    Args:
        condition: The condition with pattern (P) and target (T).
        n_turns: Number of hardcoded assistant turns before free generation.
        instruction_template: Name of instruction template file (without .txt).
        trial_index: Trial index for deterministic question selection.
        sample_id: Optional sample ID.

    Returns:
        A Sample ready for evaluation.
    """
    bank = load_questions(DATA_DIR / condition.question_bank) if condition.question_bank else None
    questions = get_question_sequence(trial_index, n_turns + 1, question_bank=bank)

    messages = _build_conversation(
        condition=condition,
        n_turns=n_turns,
        questions=questions,
        instruction_template=instruction_template,
    )

    return Sample(
        id=sample_id
        or f"{condition.name}_n{n_turns}_{instruction_template}_trial{trial_index}",
        input=messages,
        target=condition.target,
        metadata=_build_metadata(condition, n_turns, instruction_template, trial_index),
    )


def build_prediction_sample(
    condition: Condition,
    n_turns: int,
    instruction_template: str = "instruction_hint",
    trial_index: int = 0,
    sample_id: str | None = None,
) -> Sample:
    """Build a Sample for Protocol 2 (Self-Prediction).

    Builds the full conversation with N hardcoded turns but stops before the
    final question. The solver inserts the prediction request and final question.

    Args:
        condition: The condition with pattern (P) and target (T).
        n_turns: Number of hardcoded assistant turns.
        instruction_template: Name of instruction template file (without .txt).
        trial_index: Trial index for deterministic question selection.
        sample_id: Optional sample ID.

    Returns:
        A Sample ready for prediction evaluation.
    """
    bank = load_questions(DATA_DIR / condition.question_bank) if condition.question_bank else None
    questions = get_question_sequence(trial_index, n_turns + 2, question_bank=bank)

    messages: list[ChatMessage] = []

    system_content = _load_prompt(condition.system_template or "system.txt")
    messages.append(ChatMessageSystem(content=system_content))

    instruction_content = _load_prompt(f"{instruction_template}.txt")
    instruction_content = instruction_content.format(
        target_description=condition.target_description,
        pattern_description=condition.pattern_description,
    )
    first_question = questions[0] if questions else "What do you think?"
    messages.append(
        ChatMessageUser(content=f"{instruction_content}\n\n{first_question}")
    )
    messages.append(
        ChatMessageAssistant(content=_get_hardcoded_response(condition, first_question))
    )

    for question in questions[1:n_turns]:
        messages.append(ChatMessageUser(content=question))
        messages.append(
            ChatMessageAssistant(content=_get_hardcoded_response(condition, question))
        )

    meta = _build_metadata(condition, n_turns, instruction_template, trial_index)
    meta["final_question"] = (
        questions[n_turns] if len(questions) > n_turns else "What do you think?"
    )

    return Sample(
        id=sample_id
        or f"{condition.name}_n{n_turns}_{instruction_template}_trial{trial_index}_pred",
        input=messages,
        target=condition.target,
        metadata=meta,
    )
