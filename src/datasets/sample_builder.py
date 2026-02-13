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
from src.datasets.questions import sample_questions

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

_hardcoded_cache: dict[str, dict[str, str]] = {}
_set_cache: dict[str, list[str]] = {}


def _build_metadata(
    condition: Condition,
    n_turns: int,
    hint: bool,
    question_seed: int | None,
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
        "hint": hint,
        "question_seed": question_seed,
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
    hint: bool = True,
) -> list[ChatMessage]:
    """Build the conversation history for the sample.

    Args:
        condition: The condition with pattern (P) and target (T).
        n_turns: Number of hardcoded assistant turns.
        questions: List of questions to use.
        hint: Whether to include the hardcoding hint in instruction.

    Returns:
        List of ChatMessage objects representing the conversation.
    """
    messages: list[ChatMessage] = []

    system_content = _load_prompt(condition.system_template or "system.txt")
    messages.append(ChatMessageSystem(content=system_content))

    instruction_template = _load_prompt(
        "instruction_hint.txt" if hint else "instruction_no_hint.txt"
    )
    instruction_content = instruction_template.format(
        target_description=condition.target_description,
        pattern_description=condition.pattern_description,
    )
    # First question is concatenated with the instruction in a single user message
    first_question = questions[0] if questions else "What do you think?"
    messages.append(
        ChatMessageUser(content=f"{instruction_content}\n\n{first_question}")
    )
    messages.append(
        ChatMessageAssistant(
            content=_get_hardcoded_response(condition, first_question)
        )
    )

    # Remaining hardcoded turns (questions[1] through questions[n_turns-1])
    for question in questions[1:n_turns]:
        messages.append(ChatMessageUser(content=question))
        messages.append(
            ChatMessageAssistant(
                content=_get_hardcoded_response(condition, question)
            )
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
    hint: bool = True,
    question_seed: int | None = None,
    sample_id: str | None = None,
) -> Sample:
    """Build a Sample for Protocol 1 (Behavioral Baseline).

    Args:
        condition: The condition with pattern (P) and target (T).
        n_turns: Number of hardcoded assistant turns before free generation.
        hint: Whether to include the hardcoding hint.
        question_seed: Random seed for question selection.
        sample_id: Optional sample ID.

    Returns:
        A Sample ready for evaluation.
    """
    questions = sample_questions(n=n_turns + 1, seed=question_seed)

    messages = _build_conversation(
        condition=condition,
        n_turns=n_turns,
        questions=questions,
        hint=hint,
    )

    return Sample(
        id=sample_id or f"{condition.name}_n{n_turns}_hint{hint}",
        input=messages,
        target=condition.target,
        metadata=_build_metadata(condition, n_turns, hint, question_seed),
    )


def build_prediction_sample(
    condition: Condition,
    n_turns: int,
    hint: bool = True,
    question_seed: int | None = None,
    sample_id: str | None = None,
) -> Sample:
    """Build a Sample for Protocol 2 (Self-Prediction).

    Builds the full conversation with N hardcoded turns but stops before the
    final question. The solver inserts the prediction request and final question.

    Args:
        condition: The condition with pattern (P) and target (T).
        n_turns: Number of hardcoded assistant turns.
        hint: Whether to include the hardcoding hint.
        question_seed: Random seed for question selection.
        sample_id: Optional sample ID.

    Returns:
        A Sample ready for prediction evaluation.
    """
    questions = sample_questions(n=n_turns + 2, seed=question_seed)

    # Build conversation with n_turns hardcoded turns (no final question)
    messages: list[ChatMessage] = []

    system_content = _load_prompt(condition.system_template or "system.txt")
    messages.append(ChatMessageSystem(content=system_content))

    instruction_template = _load_prompt(
        "instruction_hint.txt" if hint else "instruction_no_hint.txt"
    )
    instruction_content = instruction_template.format(
        target_description=condition.target_description,
        pattern_description=condition.pattern_description,
    )
    # First question concatenated with instruction
    first_question = questions[0] if questions else "What do you think?"
    messages.append(
        ChatMessageUser(content=f"{instruction_content}\n\n{first_question}")
    )
    messages.append(
        ChatMessageAssistant(
            content=_get_hardcoded_response(condition, first_question)
        )
    )

    for question in questions[1:n_turns]:
        messages.append(ChatMessageUser(content=question))
        messages.append(
            ChatMessageAssistant(
                content=_get_hardcoded_response(condition, question)
            )
        )

    meta = _build_metadata(condition, n_turns, hint, question_seed)
    meta["final_question"] = (
        questions[n_turns] if len(questions) > n_turns else "What do you think?"
    )

    return Sample(
        id=sample_id or f"{condition.name}_n{n_turns}_hint{hint}_pred",
        input=messages,
        target=condition.target,
        metadata=meta,
    )
