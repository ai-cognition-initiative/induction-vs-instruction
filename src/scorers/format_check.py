from __future__ import annotations

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
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

from src.scorers.classify import GRADER_MODEL


def _check_uppercase(text: str) -> bool:
    stripped = text.strip()
    return stripped == stripped.upper() and len(stripped) > 0


def _check_lowercase(text: str) -> bool:
    stripped = text.strip()
    return stripped == stripped.lower() and len(stripped) > 0


def _check_short(text: str) -> bool:
    return len(text.split()) < 25


def _check_long(text: str) -> bool:
    return len(text.split()) > 40


CODE_LANGUAGE_RUBRIC = """You are evaluating which programming language a code snippet is written in.

Here is the code to evaluate:
---
{output}
---

Which programming language is this code written in?
Answer with exactly one of:
- PYTHON: if the code is Python (e.g., def, import, print(), self, elif, indentation-based)
- JAVASCRIPT: if the code is JavaScript (e.g., function, const, =>, let, var, console.log, braces)
- NEITHER: if the code is not clearly Python or JavaScript, or is ambiguous

Respond with just one word: PYTHON, JAVASCRIPT, or NEITHER."""


async def _classify_code_language(text: str) -> str:
    """Classify code language via LLM judge. Returns 'python', 'javascript', or 'unknown'."""
    grader = get_model(GRADER_MODEL)
    result = await grader.generate(
        [
            ChatMessageSystem(
                content="You are a precise code evaluator. Respond with exactly one word."
            ),
            ChatMessageUser(content=CODE_LANGUAGE_RUBRIC.format(output=text)),
        ]
    )
    judgment = result.completion.strip().upper()
    if "PYTHON" in judgment:
        return "python"
    if "JAVASCRIPT" in judgment:
        return "javascript"
    return "unknown"


@scorer(metrics=[accuracy(), stderr()])
def format_scorer(condition_name: str) -> Scorer:
    """Score based on whether output matches the expected format.

    Dispatches on the condition name to determine which format check to apply.
    The target format is what the instruction asks for; the pattern format is
    what the hardcoded turns demonstrated.
    """

    async def score(state: TaskState, target: Target) -> Score:
        output = state.output.completion

        if "uppercase" in condition_name and "lowercase" in condition_name:
            # Naming convention: {instruction}_{pattern}, so instruction is first word
            if condition_name == "style_uppercase_lowercase":
                # instruction=uppercase → target=UPPERCASE, pattern=lowercase
                target_match = _check_uppercase(output)
                pattern_match = _check_lowercase(output)
            else:
                # style_lowercase_uppercase: instruction=lowercase → target=lowercase, pattern=UPPERCASE
                target_match = _check_lowercase(output)
                pattern_match = _check_uppercase(output)
        elif "short" in condition_name and "long" in condition_name:
            if condition_name == "style_short_long":
                # Target is long, pattern is short
                target_match = _check_long(output)
                pattern_match = _check_short(output)
            else:
                # Target is short, pattern is long
                target_match = _check_short(output)
                pattern_match = _check_long(output)
        elif "python" in condition_name and "javascript" in condition_name:
            language = await _classify_code_language(output)
            if condition_name == "style_python_javascript":
                target_match = language == "python"
                pattern_match = language == "javascript"
            else:
                target_match = language == "javascript"
                pattern_match = language == "python"
        else:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation=f"Unknown format condition: {condition_name}",
                metadata={"classification": "unknown"},
            )

        if target_match:
            return Score(
                value=CORRECT,
                answer=output,
                explanation=f"Output matches target format for {condition_name}",
                metadata={"classification": "target"},
            )
        elif pattern_match:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation=f"Output matches pattern format for {condition_name}",
                metadata={"classification": "pattern"},
            )
        else:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation=f"Output matches neither format for {condition_name}",
                metadata={"classification": "unknown"},
            )

    return score
