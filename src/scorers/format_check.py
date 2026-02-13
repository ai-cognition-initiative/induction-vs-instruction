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


def _check_python(text: str) -> float:
    """Return positive score for Python, negative for JavaScript."""
    python_keywords = ["def ", "import ", "print(", "self.", "elif ", "class "]
    js_keywords = ["function ", "const ", "=> ", "let ", "var ", "console.log"]
    py_score = sum(1 for kw in python_keywords if kw in text)
    js_score = sum(1 for kw in js_keywords if kw in text)
    return py_score - js_score


def _check_javascript(text: str) -> float:
    """Return positive score for JavaScript, negative for Python."""
    return -_check_python(text)


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
            # style_uppercase_lowercase or style_lowercase_uppercase
            if condition_name == "style_uppercase_lowercase":
                # Target is lowercase, pattern is uppercase
                target_match = _check_lowercase(output)
                pattern_match = _check_uppercase(output)
            else:
                # Target is uppercase, pattern is lowercase
                target_match = _check_uppercase(output)
                pattern_match = _check_lowercase(output)
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
            if condition_name == "style_python_javascript":
                # Target is JavaScript, pattern is Python
                code_score = _check_javascript(output)
                target_match = code_score > 0
                pattern_match = code_score < 0
            else:
                # Target is Python, pattern is JavaScript
                code_score = _check_python(output)
                target_match = code_score > 0
                pattern_match = code_score < 0
        else:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation=f"Unknown format condition: {condition_name}",
            )

        if target_match:
            return Score(
                value=CORRECT,
                answer=output,
                explanation=f"Output matches target format for {condition_name}",
            )
        elif pattern_match:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation=f"Output matches pattern format for {condition_name}",
            )
        else:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation=f"Output matches neither format for {condition_name}",
            )

    return score
