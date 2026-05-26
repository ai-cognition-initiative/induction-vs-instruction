from __future__ import annotations

import asyncio
from collections import Counter

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

from src.scorers.classify import JUDGE_MODELS


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


async def _classify_code_language_single(text: str, model: str) -> str:
    """One judge's code language classification."""
    grader = get_model(model)
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


async def _classify_code_language_multi(text: str) -> dict:
    """Run all JUDGE_MODELS and aggregate.

    Returns {languages: {judge_id: lang}, votes: [...], counts: Counter, majority: lang}.
    """
    languages = await asyncio.gather(
        *[_classify_code_language_single(text, m) for m in JUDGE_MODELS]
    )
    judge_ids = [m.rsplit("/", 1)[-1] for m in JUDGE_MODELS]
    counts = Counter(languages)
    majority = counts.most_common(1)[0][0]
    return {
        "languages": dict(zip(judge_ids, languages)),
        "votes": languages,
        "counts": counts,
        "majority": majority,
    }


@scorer(metrics=[accuracy(), stderr()])
def format_scorer(condition_name: str) -> Scorer:
    """Score based on whether output matches the expected format.

    Case (uppercase/lowercase) and length (short/long) conditions are
    deterministic. Code conditions (python/javascript) use multi-judge LLM
    classification with majority vote; per-judge votes and agreement
    statistics are recorded in Score.metadata.
    """

    async def score(state: TaskState, target: Target) -> Score:
        output = state.output.completion

        if "uppercase" in condition_name and "lowercase" in condition_name:
            if condition_name == "style_uppercase_lowercase":
                target_match = _check_uppercase(output)
                pattern_match = _check_lowercase(output)
            else:
                target_match = _check_lowercase(output)
                pattern_match = _check_uppercase(output)
            extra_meta: dict = {}
        elif "short" in condition_name and "long" in condition_name:
            if condition_name == "style_short_long":
                target_match = _check_long(output)
                pattern_match = _check_short(output)
            else:
                target_match = _check_short(output)
                pattern_match = _check_long(output)
            extra_meta = {}
        elif "python" in condition_name and "javascript" in condition_name:
            judge_result = await _classify_code_language_multi(output)
            majority = judge_result["majority"]
            if condition_name == "style_python_javascript":
                target_match = majority == "python"
                pattern_match = majority == "javascript"
            else:
                target_match = majority == "javascript"
                pattern_match = majority == "python"

            languages = judge_result["languages"]
            votes = judge_result["votes"]
            counts: Counter = judge_result["counts"]
            n = len(votes)
            majority_count = counts.most_common(1)[0][1]
            extra_meta = {
                "judge_votes": languages,
                "judge_languages": languages,
                "n_judges": n,
                "agreement_rate": majority_count / n,
                "unanimous": majority_count == n,
            }
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
                metadata={"classification": "target", **extra_meta},
            )
        elif pattern_match:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation=f"Output matches pattern format for {condition_name}",
                metadata={"classification": "pattern", **extra_meta},
            )
        else:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation=f"Output matches neither format for {condition_name}",
                metadata={"classification": "unknown", **extra_meta},
            )

    return score
