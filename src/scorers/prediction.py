from __future__ import annotations

import json
from pathlib import Path

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
from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0

DATA_DIR = Path(__file__).parent.parent.parent / "data"

_set_cache: dict[str, list[str]] = {}

LANGUAGE_CODE_MAP = {
    "French": "fr",
    "Russian": "ru",
    "English": "en",
}


def _load_set(filename: str) -> list[str]:
    if filename not in _set_cache:
        with open(DATA_DIR / "sets" / filename) as f:
            data = json.load(f)
        _set_cache[filename] = data["members"]
    return _set_cache[filename]


def _classify_static(text: str, pattern: str, target: str) -> str:
    """Classify text as 'target', 'pattern', or 'unknown' for static conditions."""
    text_lower = text.lower().strip()
    pattern_lower = pattern.lower().strip()
    target_lower = target.lower().strip()

    has_target = target_lower in text_lower
    has_pattern = pattern_lower in text_lower

    if has_target and not has_pattern:
        return "target"
    if has_pattern and not has_target:
        return "pattern"
    if has_target and has_pattern:
        return "target"
    return "unknown"


def _classify_set_membership(
    text: str, pattern_set: str, target_set: str
) -> str:
    """Classify text as 'target', 'pattern', or 'unknown' for set membership conditions."""
    normalized = text.strip().lower()
    t_members = _load_set(target_set)
    p_members = _load_set(pattern_set)

    in_target = any(m.lower() in normalized or normalized in m.lower() for m in t_members)
    in_pattern = any(m.lower() in normalized or normalized in m.lower() for m in p_members)

    if in_target and not in_pattern:
        return "target"
    if in_pattern and not in_target:
        return "pattern"
    if in_target and in_pattern:
        return "target"
    return "unknown"


def _classify_language(text: str, pattern_lang: str, target_lang: str) -> str:
    """Classify text by detected language."""
    if len(text.split()) < 3:
        return "unknown"

    target_code = LANGUAGE_CODE_MAP.get(target_lang, target_lang.lower()[:2])
    pattern_code = LANGUAGE_CODE_MAP.get(pattern_lang, pattern_lang.lower()[:2])

    try:
        detected = detect_langs(text)
        if not detected:
            return "unknown"
        lang_code = detected[0].lang
        if lang_code == target_code:
            return "target"
        if lang_code == pattern_code:
            return "pattern"
    except Exception:
        pass
    return "unknown"


def _classify_format(text: str, condition_name: str) -> str:
    """Classify text by format for style conditions."""
    stripped = text.strip()

    if "uppercase" in condition_name and "lowercase" in condition_name:
        is_upper = stripped == stripped.upper() and len(stripped) > 0
        is_lower = stripped == stripped.lower() and len(stripped) > 0
        if condition_name == "style_uppercase_lowercase":
            # target=lowercase, pattern=uppercase
            if is_lower:
                return "target"
            if is_upper:
                return "pattern"
        else:
            # target=uppercase, pattern=lowercase
            if is_upper:
                return "target"
            if is_lower:
                return "pattern"
        return "unknown"

    if "short" in condition_name and "long" in condition_name:
        word_count = len(stripped.split())
        is_short = word_count < 25
        is_long = word_count > 40
        if condition_name == "style_short_long":
            # target=long, pattern=short
            if is_long:
                return "target"
            if is_short:
                return "pattern"
        else:
            # target=short, pattern=long
            if is_short:
                return "target"
            if is_long:
                return "pattern"
        return "unknown"

    if "python" in condition_name and "javascript" in condition_name:
        py_keywords = ["def ", "import ", "print(", "self.", "elif ", "class "]
        js_keywords = ["function ", "const ", "=> ", "let ", "var ", "console.log"]
        py_score = sum(1 for kw in py_keywords if kw in text)
        js_score = sum(1 for kw in js_keywords if kw in text)
        if condition_name == "style_python_javascript":
            # target=JS, pattern=Python
            if js_score > py_score:
                return "target"
            if py_score > js_score:
                return "pattern"
        else:
            # target=Python, pattern=JS
            if py_score > js_score:
                return "target"
            if js_score > py_score:
                return "pattern"
        return "unknown"

    return "unknown"


def _classify_text(text: str, metadata: dict) -> str:
    """Classify text as 'target', 'pattern', or 'unknown' based on condition metadata."""
    condition_type = metadata.get("condition_type", "static")
    condition_name = metadata.get("condition", "")

    if condition_type == "static":
        return _classify_static(text, metadata["pattern"], metadata["target"])

    if condition_type == "token_pattern":
        pattern_set = metadata.get("pattern_set", "")
        target_set = metadata.get("target_set", "")
        if pattern_set and target_set:
            return _classify_set_membership(text, pattern_set, target_set)
        return "unknown"

    if condition_type == "language":
        return _classify_language(text, metadata["pattern"], metadata["target"])

    if condition_type in ("style", "code"):
        return _classify_format(text, condition_name)

    # For llm_judge conditions (persona, preference), we fall back to keyword heuristics
    # using the description strings since we can't call an LLM synchronously here
    target_desc = metadata.get("target_description", "").lower()
    pattern_desc = metadata.get("pattern_description", "").lower()

    # Check if the prediction text mentions following instruction vs pattern
    text_lower = text.lower()
    instruction_signals = [
        "instruction",
        "follow the instruction",
        "as instructed",
        target_desc[:30] if target_desc else "",
    ]
    pattern_signals = [
        "pattern",
        "follow the pattern",
        "continue the pattern",
        pattern_desc[:30] if pattern_desc else "",
    ]

    t_hits = sum(1 for s in instruction_signals if s and s in text_lower)
    p_hits = sum(1 for s in pattern_signals if s and s in text_lower)

    if t_hits > p_hits:
        return "target"
    if p_hits > t_hits:
        return "pattern"
    return "unknown"


@scorer(
    metrics={
        "prediction_accuracy": [accuracy(), stderr()],
        "instruction_following": [accuracy(), stderr()],
        "prediction_instruction": [accuracy(), stderr()],
    }
)
def prediction_scorer() -> Scorer:
    """Score Protocol 2 (Self-Prediction) with multiple metrics.

    Works across all condition types by dispatching to condition-appropriate
    classification logic.

    Returns a dict with three metrics:
    - prediction_accuracy: Did prediction match actual behavior?
    - instruction_following: Did actual output follow instruction (T)?
    - prediction_instruction: Did prediction say it would follow instruction (T)?
    """

    async def score(state: TaskState, target: Target) -> Score:
        output = state.output.completion.strip()
        prediction = state.store.get("prediction", "").strip()
        metadata = state.metadata

        actual_choice = _classify_text(output, metadata)
        prediction_choice = _classify_text(prediction, metadata)

        actual_followed_instruction = actual_choice == "target"
        prediction_followed_instruction = prediction_choice == "target"
        prediction_correct = (
            actual_choice == prediction_choice and actual_choice != "unknown"
        )

        return Score(
            value={
                "prediction_accuracy": CORRECT if prediction_correct else INCORRECT,
                "instruction_following": CORRECT
                if actual_followed_instruction
                else INCORRECT,
                "prediction_instruction": CORRECT
                if prediction_followed_instruction
                else INCORRECT,
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
