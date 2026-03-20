from __future__ import annotations

from inspect_ai.scorer import Scorer

from src.config import Condition
from src.scorers.classify import classify_actual, classify_prediction
from src.scorers.classify_fixed import classify_fixed_scorer
from src.scorers.format_check import format_scorer
from src.scorers.language_detect import language_scorer
from src.scorers.pattern_match import pattern_match
from src.scorers.prediction import prediction_scorer
from src.scorers.set_membership import set_membership_scorer
from src.scorers.style_judge import style_scorer

__all__ = [
    "classify_actual",
    "classify_fixed_scorer",
    "classify_prediction",
    "format_scorer",
    "get_behavioral_scorer",
    "language_scorer",
    "pattern_match",
    "prediction_scorer",
    "set_membership_scorer",
    "style_scorer",
]


def get_behavioral_scorer(condition: Condition) -> Scorer:
    """Return the appropriate behavioral scorer based on condition's scorer_type."""
    match condition.scorer_type:
        case "string_match":
            return pattern_match()
        case "set_membership":
            return set_membership_scorer(
                pattern_set=condition.pattern_set,
                target_set=condition.target_set,
            )
        case "language_detect":
            return language_scorer(
                target_language=condition.target,
                pattern_language=condition.pattern,
            )
        case "llm_judge":
            return style_scorer(
                target_description=condition.target_description,
                pattern_description=condition.pattern_description,
                condition_type=condition.condition_type,
            )
        case "format_check":
            return format_scorer(condition_name=condition.name)
        case "classify_fixed":
            return classify_fixed_scorer(condition_name=condition.name)
        case _:
            raise ValueError(f"Unknown scorer_type: {condition.scorer_type}")
