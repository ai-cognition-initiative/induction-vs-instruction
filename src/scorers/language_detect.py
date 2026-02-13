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
from langdetect import DetectorFactory, detect_langs

# Make langdetect deterministic
DetectorFactory.seed = 0

LANGUAGE_CODE_MAP = {
    "French": "fr",
    "Russian": "ru",
    "English": "en",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Korean": "ko",
}


@scorer(metrics=[accuracy(), stderr()])
def language_scorer(
    target_language: str,
    pattern_language: str,
) -> Scorer:
    """Score based on whether output is in the target language or pattern language."""

    target_code = LANGUAGE_CODE_MAP.get(target_language, target_language.lower()[:2])
    pattern_code = LANGUAGE_CODE_MAP.get(pattern_language, pattern_language.lower()[:2])

    async def score(state: TaskState, target: Target) -> Score:
        output = state.output.completion.strip()

        if len(output.split()) < 3:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation="Output too short for reliable language detection",
                metadata={"detected_language": "unknown", "confidence": 0.0},
            )

        try:
            detected = detect_langs(output)
            if not detected:
                return Score(
                    value=INCORRECT,
                    answer=output,
                    explanation="Language detection returned no results",
                    metadata={"detected_language": "unknown", "confidence": 0.0},
                )

            top_lang = detected[0]
            lang_code = top_lang.lang
            confidence = top_lang.prob

            if lang_code == target_code:
                return Score(
                    value=CORRECT,
                    answer=output,
                    explanation=f"Detected {lang_code} (target: {target_code}, confidence: {confidence:.2f})",
                    metadata={"detected_language": lang_code, "confidence": confidence},
                )
            elif lang_code == pattern_code:
                return Score(
                    value=INCORRECT,
                    answer=output,
                    explanation=f"Detected {lang_code} (pattern: {pattern_code}, confidence: {confidence:.2f})",
                    metadata={"detected_language": lang_code, "confidence": confidence},
                )
            else:
                return Score(
                    value=INCORRECT,
                    answer=output,
                    explanation=f"Detected {lang_code} (neither target nor pattern, confidence: {confidence:.2f})",
                    metadata={"detected_language": lang_code, "confidence": confidence},
                )

        except Exception as e:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation=f"Language detection failed: {e}",
                metadata={"detected_language": "error", "confidence": 0.0},
            )

    return score
