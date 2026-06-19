from __future__ import annotations

from src.plotting_utils import DISPLAY_NAMES as DISPLAY_NAMES  # noqa: F401  # re-exported

# Core T=0 models — non-reasoning only (static + dynamic main results, §3.1–3.2)
CORE_MODELS = [
    "claude-sonnet-4.6", "claude-opus-4.6", "gemini-2.5-flash",
    "gemma-3-12b-it", "gemma-3-27b-it", "gpt-5.2",
    "hermes-4-70b", "kimi-k2",
    "llama-3.1-70b-instruct",
    "llama-3.3-70b-instruct",
    "olmo-3.1-32b-instruct", "qwen3-235b-a22b-instruct-2507",
    "qwen3-30b-a3b-instruct-2507",
]

# Reasoning variants — used for §3.4 reasoning analysis only
REASONING_MODELS = [
    "gpt-5.2-medium",           # GPT-5.2 with reasoning (medium budget)
    "hermes-4-70b-reasoning",   # Hermes-4 70B with CoT reasoning
]

TRAINING_MODELS = [
    "olmo-3.1-32b-instruct-sft", "olmo-3.1-32b-instruct-dpo",
    "olmo-3.1-32b-instruct", "llama-3.1-70b-instruct", "llama-3.3-70b-instruct",
]

# Normalize legacy/variant model names to canonical names used in CORE_MODELS /
# TRAINING_MODELS. Canonical names match what prepare_viz_data.py emits via
# MODEL_CANONICAL after the pipeline change. Older parquets that pre-date that
# change may still contain the legacy keys on the left.
MODEL_ALIASES: dict[str, str] = {
    "claude-4.6-sonnet": "claude-sonnet-4.6",  # Gradient/legacy spelling
    "kimi-k2-instruct": "kimi-k2",  # Nebius/old OpenRouter spelling
    "kimi-k2-0905": "kimi-k2",  # OpenRouter 09-05 release spelling
}

