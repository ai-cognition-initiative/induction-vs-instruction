from __future__ import annotations

# Core T=0 models — non-reasoning only (static + dynamic main results, §3.1–3.2)
CORE_MODELS = [
    "claude-4.6-sonnet", "claude-opus-4.6", "gemini-2.5-flash",
    "gemma-3-12b-it", "gemma-3-27b-it", "gpt-5.2",
    "hermes-4-70b", "kimi-k2-instruct",
  #  "llama-3.1-70b-instruct", 
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

# Normalize variant names to canonical names used in CORE_MODELS / TRAINING_MODELS.
# e.g., static-T1 uses "claude-sonnet-4.6" while T0 data uses "claude-4.6-sonnet"
MODEL_ALIASES: dict[str, str] = {
    "claude-sonnet-4.6": "claude-4.6-sonnet",
    # extend as new datasets are added
}

DISPLAY_NAMES: dict[str, str] = {
    "claude-4.6-sonnet": "Claude 4.6 Sonnet",
    "claude-opus-4.6": "Claude 4.6 Opus ",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemma-3-12b-it": "Gemma-3 12B",
    "gemma-3-27b-it": "Gemma-3 27B",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.2-medium": "GPT-5.2-medium",
    "hermes-4-70b": "Hermes-4 70B",
    "hermes-4-70b-reasoning": "Hermes-4 70B (reasoning)",
    "kimi-k2-instruct": "Kimi K2",
    "llama-3.1-70b-instruct": "Llama 3.1 70B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "olmo-3.1-32b-instruct": "OLMo 3.1 32B Instruct",
    "olmo-3.1-32b-instruct-dpo": "OLMo 3.1 32B (SFT+DPO)",
    "olmo-3.1-32b-instruct-sft": "OLMo 3.1 32B (SFT)",
    "qwen3-235b-a22b-instruct-2507": "Qwen3 235B A22B",
    "qwen3-30b-a3b-instruct-2507": "Qwen3 30B A3B",
}
