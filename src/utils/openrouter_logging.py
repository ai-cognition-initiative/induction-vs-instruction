"""OpenRouter provider logging integration for inspect-ai.

Logs OpenRouter-specific metadata (provider, pricing, model info) to
inspect-ai's logging system during evaluations.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error

from decimal import Decimal
from typing import Any

from inspect_ai.model import ModelOutput, ModelUsage

logger = logging.getLogger(__name__)

_PRICING_CACHE: dict[str, dict[str, Decimal]] = {}
_PROVIDER_INFO_CACHE: dict[str, dict[str, Any]] = {}


def fetch_openrouter_model_info(model_id: str) -> dict[str, Any]:
    """Fetch model info from OpenRouter API."""
    if model_id in _PROVIDER_INFO_CACHE:
        return _PROVIDER_INFO_CACHE[model_id]

    normalized_id = model_id.removeprefix("openrouter/")

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "inspect-ai-openrouter-logging/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            for model in data.get("data", []):
                if model.get("id") == normalized_id:
                    info = {
                        "id": model.get("id"),
                        "name": model.get("name"),
                        "context_length": model.get("context_length"),
                        "pricing": model.get("pricing", {}),
                        "top_provider": model.get("top_provider", {}),
                        "per_request_limits": model.get("per_request_limits"),
                        "architecture": model.get("architecture", {}),
                    }
                    _PROVIDER_INFO_CACHE[model_id] = info
                    return info
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        logger.warning(f"Could not fetch OpenRouter model info: {e}")

    return {}


def log_openrouter_metadata(model: str) -> dict[str, Any]:
    """Log OpenRouter provider metadata at eval start.

    Call this from your task or solver to record provider info.
    Returns metadata dict suitable for sample/store metadata.
    """
    if not model.startswith("openrouter/"):
        return {}

    info = fetch_openrouter_model_info(model)

    if info:
        logger.info(f"OpenRouter model: {info.get('name', model)}")
        logger.info(f"Context length: {info.get('context_length', 'unknown')}")

        pricing = info.get("pricing", {})
        if pricing:
            prompt_price = pricing.get("prompt", "0")
            completion_price = pricing.get("completion", "0")
            logger.info(
                f"Pricing: ${prompt_price}/1M prompt, ${completion_price}/1M completion"
            )

        top_provider = info.get("top_provider", {})
        if top_provider:
            logger.debug(f"Provider info: {top_provider}")

    return {
        "openrouter_model_info": info,
        "provider": "openrouter",
    }


def log_model_output_usage(output: ModelOutput, model: str) -> None:
    """Log token usage from a ModelOutput.

    Call this after model generation to log detailed usage stats.
    """
    if not model.startswith("openrouter/"):
        return

    usage = output.usage
    if usage:
        logger.debug(
            f"Token usage - input: {usage.input_tokens}, "
            f"output: {usage.output_tokens}, "
            f"total: {usage.total_tokens}"
        )
        if usage.reasoning_tokens:
            logger.debug(f"Reasoning tokens: {usage.reasoning_tokens}")

        if usage.total_tokens > 0:
            pricing = _get_pricing(model)
            if pricing:
                cost = Decimal(usage.input_tokens) * pricing["prompt"] / Decimal(
                    "1_000_000"
                ) + Decimal(usage.output_tokens) * pricing["completion"] / Decimal(
                    "1_000_000"
                )
                logger.info(f"Estimated cost: ${cost:.6f}")


def _get_pricing(model_id: str) -> dict[str, Decimal] | None:
    """Get pricing for a model, fetching from API if needed."""
    if model_id in _PRICING_CACHE:
        return _PRICING_CACHE[model_id]

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "inspect-ai-openrouter-logging/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            for model in data.get("data", []):
                mid = model.get("id", "")
                model_pricing = model.get("pricing", {})
                _PRICING_CACHE[f"openrouter/{mid}"] = {
                    "prompt": Decimal(model_pricing.get("prompt", "0") or "0"),
                    "completion": Decimal(model_pricing.get("completion", "0") or "0"),
                }
            return _PRICING_CACHE.get(model_id)
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        logger.warning(f"Could not fetch pricing from OpenRouter: {e}")

    return None


class OpenRouterUsageTracker:
    """Track cumulative OpenRouter usage across an evaluation.

    Usage:
        tracker = OpenRouterUsageTracker()
        # ... after each model call ...
        tracker.record(usage)
        # ... at end of eval ...
        tracker.log_summary()
    """

    def __init__(self, model: str):
        self.model = model
        self.total_input = 0
        self.total_output = 0
        self.total_reasoning = 0
        self.call_count = 0

    def record(self, usage: ModelUsage | None) -> None:
        """Record usage from a model call."""
        if not usage:
            return

        self.total_input += usage.input_tokens
        self.total_output += usage.output_tokens
        self.total_reasoning += usage.reasoning_tokens or 0
        self.call_count += 1

    def log_summary(self) -> dict[str, Any]:
        """Log and return a summary of tracked usage."""
        summary = {
            "model": self.model,
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_reasoning_tokens": self.total_reasoning,
            "total_tokens": self.total_input + self.total_output,
            "call_count": self.call_count,
        }

        logger.info(f"OpenRouter usage summary for {self.model}:")
        logger.info(f"  Total input tokens: {self.total_input:,}")
        logger.info(f"  Total output tokens: {self.total_output:,}")
        if self.total_reasoning:
            logger.info(f"  Total reasoning tokens: {self.total_reasoning:,}")
        logger.info(f"  Total calls: {self.call_count}")

        pricing = _get_pricing(self.model)
        if pricing:
            cost = Decimal(self.total_input) * pricing["prompt"] / Decimal(
                "1_000_000"
            ) + Decimal(self.total_output) * pricing["completion"] / Decimal(
                "1_000_000"
            )
            summary["estimated_cost_usd"] = float(cost)
            logger.info(f"  Estimated total cost: ${cost:.4f}")

        return summary
