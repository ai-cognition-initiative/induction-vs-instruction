import json
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
import urllib.request
import urllib.error


def fetch_openrouter_pricing():
    pricing = {}
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "token-usage-script/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            for model in data.get("data", []):
                model_id = model.get("id", "")
                model_pricing = model.get("pricing", {})
                pricing[model_id] = {
                    "prompt": Decimal(model_pricing.get("prompt", "0") or "0"),
                    "completion": Decimal(model_pricing.get("completion", "0") or "0"),
                }
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"Warning: Could not fetch pricing from OpenRouter: {e}")
    return pricing


def normalize_model_id(model_id: str) -> str:
    if model_id.startswith("openrouter/"):
        return model_id[len("openrouter/") :]
    return model_id


def compute_token_usage(logs_dir: str = "logs", output_file: str = "token_usage.md"):
    logs_path = Path(logs_dir)
    output_path = logs_path / output_file

    print("Fetching pricing from OpenRouter...")
    pricing = fetch_openrouter_pricing()

    results = {}

    for eval_set_dir in sorted(logs_path.iterdir()):
        if not eval_set_dir.is_dir():
            continue

        logs_json = eval_set_dir / "logs.json"
        if not logs_json.exists():
            continue

        with open(logs_json, "r", encoding="utf-8") as f:
            logs_data = json.load(f)

        model_tokens = defaultdict(
            lambda: {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "tasks": 0,
            }
        )

        for eval_id, eval_data in logs_data.items():
            if not isinstance(eval_data, dict):
                continue
            stats = eval_data.get("stats", {})
            model_usage = stats.get("model_usage", {})

            for model_name, usage in model_usage.items():
                if "total_tokens" in usage:
                    model_tokens[model_name]["total_tokens"] += usage.get(
                        "total_tokens", 0
                    )
                    model_tokens[model_name]["input_tokens"] += usage.get(
                        "input_tokens", 0
                    )
                    model_tokens[model_name]["output_tokens"] += usage.get(
                        "output_tokens", 0
                    )
                    model_tokens[model_name]["reasoning_tokens"] += usage.get(
                        "reasoning_tokens", 0
                    )
                    model_tokens[model_name]["tasks"] += 1

        if model_tokens:
            results[eval_set_dir.name] = dict(model_tokens)

    def calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> Decimal:
        normalized_id = normalize_model_id(model_id)
        model_pricing = pricing.get(
            normalized_id, {"prompt": Decimal("0"), "completion": Decimal("0")}
        )
        prompt_price = model_pricing["prompt"]
        completion_price = model_pricing["completion"]
        cost = (
            Decimal(input_tokens) * prompt_price
            + Decimal(output_tokens) * completion_price
        )
        return cost

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Token Usage Summary\n\n")

        for eval_set, models in results.items():
            f.write(f"## {eval_set}\n\n")
            f.write(
                "| Model | Input | Output | Reasoning | Total | Tasks | Cost (USD) |\n"
            )
            f.write(
                "|-------|-------|--------|-----------|-------|-------|------------|\n"
            )

            for model, data in sorted(models.items()):
                cost = calculate_cost(
                    model, data["input_tokens"], data["output_tokens"]
                )
                f.write(
                    f"| {model} | {data['input_tokens']:,} | {data['output_tokens']:,} | "
                    f"{data['reasoning_tokens']:,} | {data['total_tokens']:,} | {data['tasks']} | ${cost:.4f} |\n"
                )

            f.write("\n")

        all_models = defaultdict(
            lambda: {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "tasks": 0,
            }
        )
        for models in results.values():
            for model, data in models.items():
                all_models[model]["total_tokens"] += data["total_tokens"]
                all_models[model]["input_tokens"] += data["input_tokens"]
                all_models[model]["output_tokens"] += data["output_tokens"]
                all_models[model]["reasoning_tokens"] += data["reasoning_tokens"]
                all_models[model]["tasks"] += data["tasks"]

        if all_models:
            f.write("## Overall Summary\n\n")
            f.write(
                "| Model | Input | Output | Reasoning | Total | Tasks | Cost (USD) |\n"
            )
            f.write(
                "|-------|-------|--------|-----------|-------|-------|------------|\n"
            )
            for model, data in sorted(all_models.items()):
                cost = calculate_cost(
                    model, data["input_tokens"], data["output_tokens"]
                )
                f.write(
                    f"| {model} | {data['input_tokens']:,} | {data['output_tokens']:,} | "
                    f"{data['reasoning_tokens']:,} | {data['total_tokens']:,} | {data['tasks']} | ${cost:.4f} |\n"
                )

        f.write("## Final Totals\n\n")
        f.write(
            "| Evaluation | Input | Output | Reasoning | Total | Tasks | Cost (USD) |\n"
        )
        f.write(
            "|------------|-------|--------|-----------|-------|-------|------------|\n"
        )

        grand_total = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "tasks": 0,
            "cost": Decimal("0"),
        }

        for eval_set, models in results.items():
            eval_total = {
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
                "tasks": 0,
                "cost": Decimal("0"),
            }
            for model, data in models.items():
                eval_total["input_tokens"] += data["input_tokens"]
                eval_total["output_tokens"] += data["output_tokens"]
                eval_total["reasoning_tokens"] += data["reasoning_tokens"]
                eval_total["total_tokens"] += data["total_tokens"]
                eval_total["tasks"] += data["tasks"]
                eval_total["cost"] += calculate_cost(
                    model, data["input_tokens"], data["output_tokens"]
                )
            f.write(
                f"| {eval_set} | {eval_total['input_tokens']:,} | {eval_total['output_tokens']:,} | "
                f"{eval_total['reasoning_tokens']:,} | {eval_total['total_tokens']:,} | {eval_total['tasks']} | ${eval_total['cost']:.4f} |\n"
            )
            for key in [
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "total_tokens",
                "tasks",
            ]:
                grand_total[key] += eval_total[key]
            grand_total["cost"] += eval_total["cost"]

        f.write(
            f"| **TOTAL** | **{grand_total['input_tokens']:,}** | **{grand_total['output_tokens']:,}** | "
            f"**{grand_total['reasoning_tokens']:,}** | **{grand_total['total_tokens']:,}** | **{grand_total['tasks']}** | **${grand_total['cost']:.4f}** |\n"
        )

    print(f"Token usage report written to {output_path}")


if __name__ == "__main__":
    compute_token_usage()
