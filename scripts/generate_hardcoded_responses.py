"""Generate pre-computed hardcoded response JSON files for all conditions.

Uses OpenRouter API to generate responses for every question in each relevant
question bank, so any runtime sampling will find a match.

Usage:
    uv run python scripts/generate_hardcoded_responses.py
    uv run python scripts/generate_hardcoded_responses.py --model anthropic/claude-3-5-haiku
    uv run python scripts/generate_hardcoded_responses.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

MAX_WORKERS = 20

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "hardcoded_responses"
DEFAULT_MODEL = "anthropic/claude-3-5-haiku"

# Each entry: (output_filename, question_bank_file, system_prompt, description)
LLM_TASKS: list[tuple[str, str, str, str]] = [
    (
        "language_french",
        "questions_factual.json",
        "You are a translator. Translate the user's question into French. "
        "Output ONLY the French translation, nothing else.",
        "French translations of factual questions",
    ),
    (
        "language_russian",
        "questions_factual.json",
        "You are a translator. Translate the user's question into Russian. "
        "Output ONLY the Russian translation, nothing else.",
        "Russian translations of factual questions",
    ),
    (
        "persona_formal",
        "questions_factual.json",
        "You are a formal academic advisor. Answer the user's question with "
        "scholarly precision, proper grammar, and intellectual rigor. "
        "Keep your answer to 1-3 sentences.",
        "Formal academic answers to factual questions",
    ),
    (
        "persona_casual",
        "questions_factual.json",
        "You are a laid-back, friendly chatbot. Answer the user's question "
        "like you're talking to a close friend — use slang, contractions, "
        "and emoji naturally. Keep your answer to 1-3 sentences.",
        "Casual answers with emoji/slang to factual questions",
    ),
    (
        "style_base",
        "questions_factual.json",
        "Answer the user's question briefly and neutrally in 1-2 sentences.",
        "Brief neutral answers (base for computed style transforms)",
    ),
    (
        "code_python",
        "questions_programming.json",
        "Answer the user's question using ONLY Python code. "
        "Output a short Python snippet that solves the problem. "
        "No prose, no markdown fences — just raw Python code.",
        "Python code answers to programming questions",
    ),
    (
        "code_javascript",
        "questions_programming.json",
        "Answer the user's question using ONLY JavaScript code. "
        "Output a short JavaScript snippet that solves the problem. "
        "No prose, no markdown fences — just raw JavaScript code.",
        "JavaScript code answers to programming questions",
    ),
    (
        "preference_love_cats",
        "questions_subjective.json",
        "Answer the user's question naturally, but weave in the fact that "
        "you absolutely love cats. Find a way to mention cats, reference cats, "
        "or connect the topic to cats in a genuine way. "
        "Keep your answer to 2-4 sentences.",
        "Answers weaving in love of cats (subjective questions)",
    ),
    (
        "preference_hate_cats",
        "questions_subjective.json",
        "Answer the user's question naturally, but weave in the fact that "
        "you really dislike cats. Find a way to mention your distaste for cats "
        "or connect the topic to your dislike of cats in a genuine way. "
        "Keep your answer to 2-4 sentences.",
        "Answers weaving in dislike of cats (subjective questions)",
    ),
]

# Computed from style_base: (output_filename, transform_fn, description)
COMPUTED_TASKS: list[tuple[str, str]] = [
    ("style_uppercase", "UPPERCASE transforms of base answers"),
    ("style_lowercase", "lowercase transforms of base answers"),
]


def load_questions(filename: str) -> list[str]:
    path = DATA_DIR / filename
    with open(path) as f:
        data = json.load(f)
    return data["questions"]


def _call_llm(
    client: OpenAI, model: str, system_prompt: str, question: str
) -> tuple[str, str]:
    """Single LLM call. Returns (question, response)."""
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return question, resp.choices[0].message.content.strip()


def generate_llm_responses(
    client: OpenAI,
    model: str,
    questions: list[str],
    system_prompt: str,
    desc: str,
) -> dict[str, str]:
    responses: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_call_llm, client, model, system_prompt, q): q
            for q in questions
        }
        for future in tqdm(
            as_completed(futures), total=len(questions), desc=desc, leave=False
        ):
            question, response = future.result()
            responses[question] = response
    return responses


def save_responses(
    filename: str, description: str, responses: dict[str, str]
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{filename}.json"
    data = {
        "condition_description": description,
        "responses": responses,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {path} ({len(responses)} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hardcoded response JSON files for all conditions."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing files",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set in environment.", file=sys.stderr)
        print("Set it in .env or export it before running.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # --- LLM-generated files ---
    for filename, bank_file, system_prompt, description in LLM_TASKS:
        path = OUTPUT_DIR / f"{filename}.json"
        if path.exists() and not args.force:
            print(f"Skipping {filename} (exists, use --force to regenerate)")
            continue

        print(f"Generating {filename}...")
        questions = load_questions(bank_file)
        responses = generate_llm_responses(
            client, args.model, questions, system_prompt, filename
        )
        save_responses(filename, description, responses)

    # --- Computed files (from style_base) ---
    base_path = OUTPUT_DIR / "style_base.json"
    if not base_path.exists():
        print("Error: style_base.json must exist before computing derived files.")
        sys.exit(1)

    with open(base_path) as f:
        base_data = json.load(f)
    base_responses = base_data["responses"]

    for filename, description in COMPUTED_TASKS:
        path = OUTPUT_DIR / f"{filename}.json"
        if path.exists() and not args.force:
            print(f"Skipping {filename} (exists, use --force to regenerate)")
            continue

        print(f"Computing {filename}...")
        if filename == "style_uppercase":
            transformed = {q: r.upper() for q, r in base_responses.items()}
        elif filename == "style_lowercase":
            transformed = {q: r.lower() for q, r in base_responses.items()}
        else:
            raise ValueError(f"Unknown computed task: {filename}")

        save_responses(filename, description, transformed)

    print("\nDone! All hardcoded response files generated.")


if __name__ == "__main__":
    main()
