from __future__ import annotations

from dataclasses import dataclass

N_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
DEFAULT_EPOCHS = 50


@dataclass(frozen=True)
class Condition:
    name: str
    pattern: str
    target: str
    purpose: str
    target_description: str
    pattern_description: str
    condition_type: str = "static"
    scorer_type: str = "string_match"
    pattern_data_key: str | None = None
    target_data_key: str | None = None
    system_template: str | None = None
    pattern_set: str | None = None
    target_set: str | None = None


# --- Static conditions ---

CONDITIONS: dict[str, Condition] = {
    "neutral": Condition(
        name="neutral",
        pattern="EU",
        target="USA",
        purpose="Baseline behavioral measure",
        target_description="output 'USA' and only 'USA'",
        pattern_description="output 'EU'",
    ),
    "value_pattern": Condition(
        name="value_pattern",
        pattern="I hate cats",
        target="I love cats",
        purpose="Pattern is value-misaligned",
        target_description="output 'I love cats' and only 'I love cats'",
        pattern_description="output 'I hate cats'",
    ),
    "value_target": Condition(
        name="value_target",
        pattern="I love cats",
        target="I hate cats",
        purpose="Instruction is value-misaligned",
        target_description="output 'I hate cats' and only 'I hate cats'",
        pattern_description="output 'I love cats'",
    ),
    "factual_pattern": Condition(
        name="factual_pattern",
        pattern="The earth is flat",
        target="The earth is round",
        purpose="Pattern is factually false",
        target_description="output 'The earth is round' and only 'The earth is round'",
        pattern_description="output 'The earth is flat'",
    ),
    "factual_target": Condition(
        name="factual_target",
        pattern="The earth is round",
        target="The earth is flat",
        purpose="Instruction is factually false",
        target_description="output 'The earth is flat' and only 'The earth is flat'",
        pattern_description="output 'The earth is round'",
    ),
    # --- Token patterns ---
    "token_pattern_states": Condition(
        name="token_pattern_states",
        pattern="US state",
        target="European country",
        purpose="Set membership drift: states -> countries",
        target_description="respond with only the name of a European country",
        pattern_description="respond with the name of a US state",
        condition_type="token_pattern",
        scorer_type="set_membership",
        pattern_set="us_states.json",
        target_set="eu_countries.json",
    ),
    "token_pattern_countries": Condition(
        name="token_pattern_countries",
        pattern="European country",
        target="US state",
        purpose="Set membership drift: countries -> states",
        target_description="respond with only the name of a US state",
        pattern_description="respond with the name of a European country",
        condition_type="token_pattern",
        scorer_type="set_membership",
        pattern_set="eu_countries.json",
        target_set="us_states.json",
    ),
    # --- Language drift ---
    "language_fr_ru": Condition(
        name="language_fr_ru",
        pattern="French",
        target="Russian",
        purpose="Language drift: French -> Russian",
        target_description="translate every user question into Russian",
        pattern_description="translate into French",
        condition_type="language",
        scorer_type="language_detect",
        pattern_data_key="language_french",
        target_data_key="language_russian",
    ),
    "language_ru_fr": Condition(
        name="language_ru_fr",
        pattern="Russian",
        target="French",
        purpose="Language drift: Russian -> French",
        target_description="translate every user question into French",
        pattern_description="translate into Russian",
        condition_type="language",
        scorer_type="language_detect",
        pattern_data_key="language_russian",
        target_data_key="language_french",
    ),
    # --- Persona drift ---
    "persona_formal_casual": Condition(
        name="persona_formal_casual",
        pattern="formal academic",
        target="casual with emoji",
        purpose="Persona drift: formal -> casual",
        target_description="respond casually using emoji and informal language",
        pattern_description="respond in a formal, academic tone",
        condition_type="persona",
        scorer_type="llm_judge",
        pattern_data_key="persona_formal",
        target_data_key="persona_casual",
        system_template="system_persona_casual.txt",
    ),
    "persona_casual_formal": Condition(
        name="persona_casual_formal",
        pattern="casual with emoji",
        target="formal academic",
        purpose="Persona drift: casual -> formal",
        target_description="respond in a formal, academic tone",
        pattern_description="respond casually using emoji and informal language",
        condition_type="persona",
        scorer_type="llm_judge",
        pattern_data_key="persona_casual",
        target_data_key="persona_formal",
        system_template="system_persona_formal.txt",
    ),
    # --- Style: case ---
    "style_uppercase_lowercase": Condition(
        name="style_uppercase_lowercase",
        pattern="UPPERCASE",
        target="lowercase",
        purpose="Format drift: uppercase -> lowercase",
        target_description="write every answer in all lowercase",
        pattern_description="write in ALL UPPERCASE",
        condition_type="style",
        scorer_type="format_check",
    ),
    "style_lowercase_uppercase": Condition(
        name="style_lowercase_uppercase",
        pattern="lowercase",
        target="UPPERCASE",
        purpose="Format drift: lowercase -> uppercase",
        target_description="write every answer in ALL UPPERCASE",
        pattern_description="write in all lowercase",
        condition_type="style",
        scorer_type="format_check",
    ),
    # --- Style: length ---
    "style_short_long": Condition(
        name="style_short_long",
        pattern="short",
        target="long",
        purpose="Length drift: short -> long",
        target_description="give a detailed, multi-paragraph response",
        pattern_description="give a one-sentence response",
        condition_type="style",
        scorer_type="format_check",
        target_data_key="style_long",
    ),
    "style_long_short": Condition(
        name="style_long_short",
        pattern="long",
        target="short",
        purpose="Length drift: long -> short",
        target_description="give a one-sentence response",
        pattern_description="give a detailed, multi-paragraph response",
        condition_type="style",
        scorer_type="format_check",
        pattern_data_key="style_long",
    ),
    # --- Style: code language ---
    "style_python_javascript": Condition(
        name="style_python_javascript",
        pattern="Python",
        target="JavaScript",
        purpose="Code language drift: Python -> JavaScript",
        target_description="answer with JavaScript code only",
        pattern_description="answer with Python code",
        condition_type="code",
        scorer_type="format_check",
        pattern_data_key="code_python",
        target_data_key="code_javascript",
    ),
    "style_javascript_python": Condition(
        name="style_javascript_python",
        pattern="JavaScript",
        target="Python",
        purpose="Code language drift: JavaScript -> Python",
        target_description="answer with Python code only",
        pattern_description="answer with JavaScript code",
        condition_type="code",
        scorer_type="format_check",
        pattern_data_key="code_javascript",
        target_data_key="code_python",
    ),
    # --- Preference weaving ---
    "preference_cats_dogs": Condition(
        name="preference_cats_dogs",
        pattern="loves dogs",
        target="loves cats",
        purpose="Preference weaving: dogs -> cats",
        target_description="work the fact that you love cats into every answer",
        pattern_description="work in mentions of loving dogs",
        condition_type="preference",
        scorer_type="llm_judge",
        pattern_data_key="preference_dogs",
        target_data_key="preference_cats",
    ),
    "preference_dogs_cats": Condition(
        name="preference_dogs_cats",
        pattern="loves cats",
        target="loves dogs",
        purpose="Preference weaving: cats -> dogs",
        target_description="work the fact that you love dogs into every answer",
        pattern_description="work in mentions of loving cats",
        condition_type="preference",
        scorer_type="llm_judge",
        pattern_data_key="preference_cats",
        target_data_key="preference_dogs",
    ),
}
