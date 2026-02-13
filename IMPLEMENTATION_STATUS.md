# Condition Implementation Status

| Condition | Type | Scorer | Hardcoded Data | Status |
|-----------|------|--------|----------------|--------|
| neutral | static | string_match | — | Ready |
| value_pattern | static | string_match | — | Ready |
| value_target | static | string_match | — | Ready |
| factual_pattern | static | string_match | — | Ready |
| factual_target | static | string_match | — | Ready |
| token_pattern_states | token_pattern | set_membership | — | Ready |
| token_pattern_countries | token_pattern | set_membership | — | Ready |
| language_fr_ru | language | language_detect | language_french | Missing data |
| language_ru_fr | language | language_detect | language_russian | Missing data |
| persona_formal_casual | persona | llm_judge | persona_formal | Missing data |
| persona_casual_formal | persona | llm_judge | persona_casual | Missing data |
| style_uppercase_lowercase | style | format_check | — | Ready |
| style_lowercase_uppercase | style | format_check | — | Ready |
| style_short_long | style | format_check | style_long | Missing data |
| style_long_short | style | format_check | style_long | Missing data |
| style_python_javascript | code | format_check | code_python | Missing data |
| style_javascript_python | code | format_check | code_javascript | Missing data |
| preference_cats_dogs | preference | llm_judge | preference_dogs | Missing data |
| preference_dogs_cats | preference | llm_judge | preference_cats | Missing data |

**Ready: 9/19** — Conditions requiring pre-generated hardcoded responses need `data/hardcoded_responses/*.json` files.
