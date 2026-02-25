+-- ./
|   `-- run.py
|       PROTOCOL_MAP, load_models_from_yaml, main
|
+-- notebooks/
|   `-- inspect_log_viewer.py
|       _, _, _, _, _, _, _
|
+-- scripts/
|   +-- generate_hardcoded_responses.py
|   |   MAX_WORKERS, DATA_DIR, OUTPUT_DIR, DEFAULT_MODEL, load_questions, _call_llm, generate_llm_responses, save_responses, main
|   +-- generate_report_from_config.py
|   |   load_config, validate_config, get_quarto_python, render_notebook, main
|   +-- prepare_viz_data.py
|   |   CONDITION_PAIR_MAP, ALIGNED_CONDITIONS, MISALIGNED_CONDITIONS, ALIGNMENT_AXIS_PAIRS, PREDICTION_SCORE_RENAME, add_pairing_columns, _coalesce_scores, _rename_prediction_scores, _common_prep, prepare_behavioral, prepare_prediction, prepare_combined, load_evals_from_folders, process_raw_evals, process_behavioral_df, process_prediction_df, _merge_behavioral_prediction, prepare_behavioral_multi, prepare_prediction_multi, prepare_combined_multi
|   +-- repomap.py
|   |   extract_symbols, generate_repomap
|   `-- viz_helpers.py
|       labeled_line_chart, bullet_graph, paired_bullet_graph, calibration_line_chart, overview_heatmap
|
+-- src/
|   +-- __init__.py
|   `-- config.py
|       N_VALUES, DEFAULT_N_TRIALS, Condition
|
+-- src\datasets/
|   +-- __init__.py
|   +-- questions.py
|   |   DEFAULT_QUESTIONS_PATH, load_questions, get_question_sequence, sample_questions
|   `-- sample_builder.py
|       PROMPTS_DIR, DATA_DIR, _build_metadata, _load_prompt, _load_set, _load_hardcoded_responses, _get_hardcoded_response, _build_conversation, build_behavioral_sample, build_prediction_sample
|
+-- src\scorers/
|   +-- __init__.py
|   |   get_behavioral_scorer
|   +-- classify.py
|   |   DATA_DIR, GRADER_MODEL, LANGUAGE_CODE_MAP, ACTUAL_OUTPUT_RUBRIC, PREDICTION_RUBRIC, _load_set, _fuzzy_match, classify_static, classify_set_membership, classify_language, classify_format, _call_llm_judge, classify_llm_actual, classify_llm_prediction, classify_actual, classify_prediction
|   +-- format_check.py
|   |   _check_uppercase, _check_lowercase, _check_short, _check_long, _check_python, _check_javascript, format_scorer
|   +-- language_detect.py
|   |   LANGUAGE_CODE_MAP, language_scorer
|   +-- pattern_match.py
|   |   pattern_match, _fuzzy_match
|   +-- prediction.py
|   |   prediction_scorer
|   +-- set_membership.py
|   |   DATA_DIR, _load_set, _normalize, _fuzzy_set_match, set_membership_scorer
|   `-- style_judge.py
|       PERSONA_RUBRIC, PREFERENCE_RUBRIC, style_scorer
|
+-- src\solvers/
|   +-- __init__.py
|   `-- protocols.py
|       PROMPTS_DIR, _load_prompt, behavioral_solver, prediction_solver
|
+-- src\tasks/
|   +-- __init__.py
|   +-- behavioral.py
|   |   behavioral_baseline
|   `-- prediction.py
|       self_prediction
|
+-- src\utils/
|   +-- __init__.py
|   +-- openrouter_logging.py
|   |   OpenRouterUsageTracker(__init__, record, log_summary), fetch_openrouter_model_info, log_openrouter_metadata, log_model_output_usage, _get_pricing
|   `-- token_usage.py
|       fetch_openrouter_pricing, normalize_model_id, compute_token_usage
|
