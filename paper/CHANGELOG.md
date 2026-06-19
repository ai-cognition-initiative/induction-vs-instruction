# Paper changelog

Tracks non-obvious edits to `main.tex` / `arxiv.tex` and shared
`sections/`, `figures/`, `articles.bib`, `colm2026_conference.bib`.
Pure typo fixes (driver, Llama, Protocol 1 spacing, missing space after
`26.8%).`, Martín diacritic, Yin & Steinhardt 2025 year) are not
individually listed.

## 2026-05-19

### Naming standardization

- **Claude model display names** standardized to `Claude Opus 4.6` and
  `Claude Sonnet 4.6` throughout prose, tables, and figure captions
  (previously mixed with `Claude 4.6 Sonnet`).
  - Note: OpenRouter model identifiers in Appendix model list
    (`anthropic/claude-opus-4.6`, `anthropic/claude-4.6-sonnet`) are
    left as-is because they are API identifiers, not display names.
  - **TODO**: Figure 2 heatmap PNGs (`outputs/plots/.../heatmap_*.png`)
    still show `Claude 4.6 Opus` / `Claude 4.6 Sonnet`. Regenerate plots
    to update bar labels.

- **Follow-up condition family** renamed from `Variety` to
  `random-facts` throughout Appendix J (Table 11, paragraph headings,
  prose, and inline references). Section 3.2 also updated
  (`random-question` → `random-facts`).
  - Internal condition keys updated in Table 11:
    `variety_animals_geography` → `random_animals_geography`,
    `variety_geography_animals` → `random_geography_animals`.
  - **TODO**: Figure 19 PNG (`f3_followup_by_model.png`) bar labels
    still read `random-facts-animals` / `random-facts-geogra...` for
    bars while the caption now says `random-facts`. Verify labels are
    consistent on re-render.

- **Follow-up condition count**: Methods §2.2 changed from
  `Three additional follow-up conditions` to
  `Two additional follow-up conditions`, treating the two random-facts
  directions as a single family alongside the classify condition.
  Section 3.2 wording ("two follow-up conditions") was already
  consistent; Appendix tables still list three specific conditions
  (two directions + classify), matching the "16 different instructions"
  count in the abstract (13 main + 3 follow-up conditions).

### Numerical corrections

- **Llama 3.3 70B at $N{=}50$**: §3.2 value corrected from `0.74` to
  `0.95`. The 0.74 was computed over all 7 static conditions including
  the two token-set conditions, which are excluded from the paper.
  The correct value over the 5 core fixed-output conditions is 0.95,
  consistent with Appendix H.1 and the heatmap cell.

- **Hermes-4 70B fixed-output Avg IF**: Table 4 (`tab_reasoning.tex`)
  corrected from `3% [0–6%]` to `2% [0–5%]` to match Table 2
  (`tab_results.tex`). Prose in §3.5 (`Hermes-4 70B (0.03)`) updated
  to `(0.02)`.

- **MMLU Pro** removed from the main text and from
  `fig_capability.tex` caption: Figure 8 only plots GPQA and IFBench,
  so the previous reference to "three external benchmarks" with the
  parenthetical `MMLU Pro r=−0.36, p=0.29` was inconsistent with the
  figure. Appendix capability section unchanged (verify whether any
  appendix subplots still show MMLU Pro; if not, the bib entry
  `wang_mmlu-pro_2024` becomes orphaned and could be removed).

### Wording reconciliation

- **§3.5 self-prediction effect**: Previous text said the overall
  effect "is not significant" while Figure 5b reports $p = 0.004$.
  Reworded to "statistically detectable but small in magnitude
  ($\Delta\text{IF} \approx -0.026$, $p = 0.004$) ... does not produce
  a practically meaningful shift" to be consistent with the figure
  caption.

### Bibliography

- `colm2026_conference.bib`: added `year = {2025}` to `yinattention`
  (Yin & Steinhardt, "Which attention heads matter for in-context
  learning?", ICML 2025). Inline cite previously rendered without a
  year.
- `articles.bib`: fixed broken diacritic `Martın` → `Martín` in
  `betleyTELLMEYOURSELF`; added `year = {2025}` since the entry
  previously had none.
- Rebuild via `bibtex` (or `latexmk -pdf`) required to regenerate
  `main.bbl` from the updated `.bib` files so the year and Martín
  diacritic appear in the rendered PDFs.

## Open TODOs (author input needed)

- **Regenerate Figure 2 heatmaps** to update Claude model bar labels.

- **Regenerate Figure 19 follow-up bar plot** if internal condition
  keys changed (random_* instead of variety_*).
