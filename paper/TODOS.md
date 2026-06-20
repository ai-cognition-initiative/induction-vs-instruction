# Review checklist — *Do as I Say, Not as I Do: Instruction-Induction Conflict in LLMs*

Items are grouped by severity. References point to line numbers / tables / figures in the submitted PDF.

Status legend: `[x]` done · `[~]` partial · `[ ]` open (needs author input). Both wrappers
(`main.tex`, `arxiv.tex`) rebuild clean after these edits (EXIT 0, 0 undefined refs).

---

## A. Substantive — numbers that disagree across tables/text

- [x] **GPT-5.2 (non-reasoning), fixed-output Avg IF disagrees.** ~~Table 2 = 10%, Table 5/§3.6 = 17%.~~
  **Root cause:** Table 5 (reasoning) was computed *including* the two token conditions, which the
  rest of the paper excludes. Recomputed the **whole** reasoning table over the 5 fixed-output
  conditions (token-excluded) so it matches Table 2 / §3.1. GPT-5.2 non-reasoning fixed = **10%
  [5–15], N₅₀=1** everywhere. (Knock-on: GPT-5.2-medium fixed 64→**53%** with N₅₀=31;
  Hermes-reasoning 68→**72%**; §3.6 prose updated to match.) ⚠️ *flagged below — sanity-check the
  medium 64→53 change.*

- [x] **Table 12 (temperature) reports fixed-output only, but text said "all conditions."** Setup
  text now reads "across the five fixed-output conditions" (the T=1 sweep only covers fixed-output
  conditions — confirmed from the parquet).

- [x] **GPT-5.2 in Table 12 (T=0 = 53%) didn't match Table 5.** Auto-resolved: 53% is
  GPT-5.2-medium over the 5 fixed-output conditions; once Table 5 is token-excluded its
  medium fixed-output cell is also **53%**, so they now agree.

- [x] **Scorer method contradiction (language + code-style).** Ground truth from `src/config.py`:
  `language_*` → `language_detect`, code-style → `format_check` (both deterministic); only
  `persona_*`/`preference_*` use `llm_judge`. Fixed §2.4, App L, App M to match Table 7 + App A.
  The judge panel is reframed as **score-of-record for persona/preference** and a **validation
  cross-check for language/code** (which keeps the κ rows in Table 13 meaningful).

- [x] **Instruction is a user turn, but described as a system instruction.** Fig 1 caption
  "A system instruction" → "An instruction"; Llama-anomaly "system prompt adherence" →
  "instruction adherence" (results.tex + discussion.tex).

---

## B. Substantive — counts and stale text

- [x] **Condition count: 16 vs 13.** Decision: **keep 16.** §2.2 now states the count explicitly
  — the two follow-up conditions are *three instructions* (classify + random-facts in two
  directions), so 13 main + 3 follow-up = 16. Kept "two follow-up conditions" in §3.5 / App E to
  preserve the 2×2 (engagement × diversity) framing; abstract + intro "16" unchanged.

- [x] **Orphaned "style case" conditions.** Removed the `.upper()/.lower()` sentence in App A and
  the "case- and length-based style conditions" parenthetical in App L — those conditions are not
  among the 13.

- [x] **"Five condition families" but four listed.** App L now says **four** (language, persona,
  preference, code-style).

---

## C. Minor numerical mismatches

- [x] **Table 2 task-based Avg IF ≠ Tables 10/11.** Corrected Table 2 to the data: Kimi 54→**53**,
  OLMo 22→**23**, Gemini 19→**20**, Qwen235 32→**33**, Qwen30 42→**41** — plus GPT-5.2 57→**56**
  and Gemma-3 27B 38→**39**, which were off by 1 the same way (not on the original list but same
  fix). Grand means (27% / 43%) unchanged.

- [x] **GPT-5.2-medium fixed-output:** now **53%** in both Table 5 and §3.6 (token-excluded recompute).

- [x] **GPT-5.2 non-reasoning task-based:** Table 5 now **56%** (matches text + Tables 10/11).

- [x] **Hermes-4 70B non-reasoning fixed-output:** §3.6 text 3% → **2%** (matches Table 2 / Table 5).

- [x] **Table 12 Mean row.** Recomputed on current data: mean Δ now **−4%** with columns 38→34
  (internally consistent); prose stats updated (Δ=−0.04, t(5)=−1.97, p=0.11).

- [x] **`preference aligned helpful` non-unanimous rate:** 48.5% → **50%** (matches Table 13's 0.50
  unanimous and the "50% still unanimous" line).

---

## D. Typos and wording

- [x] **Line 29:** mangled quotes around *porosity* → ``porosity''.
- [x] **Line 537:** "produce 1/3 sentences" → "1–3 sentences".
- [x] **Line 115:** "(AliBaba)" → "Alibaba".
- [x] **Term order:** intro + related "induction-instruction" → "instruction-induction".
- [~] **Model naming flips.** Fixed the LaTeX tables (follow-up + hint tables: "Claude 4.6 Opus/
  Sonnet" → "Claude Opus/Sonnet 4.6"). **Remaining:** heatmap / per-model **PNG** bar labels still
  read "Claude 4.6 Opus" (needs `DISPLAY_NAMES` change + plot regen). App B **API slugs**
  (`claude-opus-4.6` vs `claude-4.6-sonnet`) left as-is — they're real OpenRouter identifiers
  (the CHANGELOG deliberately kept them); confirm if you want them aligned.
- [x] **Line 164:** "smaller than other models tested" → "smaller than the largest models tested".
- [ ] **Random-facts naming order.** `random_geography_animals` has T=animals (pattern-first),
  opposite to the `language_ru_fr` target-first convention. Not changed — fixing means renaming
  data keys (risky); T/P are explicit in Table 8. Flag for author.
- [x] **Line 648:** traces intro reframed — Trace 1 = rationalized capitulation, Trace 2 = inferred
  incapacity, Trace 3 = pure deliberation–output dissociation (no longer lumps all three).
- [x] **Line 108:** missing period added at end of §2.2.


---

*Note: a few items (esp. diacritics and exact quote glyphs) could be PDF-extraction artifacts rather than errors in the source — verify against the LaTeX.*
