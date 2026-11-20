# ADR-0009 — Synthetic dialect robustness probe

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner

## Context

The model is trained on Amazon MASSIVE `es-ES`, but the owner's users are in
Latin America. Research on 2026-11-19 found no open-licensed, redistributable
Spanish classification corpus with natural Latin-American dialect labels: the
largest candidate (MeLiSA) has a demo-only, non-redistributable licence, and
TASS/InterTASS publishes no clear licence. Reporting accuracy on a corpus we
cannot ship would make the metric unreproducible.

S9 still requires robustness evidence. Perturbations must be pure, deterministic
and cheap enough to run on every final evaluation, and every report, README and
model-card mention must say plainly that the dialect probe is synthetic.

## Decision

1. The dialect probe applies **lexicon substitutions** to real test utterances
   with three profiles: `mx` (Mexico), `ec` (Ecuador) and `ar` (Rioplatense
   Argentina/Uruguay). The tables live in `src/stc/domain/lexicon.py` and are
   shared pan-regional entries (`COMMON_LATAM`) plus profile-specific overlays.
2. Matching is **whole-word, case-insensitive and longest-key-first**, so
   `ahora mismo` is replaced as one unit and `que` never matches inside
   `quesito`. A replacement preserves a leading capital.
3. Every substitution is **gender- and number-preserving**. A structural test
   asserts that no replacement value is also a key, which makes `regionalize`
   idempotent and keeps a second pass from cascading.
4. The other perturbations of D11 (`no_accents`, `uppercase`, `typos`,
   `chat_abbrev`) are also pure domain functions. `typos` edits each selected
   word by one character; a substitution replaces D11's adjacent transposition
   because a transposition has Levenshtein distance 2 while S9 pins every edit
   to distance <= 1.
5. The evaluation subset is the set of utterances the perturbation actually
   changed. On that subset the report gives accuracy before/after, the delta
   with a paired bootstrap 95 % CI (B = 2000, seed 13) and calibrated mean
   confidence. Macro-F1 is not reported there: subsets are too small per class,
   and the report says so.
6. `scripts/lexicon_coverage.py` prints how many dev utterances each profile
   changes. It refuses `--split test` with exit code 6; the test split stays
   behind the audited final-evaluation path.

## Alternatives

- **Train/evaluate on a real LatAm corpus.** Rejected: no suitable
  redistributable corpus (licence evidence in `docs/DATA.md`).
- **LLM paraphrase or translation of the test set.** Rejected: non-deterministic,
  needs a local LLM, changes meaning beyond dialect and makes the metric
  unreproducible.
- **Character noise only.** Rejected: it cannot show lexical dialect drift,
  which is the failure mode users actually notice.
- **A hand-labelled LatAm test set.** Out of scope for v0.1.0 and would not be a
  sensitivity probe of the same utterances.

## Consequences

The probe can show that the model is sensitive to dialect lexicon, but it
cannot estimate real LatAm accuracy and must never be reported as such. The
lexicon is data, so extending it is cheap and does not touch the evaluator.
Word-sense homographs (for example a noun that is also a verb form) can produce
ungrammatical substitutions; the conservative tables and the no-value-is-a-key
test limit, but do not eliminate, that risk. Coverage is measured, not assumed.

## Evidence

- `uv run pytest tests/unit/domain/test_lexicon.py tests/unit/domain/test_perturbations.py -q`
  covers whole-word matching, capital preservation, idempotence, typo distance,
  determinism and the registry.
- `uv run python scripts/lexicon_coverage.py --split dev` on the pinned
  MASSIVE dev split printed `mx 99`, `ec 105`, `ar 387` (2,033 utterances).
- `uv run python scripts/lexicon_coverage.py --split test` exits 6 without
  reading the split.
- The final report writes the probe to `reports/final/robustness.json` with the
  note that macro-F1 is not reported on perturbation subsets.
