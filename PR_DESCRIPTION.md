# PR Description — Week 05 Friday EDA Assessment (paste into GitHub PR)

## What does this PR do?
Full EDA pipeline on a deliberately corrupted 4,012-row support-tickets dataset: every planted problem measured before fixing, cleaned with written justification per decision, visualized in six annotated charts, verified by 21 automated checks — plus the Part 2 theory quiz answered in `theory_quiz_answers.md`.

## What was done?

**Diagnosis (before any fix)**
- All six planted issues quantified on the raw file and locked into `findings.json` (`missing_agent_id: 121`, `missing_channel: 193`, `duplicate_rows: 12`, `negative_resolution_hours: 25`, `outlier_resolution_hours: 15`)
- Two unsolicited audits: duplicate-contamination analysis (does one duplicated record carry other issues? yes — quantified) and IQR-fence vs sentinel separation (protects hundreds of legitimate slow tickets from naive outlier filters)
- Scipy skewness/kurtosis quantify distribution shape before any plotting

**Cleaning (each fix with why + why-not-alternative)**
- Duplicates dropped, casing merged, sign-flips recovered via `abs()` after magnitude-plausibility evidence, sentinels imputed with priority-group medians (computed excluding the sentinels themselves; all 15 ticket IDs printed as audit trail), channel gaps surfaced as explicit `"Unknown"`, agent gaps honestly kept NaN
- §3.9 proof cell asserts the raw DataFrame remains byte-for-byte untouched
- §3.10 consolidated audit: cleaning ledger + column-completeness scorecard + "cost of dirty data" counterfactual table (~29% phantom workload quantified per issue)

**Visualization (all charts carry their key numbers on the saved PNG)**
- Required three: histogram (stats box: n/median/mean/std/max/skew), priority bar chart (**bootstrap 95% CIs from 2,000 seeded resamples + Kruskal–Wallis test** — an honest, statistically defended null result), date-vs-duration scatter (slope, Pearson r, rolling endpoints annotated)
- Bonus three: before/after cleaning overlay, misleading-vs-honest y-axis demonstration, channel mix with visible Unknown

**Verification**
- In-notebook assert battery stops execution if any claim breaks
- `test_friday_sample.py` (given): 3/3
- `test_own_verification.py`: **18 invented adversarial checks**, incl. PNG magic-byte validation (the given size-only check accepts renamed text files), casing-fix proven by counting (catches filter-instead-of-map), findings recomputed from raw AND locked to known values, statistical canary on the mean, raw-file immutability guard, schema-drift detector, requirements-pin enforcement → **21/21 total**

## Verification
- [x] All checks passing (21/21, embedded in final notebook cell too)
- [x] Tested end-to-end: fresh kernel execution via `jupyter nbconvert --execute` — 77 cells, zero errors; full pipeline re-runnable from `generate_data.py` → `build_notebook.py` → tests
- [x] Documentation updated: README, SELF_REVIEW.md (requirement-by-requirement PASS/FAIL with evidence), technical_summary.md, theory_quiz_answers.md

## Any issues or limitations?
Honest ones, documented in notebook §6: `abs()` assumes pure sign-flips without source-system proof; the 15 imputed durations are estimates (flagged by ticket ID); 120 missing agent IDs are unrecoverable; findings counts describe the delivered raw file (divergence quantified); conclusions cover two months only.
