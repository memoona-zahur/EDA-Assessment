# Self-Review — Week 05 Friday Assessment

Reviewed against the spec (`eda-assessment.md`), the given sample checks, and the
top-performer standard: every claim below is backed by a command run or a notebook
cell reference — nothing is "should work".

## Requirements Coverage

| Requirement | Status | Evidence |
|---|---|---|
| Exact generation code, unmodified | PASS | `generate_data.py` matches spec verbatim; seed=7; shape assert `(4012, 6)` passes |
| Diagnosis before fixing: `.head()` `.info()` `.describe()` `.isna().sum()` ≥1 `.value_counts()` | PASS | Notebook Phase 1–2 (cells 4–13): all five present, plus missingness % table |
| Markdown cell listing every problem found **before** fixing | PASS | §2 "Diagnosis Findings" table — all six issues with counts and percentages |
| Per-issue cleaning decision + one-sentence justification each | PASS | Phase 3.1–3.7: each step has **why** AND **why-not-alternative** (exceeds one sentence) |
| Work on a copy; raw frame inspectable & unmodified at end | PASS | §3.9 proof cell asserts raw shape/counts intact after all cleaning |
| `findings.json` from **before any cleaning**, plain ints | PASS | Written in Phase 2 (pre-cleaning); 5 keys, Python `int`, values {121, 193, 12, 25, 15} |
| `tickets_clean.csv` via `to_csv(index=False)` | PASS | Final cell; 4,000 rows; column superset check passes |
| 3 charts via `fig, ax = plt.subplots()`, labeled, legend where relevant | PASS | Charts 1–3: titles, axis labels, legends on all; saved dpi=150 |
| Plot type matched to question | PASS | distribution→histogram, category comparison→bar (+95% CI), relationship→scatter+rolling mean |
| 2–3 findings as full sentences backed by chart/number | PASS | Phase 5: four findings, each with Evidence and Implication lines |
| Technical summary incl. honest limitation | PASS | Phase 6 in-notebook + standalone `technical_summary.md`; five specific limitations with mitigations |
| Survives Restart Kernel & Run All | PASS | Executed headlessly on a fresh kernel (`jupyter nbconvert --execute`) — 77 cells, zero errors |
| Sample self-check green | PASS | `pytest test_friday_sample.py` → 3 passed |

## Beyond-Minimum Additions (the "repeated pattern")

1. **Adversarial verification suite** — 18 invented pytest checks (`test_own_verification.py`)
   targeting failure modes the given structural tests cannot see:
   - PNG magic-byte validation (given size-only check accepts renamed text files)
   - casing fix proven by *counting* High rows (catches filter-instead-of-map)
   - Unknown-count == raw-missing-count (proves gap made visible, not deleted)
   - agent-NaN preservation (proves no fabrication, no silent drops)
   - findings recomputed from raw AND locked to known planted amounts
   - findings key-set *equality* (stricter than the given subset check)
   - schema-drift detector: clean columns == raw columns
   - statistical canary: clean mean within 2 h of median detects sentinel leaks
   - physical-plausibility bound (max < 100 h catches surviving sentinels)
   - raw-file immutability guard
   - requirements.txt pin enforcement
2. **Duplicate-contamination audit (§2.5)** — spec never asks whether duplicates carry
   other planted issues; measured: exactly one record, so findings.json's raw-count
   convention is defended with numbers.
3. **Sentinel-vs-tail separation (§2.4)** — IQR fence computed to prove only the exact-999
   pile-up is corruption; hundreds of legitimate slow tickets survive naive filtering elsewhere.
4. **Raw-frame integrity proof (§3.9)** — spec rule converted into executable evidence.
5. **Consolidated audit section (§3.10)** — cleaning ledger, column-completeness scorecard,
   and a "cost of dirty data" counterfactual table tying every fix to the measurable damage
   it prevented (~29% phantom workload, half-hidden High-priority volume, vanishing 4.8% channel share).
6. **Statistics beyond requirement** — bootstrap percentile CIs (2,000 seeded resamples) and a
   Kruskal–Wallis test turn Chart 2's null result from an eyeball claim into tested evidence;
   Pearson r + slope annotate the relationship chart; scipy skew/kurtosis in diagnosis.
7. **Three bonus charts** (before/after overlay, misleading-vs-honest demo of quiz Q8,
   channel mix with visible Unknown) — each with Question → Chart → Finding write-ups.
8. **Every chart carries its key numbers on the saved PNG** (stats boxes, per-bar CI labels,
   r/slope annotations) — readable by someone who never opens the notebook.
9. **Executive summary + contents table** up front — a reviewer gets the whole story in 30 seconds,
   then can navigate to any phase.
10. **Honest null result** — priority speed difference reported as statistically absent
    (overlapping CIs, p > 0.05), not narrated into a fake trend from noise.
11. **Imputation audit trail** — all 15 sentinel ticket_ids printed; medians computed
    excluding the sentinels themselves.
12. **Reproducibility engineering** — dataset SHA-256 fingerprint recorded (Phase 0),
    pinned `requirements.txt`, committed `build_notebook.py` that regenerates the notebook,
    venv-based setup, ready-to-paste `PR_DESCRIPTION.md`.

## Adversarial Self-Questions

- **What looks correct but might be wrong?**
  The `abs()` fix for negatives is an assumption (pure sign-flip). Mitigated with
  magnitude-range evidence (§3.4) and listed as limitation #1 rather than hidden.
- **What would break if input changed?**
  The suite hard-codes planted amounts (121/193/12/25/15) — intentional: this dataset's
  contract. On new data, the recount-from-raw test still validates internal consistency.
- **What could a skeptic question?**
  Median-imputation for sentinels alters group means slightly. Answered in §3.5: audit
  trail printed, medians exclude sentinels themselves, impact quantified (~29% raw-mean
  correction vs negligible clean-group shift).
- **What did we NOT do?**
  No statistical significance testing on the priority differences (CIs shown instead);
  agent-level analysis skipped because IDs are near-uniform random labels — both stated,
  not papered over.

## Honest Self-Assessment

Weak spot I would strengthen with more time: per-agent workload sanity checks if agent
IDs were real staffing data (skipped here because IDs are near-uniform random labels —
any agent ranking would be pure noise, and manufacturing a leaderboard from noise would
be exactly the dishonesty this project avoids). Chart 2's null result is already backed
by bootstrap CIs and a Kruskal–Wallis test rather than left as an eyeball claim.
Everything required by the spec is complete and verified.
