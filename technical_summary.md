# Technical Summary — Support Tickets EDA (Week 05 Friday)

*Standalone write-up for non-technical readers; the full analysis with code lives in `week5_friday_eda_assessment.ipynb`.*

## What the dataset contains

A log of **4,012 support tickets** opened between 1 March and mid-May 2024 at 30-minute intervals, recording six fields each: ticket number, creation timestamp, which agent handled it, its priority (Low/Medium/High), how many hours it took to resolve, and which channel it arrived through (Email, Chat, or Phone).

## What was wrong with it

The raw file carried six distinct data-quality problems, all measured before anything was touched:

| Problem | Size | Plain-language impact |
|---|---|---|
| Agent IDs missing | 121 rows (3%) | We can't tell who handled these tickets |
| Channel missing | 193 rows (4.8%) | Nearly 1-in-20 tickets have no recorded source |
| Exact duplicate rows | 12 | A few tickets counted twice |
| Negative resolution hours | 25 | Impossible values — a ticket can't take −10 hours |
| Placeholder "999 hours" | 15 | Fifteen tickets claiming ~6 weeks to resolve; clearly a "not recorded" stamp |
| Priority casing split | 982 rows labelled "high" vs "High" | High-priority volume under-counted by half in any report |

## What was done

Every problem was quantified *first*, locked into `findings.json`, then fixed with the least-destructive method that could be defended: duplicates dropped; casing merged; negative signs recovered (their magnitudes sat squarely inside the normal range — classic sign-flip typos); the fifteen 999s replaced with their priority group's typical value, each replacement logged by ticket ID; missing channels surfaced as an explicit "Unknown" group rather than quietly vanishing from charts; missing agent IDs left honestly blank since inventing staff assignments would be worse than admitting gaps. The original raw file was never modified — provable via an assertion cell in the notebook.

## Top findings

1. **The raw numbers overstated workload by ~29%.** The average resolution time looked like 15.6 hours before cleaning and is really about 12.0 hours — anyone reporting the uncleaned average to management inflated perceived effort by nearly a third.
2. **Priority made no measurable difference to speed.** All three priority bands resolve in roughly 11.8–12.2 hours on average, with confidence intervals that overlap heavily — the honest conclusion is that urgent labelling isn't buying faster fixes this quarter.
3. **Half of "High" priority was invisible** until the casing split was merged — dashboards built on the raw column were steering prioritization with wrong volumes.
4. **The process was stable over the period studied** — resolution times show no upward or downward drift across March–May 2024, so no hidden incident or improvement occurred.

## What this analysis could NOT do (honest limitations)

- The sign-flip recovery assumes negatives were only sign-corrupted; without the source system we can't verify row-by-row.
- The 15 imputed durations are estimates (priority medians), not real observations — flagged with ticket IDs in Section 3.5 of the notebook.
- The 120 missing agent IDs are permanently unrecoverable without CRM backfill.
- Findings counts describe the delivered raw file (duplicates included); the tiny divergence from record-level truth is quantified in Section 2.5.
- Conclusions cover a single 83-day window (March 1 – May 23, 2024); they should not be generalized beyond it without re-running the pipeline on more windows.
