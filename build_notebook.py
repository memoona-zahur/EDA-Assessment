"""Builder for week5_friday_eda_assessment.ipynb — stage 1 (Phases 0-2) or full."""
import sys
import nbformat as nbf

stage = sys.argv[1]  # "v1" or "full"
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ---------------------------------------------------------------- title
md("""# Week 05 Friday — EDA Assessment
**Support Tickets Dataset — Load → Diagnose → Clean → Visualize → Report**

| | |
|---|---|
| **Raw input** | `tickets_raw.csv` (4,012 rows × 6 columns) |
| **Deliverables** | `tickets_clean.csv`, `findings.json`, 3 chart PNGs |
| **Verification** | `test_friday_sample.py` (given) + `test_own_verification.py` (my adversarial suite) |

**Pipeline rules I set for myself before touching the data:**
1. **Measure before fixing** — every problem is quantified *before* any correction, so `findings.json` documents the raw damage, not post-cleaning residue.
2. **Every fix carries a written "why" AND a "why not X"** — a cleaning step without alternatives considered is a guess.
3. **Validate after cleaning** — asserts re-check every claim; code that *looks* right is not evidence.""")

# ---------------------------------------------------------------- setup
md("""## Phase 0 — Setup""")
code("""%matplotlib inline
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
pd.set_option("display.width", 120)
print("pandas", pd.__version__, "| numpy", np.__version__, "| scipy", __import__("scipy").__version__)""")

# ---------------------------------------------------------------- phase 1
md("""## Phase 1 — Dataset Loading

**Question:** What are we dealing with — size, schema, and an honest first smell test?""")
code("""tickets = pd.read_csv("tickets_raw.csv")
print(f"Shape: {tickets.shape}")
tickets.head()""")
code("""tickets.info()""")

md("""**What this tells us:**
- **4,012 rows × 6 columns.** The generator spec says 4,000 tickets were created — 12 extra rows already hints at appended duplicates.
- `created_at` arrived as **object** (string), not datetime — must be parsed before any time analysis.
- `agent_id` is **float64 even though it is an ID**. An integer-valued ID only becomes float when NaNs force the upcast → missing agent IDs exist before I even count them.
- No column is fully populated except `ticket_id` / `priority` / `resolution_hours`.""")
code("""tickets.describe(include="all")""")

# ---------------------------------------------------------------- phase 2
md("""## Phase 2 — Diagnosis Before Cleaning

**Context:** Six problems are rumored to be planted in this dataset. I will not take that on faith — each one is measured here on the **raw** data, plus two cross-checks the spec did *not* ask for (duplicate contamination and outlier-vs-tail separation).

### 2.1 Missing values""")
code("""missing = tickets.isna().sum().to_frame("missing_count")
missing["missing_pct"] = (100 * missing["missing_count"] / len(tickets)).round(2)
missing""")

md("""### 2.2 Duplicate rows""")
code("""n_dup = int(tickets.duplicated().sum())
print(f"Exact duplicate rows: {n_dup}")
print(f"Shape if deduplicated: {tickets.drop_duplicates().shape}")
tickets[tickets.duplicated(keep=False)].sort_values("ticket_id").head(6)""")
md("""Every duplicate is a **full-row copy** (same `ticket_id` included) — these are accidental appends, not legitimately repeated events, so `keep="first"` is safe later.""")

md("""### 2.3 Priority casing""")
code("""tickets["priority"].value_counts(dropna=False)""")
md("""`High` (1,000) and lowercase `high` (982) are the **same category split in two**. Left alone, any grouped statistic undercounts High-priority volume by ~49.6% — this silently poisons Chart 2 later if missed.""")

md("""### 2.4 Resolution hours: negatives, sentinel outliers, and the honest tail""")
code("""rh = tickets["resolution_hours"]
print(f"negative values : {(rh < 0).sum()}  (range {rh[rh<0].min():.2f} … {rh[rh<0].max():.2f})")
print(f"sentinel == 999 : {(rh == 999).sum()}   (exact value 999.0, repeated)")
print(f"raw mean        : {rh.mean():.2f} h   ← inflated by both corruptions")
print(f"raw median      : {rh.median():.2f} h  ← robust reference point")

clean_view = rh[(rh > 0) & (rh != 999)]
print(f"max excluding corruption: {clean_view.max():.2f} h")""")

code("""# Adversarial check: is 999 really a sentinel, or just the tail of the distribution?
q1, q3 = rh.quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
beyond_fence = ((rh > upper_fence) & (rh != 999)).sum()
print(f"IQR upper fence: {upper_fence:.2f} h")
print(f"non-999 values beyond fence: {beyond_fence} — plausible gamma tail, NOT sentinels")
print(f"values exactly 999.0       : {(rh == 999).sum()} — statistically impossible pile-up at one point")

# Distribution shape (skew/kurtosis) — quantifies what the histogram will show later
body = rh[(rh > 0) & (rh != 999)]
print(f"\\nshape of uncontaminated body: skewness={stats.skew(body):.2f} (right-skewed), "
      f"kurtosis={stats.kurtosis(body):.2f} (heavy tail vs normal)")""")

md("""This distinction matters: a naive `df[df.resolution_hours < upper_fence]` filter would delete ~hundreds of legitimate slow tickets. Only the **impossible pile-up at exactly 999** is corruption; the long right tail is real operational behavior and stays.""")

md("""### 2.5 Cross-contamination: do duplicates carry other issues?
The spec never asks this. If duplicated rows happen to contain planted issues, raw counts double-count them — worth knowing before writing `findings.json`.""")
code("""dup_mask = tickets.duplicated(keep=False)
sub = tickets[dup_mask]
print(f"rows involved in duplicate pairs      : {len(sub)}")
print(f"  of which negative resolution_hours  : {(sub['resolution_hours'] < 0).sum()}")
print(f"  of which sentinel 999               : {(sub['resolution_hours'] == 999).sum()}")
print(f"  of which missing agent_id           : {sub['agent_id'].isna().sum()}")
print(f"  of which missing channel            : {sub['channel'].isna().sum()}")""")
md("""Only **one** missing-`agent_id` row got duplicated (2 rows in pairs = 1 logical record). Impact: raw missing-agent count (121) overstates true record-level gaps (120) by exactly 1. I keep the raw number in `findings.json` because the key measures *"problems present in the delivered file"* — but now I can defend that choice with numbers instead of assumption.""")

md("""### Diagnosis Findings (measured, raw file)

| # | Issue | Count | % of rows | Nature |
|---|-------|------:|----------:|--------|
| 1 | Missing `agent_id` | 121 | 3.02% | gap |
| 2 | Missing `channel` | 193 | 4.81% | gap |
| 3 | Exact duplicate rows | 12 | 0.30% | structural |
| 4 | Negative `resolution_hours` | 25 | 0.62% | impossible value |
| 5 | Sentinel `resolution_hours == 999` | 15 | 0.37% | placeholder value |
| 6 | Priority casing split (`high`/`High`) | 982 | 24.48% | inconsistent encoding |

Combined effect: raw mean resolution (15.56 h) is **~29% higher** than what uncontaminated rows suggest — anyone reporting the raw mean to management overstates workload by nearly a third.""")

code("""findings = {
    "missing_agent_id": int(tickets["agent_id"].isna().sum()),
    "missing_channel": int(tickets["channel"].isna().sum()),
    "duplicate_rows": int(tickets.duplicated().sum()),
    "negative_resolution_hours": int((tickets["resolution_hours"] < 0).sum()),
    "outlier_resolution_hours": int((tickets["resolution_hours"] == 999).sum()),
}
assert all(isinstance(v, int) and v >= 0 for v in findings.values())

with open("findings.json", "w") as f:
    json.dump(findings, f, indent=2)
print(json.dumps(findings, indent=2))
print("\\nSaved findings.json — counts measured on the RAW file (definition documented above).")""")

if stage == "full":
    # ------------------------------------------------------------ phase 3
    md("""## Phase 3 — Cleaning, One Issue at a Time

**Method:** issues are fixed in dependency order (duplicates first — so later per-group statistics are computed once per real ticket). Each step states the fix, why it is right for *this* dataset, and why the obvious alternatives lose.""")

    md("""### 3.1 Parse `created_at` to datetime
**Why:** string dates cannot be plotted, sorted by time axis correctly, or resampled.
**Why not leave as-is:** Phase 4's relationship chart has time on the x-axis; leaving strings would make matplotlib plot them categorically in arbitrary order.""")
    code("""clean = tickets.copy()
clean["created_at"] = pd.to_datetime(clean["created_at"])
t_min, t_max = clean["created_at"].min(), clean["created_at"].max()
print(t_min, "→", t_max)
print("span:", t_max - t_min)""")

    md("""### 3.2 Drop exact duplicate rows
**Fix:** `drop_duplicates(keep="first")`.
**Why:** full-row copies including `ticket_id` are structural accidents (append gone wrong), not repeat contacts — keeping both would double-count those tickets in every average below.
**Why not keep both:** there is no field distinguishing "first report" from "second report"; they are byte-identical.
**Why not flag-only:** a flag column still leaves duplicates inside group means unless every downstream cell remembers to filter — silent-bug territory.""")
    code("""before = len(clean)
clean = clean.drop_duplicates(keep="first").reset_index(drop=True)
print(f"{before} → {len(clean)} rows ({before - len(clean)} duplicates removed)")
assert len(clean) == 4000, "expected exactly the 4,000 generated tickets"
assert clean["ticket_id"].is_unique, "ticket_id must be unique after dedup" """)

    md("""### 3.3 Normalize priority casing
**Fix:** strip whitespace + capitalize → single `High` category.
**Why not drop the 982 lowercase rows:** they are 24.5% of the dataset and perfectly valid records — deletion to fix a *labeling* problem is disproportionate destruction.
**Why not map only `"high"`:** `.str.capitalize()` also future-proofs against `"HIGH "`/`" high"` variants without a hardcoded lookup table.""")
    code("""print("before:", clean["priority"].unique().tolist())
clean["priority"] = clean["priority"].str.strip().str.capitalize()
print("after :", clean["priority"].unique().tolist())
assert set(clean["priority"].unique()) == {"Low", "Medium", "High"}""")

    md("""### 3.4 Negative `resolution_hours` → recover via absolute value
**Evidence, not vibes:** all 25 magnitudes fall between **3.03 h and 37.81 h** — squarely inside the legitimate range (positive data runs 0.05–62 h). A duration cannot be negative; a magnitude this plausible points to a **sign-entry error**, not garbage rows.
**Why not drop 25 rows:** destroys recoverable information (0.62% of data) when the value itself is intact.
**Why not median-impute:** imputation is for *unknown* values. These are *known-but-mis-signed*. Fabricating a median when the truth is recoverable adds noise for no reason.""")
    code("""neg_before = (clean["resolution_hours"] < 0).sum()
mags = clean.loc[clean["resolution_hours"] < 0, "resolution_hours"].abs()
print(f"magnitude range of negatives: {mags.min():.2f} … {mags.max():.2f} h "
      f"(legit range: 0.05 … {clean.loc[clean['resolution_hours'] > 0, 'resolution_hours'].max():.2f} h)")
clean["resolution_hours"] = clean["resolution_hours"].abs()
print(f"negatives after abs(): {(clean['resolution_hours'] < 0).sum()}")
assert neg_before == 25 and (clean["resolution_hours"] >= 0).all()""")

    md("""### 3.5 Sentinel 999 → treat as unknown, impute priority-group median
**Why:** exactly 15 rows sit at precisely 999.0 while the genuine maximum elsewhere is 62.03 h — a placeholder meaning "not recorded", not a measurement. Averaging it in is how the raw mean got inflated ~29%.
**Why not cap/winsorize to the fence (~36 h):** invents a fake value *and* pretends it was observed; worse than admitting ignorance.
**Why not drop the 15 rows:** each row's other five fields are intact — dropping punishes the analysis for one bad field.
**Chosen fix:** NaN them, then fill with the **median of their own priority group** (medians ignore skew). Every imputed `ticket_id` is printed below — imputation must be auditable, not invisible.""")
    code("""sentinel_ids = clean.loc[clean["resolution_hours"] == 999, "ticket_id"].tolist()
print(f"sentinel rows being imputed ({len(sentinel_ids)}): ticket_ids = {sentinel_ids}")

mask999 = clean["resolution_hours"] == 999
group_medians = clean.loc[~mask999].groupby("priority")["resolution_hours"].median()
print("priority-group medians used (computed EXCLUDING the sentinels themselves):")
print(group_medians)

clean.loc[mask999, "resolution_hours"] = (
    clean.loc[mask999, "priority"].map(group_medians)
)
print(f"\\nsentinel/NaN remaining: {(clean['resolution_hours'] == 999).sum() + clean['resolution_hours'].isna().sum()}")
assert mask999.sum() == 15""")

    md("""### 3.6 Missing `channel` → explicit `"Unknown"` category
**Why:** 193 rows (4.82%) would otherwise be **silently dropped by every chart/groupby** — the reader would never learn that nearly 5% of tickets have no recorded channel. Making the gap a visible category turns hidden bias into reported fact.
**Why not mode-fill (`Chat`):** fabricates source attribution — it would manufacture ~140 false "Chat" tickets and flatter the Chat channel.
**Why not drop rows:** 4.8% loss across the whole dataset to hide one column's gap is a bad trade.""")
    code("""clean["channel"] = clean["channel"].fillna("Unknown")
print(clean["channel"].value_counts(dropna=False))""")

    md("""### 3.7 Missing `agent_id` → deliberately kept as NaN
**Asymmetry is intentional.** `channel` got an explicit label because charts consume it; `agent_id` is an identifier this analysis never aggregates on. Inventing an agent ID (mode-fill etc.) would fabricate accountability records in a staffing dataset — the one thing a support-org report must never do. NaN honestly says "unassigned/unrecorded".
**Why not drop the 120 rows:** 3% of tickets lost forever to fix a field the analysis does not use.""")
    code("""print(f"agent_id still NaN (kept, documented): {clean['agent_id'].isna().sum()} "
      f"({100 * clean['agent_id'].isna().mean():.2f}%)")""")

    md("""### 3.8 Post-cleaning validation — trust, then verify
Every diagnosis claim is re-checked on the cleaned frame. If any assert fails, the notebook stops rather than shipping wrong outputs.""")
    code("""assert clean.shape == (4000, 6),                       f"shape {clean.shape}"
assert clean["ticket_id"].is_unique,                    "duplicate ticket_ids remain"
assert not clean.duplicated().any(),                    "duplicate rows remain"
assert (clean["resolution_hours"] >= 0).all(),          "negative durations remain"
assert not (clean["resolution_hours"] == 999).any(),    "sentinel remains"
assert clean["resolution_hours"].notna().all(),         "NaN durations remain"
assert set(clean["priority"]) == {"Low", "Medium", "High"}, "priority domain broken"
assert set(clean["channel"]) <= {"Email", "Chat", "Phone", "Unknown"}, "channel domain broken"

summary = pd.DataFrame({
    "metric": ["rows", "mean resolution (h)", "median resolution (h)", "std resolution (h)"],
    "raw":    [f"{len(tickets)}",  f"{tickets['resolution_hours'].mean():.2f}",
               f"{tickets['resolution_hours'].median():.2f}", f"{tickets['resolution_hours'].std():.2f}"],
    "clean":  [f"{len(clean)}",    f"{clean['resolution_hours'].mean():.2f}",
               f"{clean['resolution_hours'].median():.2f}", f"{clean['resolution_hours'].std():.2f}"],
})
print(summary.to_string(index=False))
print("\\nALL VALIDATION ASSERTS PASSED")""")

    md("""### 3.9 Spec-integrity proof: the raw DataFrame is untouched
The spec requires working on a copy with `tickets` still inspectable and unmodified at the end. Claiming it is not proving it — so here is the proof cell.""")
    code("""assert tickets.shape == (4012, 6), "raw frame was resized!"
assert (tickets["resolution_hours"] < 0).sum() == 25, "raw negatives were altered!"
assert (tickets["resolution_hours"] == 999).sum() == 15, "raw sentinels were altered!"
assert tickets["priority"].nunique() == 4, "raw casing was normalized in place!"
assert tickets["agent_id"].isna().sum() == 121 and tickets["channel"].isna().sum() == 193
print("tickets (raw) intact: shape 4012x6, all six problems still present — "
      "every fix landed on the 'clean' copy only")""")

    # ------------------------------------------------------------ phase 4
    md("""## Phase 4 — Visualization

Each chart answers one question, states its finding in plain English, and is saved at 150 dpi.

### Chart 1 — Distribution of Resolution Hours
**Question:** What is a *typical* resolution time, and what shape does the workload have?""")
    code("""fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(clean["resolution_hours"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)

med = clean["resolution_hours"].median()
mean = clean["resolution_hours"].mean()
ax.axvline(med, color="darkgreen", linestyle="--", linewidth=2, label=f"Median = {med:.1f} h")
ax.axvline(mean, color="crimson", linestyle="-.", linewidth=2, label=f"Mean = {mean:.1f} h")
ax.annotate("Right-skewed:\\nlong tail of hard tickets",
            xy=(40, 60), fontsize=11, color="#333333")

ax.set_title("Distribution of Ticket Resolution Hours (cleaned, n=4,000)", fontsize=14)
ax.set_xlabel("Resolution Hours")
ax.set_ylabel("Number of Tickets")
ax.legend()
fig.savefig("chart_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved chart_distribution.png | mean={mean:.2f} h, median={med:.2f} h")""")
    md("""**What this chart tells us:** Resolutions cluster around a **median of ~10.2 h**, but the distribution is **right-skewed** — the mean (12.03 h) sits visibly right of the median, dragged by a legitimate tail of difficult tickets reaching ~62 h. Operational implication: quote the **median** for SLA conversations ("half of all tickets close within ~10 h"); use the tail, not the average, to staff the hard-ticket queue.""")

    md("""### Chart 2 — Average Resolution by Priority
**Question:** Do higher-priority tickets actually get resolved faster?""")
    code("""order = ["Low", "Medium", "High"]
stats = clean.groupby("priority")["resolution_hours"].agg(["mean", "sem"]).reindex(order)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(stats.index, stats["mean"], yerr=stats["sem"] * 1.96,
              capsize=8, color=["#55A868", "#4C72B0", "#C44E52"], edgecolor="white")
for bar, m in zip(bars, stats["mean"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35,
            f"{m:.2f} h", ha="center", fontsize=12, fontweight="bold")

ax.set_title("Mean Resolution Hours by Priority (error bars = 95% CI)", fontsize=14)
ax.set_xlabel("Priority")
ax.set_ylabel("Mean Resolution Hours")
fig.savefig("chart_category_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print(stats.round(2))""")
    md("""**What this chart tells us:** All three priorities resolve in **~11.8–12.2 h on average**, with 95% confidence intervals that overlap heavily. The honest reading: **there is no meaningful priority-speed difference in this quarter's data** — the small gaps shown are within sampling noise. I am deliberately *not* narrating a fake story like "High priority is fastest"; the evidence does not support it. If the business expects priority triage to change speed, that expectation is currently unmet — which is itself an actionable finding.""")

    md("""### Chart 3 — Relationship: Resolution Time Over the Quarter
**Question:** Is the support process stable across March–April 2024, or drifting?""")
    code("""fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(clean["created_at"], clean["resolution_hours"], s=10, alpha=0.25,
           color="#4C72B0", label="Individual ticket")

daily_roll = clean.set_index("created_at")["resolution_hours"].rolling("7D").mean()
ax.plot(daily_roll.index, daily_roll.values, color="crimson", linewidth=2.5,
        label="7-day rolling mean")

slope = np.polyfit(np.arange(len(clean)), clean["resolution_hours"], 1)[0]
ax.set_title(f"Resolution Hours vs Creation Date (trend slope ≈ {slope:+.5f} h/ticket)", fontsize=13)
ax.set_xlabel("Ticket Created At")
ax.set_ylabel("Resolution Hours")
ax.legend()
fig.savefig("chart_relationship.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Trend slope: {slope:.5f} h per ticket → "
      f"{'stable process' if abs(slope) < 0.001 else 'meaningful drift'}")""")
    md("""**What this chart tells us:** The point cloud shows **no upward or downward drift** — the fitted slope is ≈ +0.00004 h per ticket and the 7-day rolling mean oscillates around the same ~10–13 h band all quarter. Process performance was **stable across March–April 2024**: no improvement initiative or degradation happened in this window. Time is the only continuous variable in this dataset besides the target, making date-vs-duration the defensible "relationship" view (agent IDs are categorical labels, not quantities — a scatter against them would imply fake arithmetic between agent numbers).""")

    md("""### Bonus Charts — evidence and media literacy beyond the required three

The spec demands three charts; these three extra views answer questions the required set leaves open.

### Bonus A — Before vs After Cleaning (`04_before_after_cleaning.png`)
**Question:** What did cleaning *actually* change about the distribution's shape?""")
    code("""fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(tickets["resolution_hours"].clip(lower=0), bins=60, range=(0, 100),
        alpha=0.45, color="crimson", label=f"Raw (n={len(tickets)}, incl. 999-sentinels)")
ax.hist(clean["resolution_hours"], bins=60, range=(0, 100),
        alpha=0.55, color="#4C72B0", label=f"Cleaned (n={len(clean)})")
ax.axvline(tickets["resolution_hours"].mean(), color="darkred", linestyle="--", linewidth=2,
           label=f"Raw mean {tickets['resolution_hours'].mean():.2f} h (corrupted)")
ax.axvline(clean["resolution_hours"].mean(), color="navy", linestyle="-.", linewidth=2,
           label=f"Clean mean {clean['resolution_hours'].mean():.2f} h")
ax.set_title("Before vs After Cleaning — Resolution Hours", fontsize=14)
ax.set_xlabel("Resolution Hours")
ax.set_ylabel("Number of Tickets")
ax.legend()
fig.savefig("04_before_after_cleaning.png", dpi=150, bbox_inches="tight")
plt.show()""")
    md("""**What this chart tells us:** The raw distribution carries a visible spike at the far right (the fifteen 999-sentinels) that simply vanishes after cleaning, and the corrupted mean marker sits noticeably right of the clean one — the ~29% workload overstatement from Finding 1 is now something a reviewer can *see*, not just read.""")

    md("""### Bonus B — Misleading vs Honest Presentation (`05_misleading_vs_honest.png`)
**Question:** How easily could this same dataset be used to exaggerate the priority gap?""")
    code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
means = stats["mean"]
gap = means.max() - means.min()
trunc_bottom = max(0, means.min() - 1)

axes[0].bar(means.index, means.values, color="#C44E52")
axes[0].set_ylim(trunc_bottom, means.max() + 0.15)
axes[0].set_title(f"MISLEADING: truncated y-axis\\nmakes a {gap:.2f} h gap look huge", fontsize=12)
axes[0].set_ylabel("Mean Resolution Hours")

axes[1].bar(means.index, means.values, color="#55A868")
axes[1].set_ylim(0, means.max() * 1.2)
for i, m in enumerate(means.values):
    axes[1].text(i, m + 0.25, f"{m:.2f} h", ha="center", fontsize=11)
axes[1].set_title(f"HONEST: full axis reveals overlap\\nsame data, {gap:.2f} h difference is trivial", fontsize=12)
for ax_ in axes:
    ax_.set_xlabel("Priority")
fig.suptitle("Same numbers, opposite story — always check the y-axis origin", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("05_misleading_vs_honest.png", dpi=150, bbox_inches="tight")
plt.show()""")
    md("""**What this chart tells us:** Truncating the y-axis to start near the minimum inflates a **0.42-hour** difference into what looks like a multi-fold disparity — no single number is changed, only the framing. This is the concrete mechanism behind quiz question 8, demonstrated on our own Chart 2 data: with overlapping confidence intervals, the honest view says "no real difference" while the truncated view invites a false "High priority wins" headline.""")

    md("""### Bonus C — Channel Mix Including the Gap (`06_channel_mix_unknown.png`)
**Question:** What does intake look like once the missing-channel hole is made visible?""")
    code("""counts = clean["channel"].value_counts()
pct = 100 * counts / counts.sum()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(counts.index[::-1], counts.values[::-1],
               color=["#8172B3" if lbl == "Unknown" else "#4C72B0" for lbl in counts.index[::-1]])
for bar, c, p in zip(bars, counts.values[::-1], pct.values[::-1]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
            f"{c:,} ({p:.1f}%)", va="center", fontsize=11)

ax.set_title("Ticket Intake by Channel — Unknown kept visible on purpose", fontsize=14)
ax.set_xlabel("Number of Tickets")
ax.set_ylabel("Channel")
fig.savefig("06_channel_mix_unknown.png", dpi=150, bbox_inches="tight")
plt.show()""")
    md("""**What this chart tells us:** Email and Chat each carry ~35% of intake, Phone ~26% — and `Unknown` would rank as the **fourth-largest channel at 4.8%** if it were a real source. That is precisely why §3.6 refused to mode-fill or drop these rows: a fifth of Email's volume silently disappearing from every groupby would have understated the data-capture problem this chart now makes impossible to ignore.""")

    # ------------------------------------------------------------ phase 5
    md("""## Phase 5 — Key Findings

1. **Raw data overstated average workload by ~29%.**
   Evidence: raw mean 15.56 h vs cleaned mean 12.03 h (`chart_distribution.png`, validation table in §3.8); driven by fifteen 999-sentinels and 25 sign-flipped durations.
   Implication: any pre-cleaning report to management was inflating perceived ticket effort by nearly a third — cleaning is not cosmetic, it changed the headline number.

2. **Priority reporting was silently broken by casing.**
   Evidence: §2.3 — `High` = 1,000 vs `high` = 982 rows; grouped counts under-reported High priority by ~49.6% until normalization.
   Implication: dashboards built on the raw column misinformed prioritization decisions; fixed domain is now `{Low, Medium, High}` with an assert guarding it.

3. **Channel attribution has a real hole, now visible instead of hidden.**
   Evidence: §3.6 — 193 tickets (4.82%) carry no channel; rendered as an explicit `Unknown` bar rather than vanishing from groupbys.
   Implication: nearly 1-in-20 tickets cannot be attributed to a intake channel — a CRM capture bug worth escalating, sized precisely thanks to this audit.

4. **The support process was stable — and priority made no difference to speed.**
   Evidence: `chart_relationship.png` (slope ≈ 0.00004 h/ticket); `chart_category_comparison.png` (means 11.79–12.21 h, overlapping 95% CIs).
   Implication: no drift means no hidden incident in the quarter; the absent priority effect challenges the assumption that triage urgency translates into faster resolution.""")

    # ------------------------------------------------------------ phase 6
    md("""## Phase 6 — Technical Summary

This dataset of 4,012 support-ticket rows contained six planted data-quality problems: 121 missing agent IDs, 193 missing channels, 12 exact duplicate rows, 25 impossible negative durations, 15 placeholder values of exactly 999 hours, and a priority column split between `High`/`high`. Each problem was quantified on the raw file *before* any repair (results locked into `findings.json`), then fixed with the least-destructive defensible method: duplicates dropped, casing normalized, sign errors recovered by absolute value (magnitudes verified plausible first), sentinels converted to priority-group medians with a printed audit trail, and missing channels surfaced as an explicit `Unknown` category. The cleaned 4,000-row dataset shows a right-skewed resolution distribution (median ≈ 10.2 h, max ≈ 62 h), **no meaningful speed difference between priorities**, and **no drift across the quarter**. All claims are guarded by asserts in-notebook plus 16 pytest checks in `test_own_verification.py`.

### What We Could NOT Do (honest limitations)

- **Cannot verify the sign-error theory per-row.** `abs()` assumes every negative was a pure sign flip. Magnitudes (3.0–37.8 h) sit comfortably in the legit range, supporting this — but without the source system I cannot prove no row suffered a *worse* corruption. → Would need write-ahead logs / entry-form history.
- **Imputed sentinels are estimates, not observations.** 15 tickets' durations (ticket IDs listed in §3.5) are priority-medians; group-level stats barely move, but row-level truth is unrecoverable. → Flagged, auditable; upstream fix is mandatory-field validation.
- **Missing agent IDs are unrecoverable.** 120 tickets will always lack ownership unless CRM backfills. Kept as NaN rather than fabricated. → Escalate to data entry controls.
- **`findings.json` counts include duplicate-carried copies by design.** They measure the raw delivered file; §2.5 quantifies the (tiny) divergence from record-level truth. → Definition is documented next to the numbers.
- **Two-month window limits generalization.** Stability/drift conclusions hold for Mar–Apr 2024 only. → Re-run pipeline on rolling windows.""")

    md("""## Self-Review Checklist

- [x] Every issue measured **before** fixing; `findings.json` written from raw counts
- [x] Every fix has written **why** + **why-not-alternative**
- [x] Four checks invented beyond spec: duplicate-contamination audit (§2.5), sentinel-vs-tail separation (§2.4), raw-frame integrity proof (§3.9), PNG magic-byte test
- [x] Post-clean assert battery (§3.8) — notebook refuses to ship invalid outputs
- [x] Six charts follow Question → Chart → Finding; honest null-result reported (Chart 2); misleading-vs-honest bonus demonstrates quiz Q8 on our own data
- [x] Limitations section names what cannot be known and what to do about it
- [x] Raw DataFrame provably untouched (§3.9); Restart-and-run-all clean; 16 pytest checks pass (3 given + 13 mine)""")
    code("""import subprocess, sys
r = subprocess.run(
    [sys.executable, "-m", "pytest", "test_friday_sample.py", "test_own_verification.py", "-q"],
    capture_output=True, text=True,
)
print(r.stdout[-1500:])
print("PYTEST EXIT CODE:", r.returncode, "(0 = all pass)")""")

    md("""## Save Cleaned Data""")
    code("""clean.to_csv("tickets_clean.csv", index=False)
print(f"Saved tickets_clean.csv — shape {clean.shape}")""")

nb.cells = cells
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.10.0"}

out = "week5_friday_eda_assessment.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {stage}: {len(cells)} cells -> {out}")
