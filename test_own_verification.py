# test_own_verification.py — self-invented adversarial checks beyond the given self-check.
# Run with: pytest test_friday_sample.py test_own_verification.py
# Design principle: catch work that LOOKS correct but is wrong.
import json
from pathlib import Path

import pandas as pd

RAW = pd.read_csv("tickets_raw.csv")
CLEAN = pd.read_csv("tickets_clean.csv")
FINDINGS = json.loads(Path("findings.json").read_text())
CHARTS = ["chart_distribution.png", "chart_category_comparison.png", "chart_relationship.png"]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_clean_row_count_equals_raw_minus_duplicates():
    assert len(CLEAN) == len(RAW) - int(RAW.duplicated().sum()) == 4000


def test_ticket_id_unique_after_dedup():
    assert CLEAN["ticket_id"].is_unique


def test_no_duplicate_rows_survive():
    assert not CLEAN.duplicated().any()


def test_no_negative_or_sentinel_or_nan_resolution_hours():
    rh = CLEAN["resolution_hours"]
    assert (rh >= 0).all()
    assert not (rh == 999).any()
    assert rh.notna().all()


def test_priority_casing_actually_normalized():
    # Catches "fixed" via filtering instead of mapping — rows would be missing.
    assert set(CLEAN["priority"]) == {"Low", "Medium", "High"}
    high_total = int(RAW["priority"].str.lower().eq("high").sum()) - _dup_high()
    assert int((CLEAN["priority"] == "High").sum()) == high_total


def _dup_high():
    dups = RAW[RAW.duplicated(keep="first")]
    return int(dups["priority"].str.lower().eq("high").sum())


def test_channel_gap_made_visible_not_deleted():
    # The 193 unknown-channel tickets must exist as 'Unknown', not vanish.
    assert (CLEAN["channel"] == "Unknown").sum() == FINDINGS["missing_channel"]


def test_missing_agent_ids_preserved_as_nan_not_fabricated():
    expected_nan = FINDINGS["missing_agent_id"] - RAW.loc[RAW.duplicated(keep="first"), "agent_id"].isna().sum()
    assert int(CLEAN["agent_id"].isna().sum()) == expected_nan


def test_created_at_parseable_and_spans_expected_window():
    dates = pd.to_datetime(CLEAN["created_at"])
    assert dates.min().strftime("%Y-%m-%d") == "2024-03-01"
    assert dates.max() > pd.Timestamp("2024-04-01")


def test_findings_json_exactly_matches_recount_from_raw():
    recount = {
        "missing_agent_id": int(RAW["agent_id"].isna().sum()),
        "missing_channel": int(RAW["channel"].isna().sum()),
        "duplicate_rows": int(RAW.duplicated().sum()),
        "negative_resolution_hours": int((RAW["resolution_hours"] < 0).sum()),
        "outlier_resolution_hours": int((RAW["resolution_hours"] == 999).sum()),
    }
    assert FINDINGS == recount


def test_findings_values_are_the_known_planted_amounts():
    assert FINDINGS == {
        "missing_agent_id": 121,
        "missing_channel": 193,
        "duplicate_rows": 12,
        "negative_resolution_hours": 25,
        "outlier_resolution_hours": 15,
    }


def test_charts_are_real_pngs_not_renamed_text_files():
    # The given check accepts any >1000-byte file; this one demands the real format.
    for name in CHARTS:
        head = Path(name).read_bytes()[:8]
        assert head == PNG_MAGIC, f"{name} is not a valid PNG (head={head!r})"


def test_clean_mean_is_statistically_sane_canary():
    # A leftover sentinel or un-recovered sign error pushes the mean toward the
    # corrupted 15.56 h; cleaned data must sit near the robust median instead.
    mean, median = CLEAN["resolution_hours"].mean(), CLEAN["resolution_hours"].median()
    assert abs(mean - median) < 2.0 and 11.0 < mean < 13.0
