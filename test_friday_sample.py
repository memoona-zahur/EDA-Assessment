# test_friday_sample.py — run with: pytest test_friday_sample.py
import json
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"ticket_id", "created_at", "agent_id", "priority", "resolution_hours", "channel"}
REQUIRED_FINDINGS_KEYS = {
    "missing_agent_id", "missing_channel", "duplicate_rows",
    "negative_resolution_hours", "outlier_resolution_hours",
}
REQUIRED_CHARTS = ["chart_distribution.png", "chart_category_comparison.png", "chart_relationship.png"]

def test_clean_csv_has_required_columns():
    df = pd.read_csv("tickets_clean.csv")
    assert REQUIRED_COLUMNS <= set(df.columns)

def test_findings_json_has_required_keys_as_ints():
    findings = json.loads(Path("findings.json").read_text())
    assert REQUIRED_FINDINGS_KEYS <= findings.keys()
    assert all(isinstance(findings[k], int) for k in REQUIRED_FINDINGS_KEYS)

def test_chart_files_exist_and_are_nonempty():
    for name in REQUIRED_CHARTS:
        p = Path(name)
        assert p.exists() and p.stat().st_size > 1000
