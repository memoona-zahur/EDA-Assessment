"""
Generate the support tickets dataset from the exact assessment spec.
Run this first to create tickets_raw.csv.

Do not modify this script — the hidden grading checklist depends on
every trainee's dataset having the same shape and the same problems.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=7)
n = 4000

tickets = pd.DataFrame({
    "ticket_id": np.arange(1, n + 1),
    "created_at": pd.date_range("2024-03-01", periods=n, freq="30min"),
    "agent_id": rng.integers(200, 260, size=n),
    "priority": rng.choice(["Low", "Medium", "High", "high"], size=n),
    "resolution_hours": rng.gamma(shape=2.0, scale=6.0, size=n).round(2),
    "channel": rng.choice(["Email", "Chat", "Phone", None], size=n, p=[0.35, 0.35, 0.25, 0.05]),
})

# Introduce the mess, on purpose — do not skip this part
tickets.loc[rng.choice(n, 120, replace=False), "agent_id"] = None
tickets.loc[rng.choice(n, 25, replace=False), "resolution_hours"] *= -1
tickets.loc[rng.choice(n, 15, replace=False), "resolution_hours"] = 999.0
tickets = pd.concat([tickets, tickets.sample(12, random_state=3)])

tickets.to_csv("tickets_raw.csv", index=False)
print(f"Saved tickets_raw.csv — shape: {tickets.shape}")

# Self-check
assert tickets.shape == (4012, 6)
assert list(tickets.columns) == ["ticket_id", "created_at", "agent_id", "priority", "resolution_hours", "channel"]
print("Setup confirmed — dataset matches spec.")
