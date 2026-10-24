"""Generate hourly contact volume for the Workforce & Demand Planner.

16 weeks of phone, chat and email arrivals for the fictional Aurora support centre,
ending at the same "now" as the ticket dataset. The patterns are the ones real
contact centres see: a daily curve that peaks late morning, busy Mondays, quiet
weekends, slow growth, Poisson noise, plus two events the planner should find:

  - a firmware update on 14 Sep 2026 that triggers a 3-day surge of contacts
  - a public holiday on 31 Aug 2026 (Malaysia's National Day) with low volume

Seeded, so re-running gives the same file.

    python scripts/generate_demand.py
"""
from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "demand.csv"
rng = np.random.default_rng(7)

END = datetime(2026, 10, 20, 0, 0)      # first hour NOT in the history
WEEKS = 16
START = END - timedelta(weeks=WEEKS)

# Peak arrivals per hour on an average weekday, before trend.
PEAK = {"phone": 110, "chat": 85, "email": 55}
# Hour-of-day shape (0..23), shared, with a channel-specific twist below.
DAY_CURVE = [0.06, 0.04, 0.03, 0.03, 0.04, 0.08, 0.18, 0.38, 0.66, 0.90, 1.00, 0.97,
             0.85, 0.88, 0.92, 0.86, 0.74, 0.60, 0.48, 0.40, 0.34, 0.26, 0.17, 0.10]
WEEKDAY = [1.15, 1.04, 1.00, 0.97, 0.93, 0.52, 0.38]   # Mon..Sun
GROWTH_PER_WEEK = 0.006
SURGE_START, SURGE_DAYS = datetime(2026, 9, 14), 3
SURGE_LIFT = {"phone": 0.85, "chat": 0.70, "email": 0.55}
HOLIDAY = datetime(2026, 8, 31)


def rate(channel: str, t: datetime) -> float:
    h = DAY_CURVE[t.hour]
    if channel == "email":            # email arrives more evenly through the day and night
        h = 0.35 + 0.65 * h
    if channel == "chat":             # chat stays busier in the evening
        h = h if t.hour < 17 else min(1.0, h * 1.25)
    weeks_in = (t - START).total_seconds() / (7 * 86400)
    r = PEAK[channel] * h * WEEKDAY[t.weekday()] * (1 + GROWTH_PER_WEEK) ** weeks_in
    days_into_surge = (t - SURGE_START).total_seconds() / 86400
    if 0 <= days_into_surge < SURGE_DAYS:
        r *= 1 + SURGE_LIFT[channel] * math.exp(-0.7 * days_into_surge)
    if t.date() == HOLIDAY.date():
        r *= 0.45
    return r


def main() -> None:
    rows = []
    t = START
    while t < END:
        for ch in PEAK:
            rows.append((t.strftime("%Y-%m-%dT%H:%M"), ch, int(rng.poisson(rate(ch, t)))))
        t += timedelta(hours=1)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hour", "channel", "contacts"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
