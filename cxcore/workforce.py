"""Workforce & demand planning: forecast contacts, staff them with Erlang C, price the plan.

The pieces, in the order a planner uses them:

  1. `forecast()`     - next weeks of hourly contacts per channel: a robust hour-of-week
                        profile (median of recent weeks, so one bad week can't bend it)
                        times a weekly level that grows at the median recent growth rate.
  2. `backtest()`     - hides the last weeks, forecasts them, and compares the error with
                        a seasonal-naive baseline ("same hour last week"). A forecast that
                        can't beat that baseline isn't worth using.
  3. `anomalies()`    - days whose volume is far from the same weekday in nearby weeks
                        (robust z-score), e.g. a firmware-update surge or a public holiday.
  4. `staff_plan()`   - agents needed per hour: Erlang C for phone and chat to hit a service
                        level target, a workload model for email, then shrinkage and cost.
  5. `brief()`        - an AI copilot that writes the plan up for managers from the computed
                        facts only, with a template fallback when no LLM is configured.

The maths is deterministic and testable; the LLM only explains numbers it is given.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import DATA_DIR
from .llm import LLM

CHANNELS = ["phone", "chat", "email"]
HOURS_PER_WEEK = 168
BAND_K = 1.45   # 80th percentile of |error| / sqrt(forecast) on a calibration window


# ---- data -----------------------------------------------------------------------------
def load_demand() -> pd.DataFrame:
    """Hourly contacts, one column per channel, indexed by hour."""
    df = pd.read_csv(DATA_DIR / "demand.csv", parse_dates=["hour"])
    wide = df.pivot(index="hour", columns="channel", values="contacts")[CHANNELS]
    return wide.asfreq("h").fillna(0)


# ---- 1. forecast ----------------------------------------------------------------------
def _weekly_blocks(series: pd.Series) -> np.ndarray:
    """Last whole weeks as a (weeks, 168) array, aligned so each row ends where the data ends."""
    n = len(series) // HOURS_PER_WEEK
    return series.to_numpy()[-n * HOURS_PER_WEEK:].reshape(n, HOURS_PER_WEEK)


def forecast(hist: pd.DataFrame, weeks: int = 1, profile_weeks: int = 6, level_weeks: int = 4,
             growth_weeks: int = 12) -> pd.DataFrame:
    """Hourly forecast for `weeks` weeks after the history, with an 80% band per channel."""
    start = hist.index[-1] + pd.Timedelta(hours=1)
    idx = pd.date_range(start, periods=weeks * HOURS_PER_WEEK, freq="h")
    out = {}
    for ch in CHANNELS:
        blocks = _weekly_blocks(hist[ch])
        totals = blocks.sum(axis=1)
        shares = blocks[-profile_weeks:] / totals[-profile_weeks:, None]
        profile = np.median(shares, axis=0)
        profile /= profile.sum()
        recent = np.log(totals[-growth_weeks:])
        growth = float(np.median(np.diff(recent)))            # robust to one surge week
        level = float(np.median(totals[-level_weeks:]))       # centred ~2 weeks before the end
        # The history's last row ends at a week boundary, so the profile lines up with `idx`.
        weekly = [level * math.exp(growth * (k + 1 + level_weeks / 2 - 0.5)) for k in range(weeks)]
        mean = np.concatenate([w * profile for w in weekly])
        # 80% band: hourly counts are close to Poisson, so the error scales with sqrt(mean).
        # BAND_K was calibrated on an earlier 2-week window, not the backtest weeks it is scored on.
        half = BAND_K * np.sqrt(np.maximum(mean, 1))
        out[ch] = mean
        out[f"{ch}_lo"] = np.clip(mean - half, 0, None)
        out[f"{ch}_hi"] = mean + half
    return pd.DataFrame(out, index=idx)


# ---- 2. backtest ----------------------------------------------------------------------
def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Weighted absolute percentage error: total |error| / total volume."""
    return float(np.abs(actual - pred).sum() / max(actual.sum(), 1))


def backtest(hist: pd.DataFrame, holdout_weeks: int = 2) -> dict:
    cut = len(hist) - holdout_weeks * HOURS_PER_WEEK
    train, test = hist.iloc[:cut], hist.iloc[cut:]
    fc = forecast(train, weeks=holdout_weeks)
    rows = []
    for ch in CHANNELS:
        naive = np.tile(train[ch].to_numpy()[-HOURS_PER_WEEK:], holdout_weeks)
        daily_actual = test[ch].resample("D").sum().to_numpy()
        daily_fc = pd.Series(fc[ch].to_numpy(), index=test.index).resample("D").sum().to_numpy()
        inside = ((test[ch].to_numpy() >= fc[f"{ch}_lo"].to_numpy()) & (test[ch].to_numpy() <= fc[f"{ch}_hi"].to_numpy())).mean()
        rows.append({"channel": ch,
                     "model_wape_hourly": wape(test[ch].to_numpy(), fc[ch].to_numpy()),
                     "naive_wape_hourly": wape(test[ch].to_numpy(), naive),
                     "model_wape_daily": wape(daily_actual, daily_fc),
                     "band_coverage": float(inside)})
    table = pd.DataFrame(rows)
    return {"table": table, "holdout_weeks": holdout_weeks,
            "model_wape": float(table.model_wape_hourly.mean()),
            "naive_wape": float(table.naive_wape_hourly.mean()),
            "forecast": fc, "actual": test}


# ---- 3. anomalies ---------------------------------------------------------------------
def anomalies(hist: pd.DataFrame, z: float = 3.5, min_change: float = 0.15, window_weeks: int = 3) -> pd.DataFrame:
    """Days far from the same weekday in the surrounding weeks.

    Both tests must pass: a robust z-score (so normal noise is ignored) and a minimum size
    of change (so a very steady weekday doesn't flag a 3% wobble as an event).
    """
    daily = hist.sum(axis=1).resample("D").sum()
    rows = []
    for day, value in daily.items():
        peers = [daily.get(day + pd.Timedelta(weeks=k)) for k in range(-window_weeks, window_weeks + 1) if k]
        peers = np.array([p for p in peers if p is not None and not np.isnan(p)])
        if len(peers) < 3:
            continue
        med = float(np.median(peers))
        mad = max(float(np.median(np.abs(peers - med))) * 1.4826, 0.02 * med, 1.0)
        score = (value - med) / mad
        if abs(score) >= z and abs(value / med - 1) >= min_change:
            by_ch = hist.loc[str(day.date())].sum()
            rows.append({"date": day.date(), "contacts": int(value), "expected": int(med),
                         "change": value / med - 1, "z": round(score, 1),
                         "phone": int(by_ch.phone), "chat": int(by_ch.chat), "email": int(by_ch.email)})
    return pd.DataFrame(rows)


# ---- 4. staffing ----------------------------------------------------------------------
def erlang_c(traffic: float, agents: int) -> float:
    """Probability a contact has to wait (Erlang C), via the stable Erlang B recursion."""
    if agents <= traffic:
        return 1.0
    b = 1.0
    for n in range(1, agents + 1):
        b = traffic * b / (n + traffic * b)
    return agents * b / (agents - traffic * (1 - b))


def service_level(arrivals_per_hour: float, aht_s: float, agents: int, target_s: float) -> float:
    """Share of contacts answered within target_s seconds."""
    if arrivals_per_hour <= 0:
        return 1.0
    traffic = arrivals_per_hour * aht_s / 3600
    if agents <= traffic:
        return 0.0
    return 1 - erlang_c(traffic, agents) * math.exp(-(agents - traffic) * target_s / aht_s)


def agents_needed(arrivals_per_hour: float, aht_s: float, sl_target: float, target_s: float,
                  max_occupancy: float = 0.88) -> int:
    """Fewest agents that hit the service level without running people above max occupancy."""
    if arrivals_per_hour <= 0:
        return 0
    traffic = arrivals_per_hour * aht_s / 3600
    n = max(1, math.ceil(traffic / max_occupancy))
    while service_level(arrivals_per_hour, aht_s, n, target_s) < sl_target:
        n += 1
    return n


@dataclass
class Scenario:
    volume_change: float = 0.0      # +0.10 = 10% more contacts
    aht_change: float = 0.0         # -0.10 = handle time 10% shorter
    ai_deflection: float = 0.0      # share of chat + email the AI assistant resolves
    sl_target: float = 0.80         # 80% answered within the target time
    phone_target_s: int = 20
    chat_target_s: int = 60
    phone_aht_s: int = 420
    chat_aht_s: int = 540
    chat_concurrency: int = 2       # chats an agent handles at once
    email_aht_s: int = 360
    email_occupancy: float = 0.85
    email_hours: tuple = (8, 20)    # emails are worked in business hours, within their 24h SLA
    shrinkage: float = 0.30         # breaks, training, leave, meetings
    cost_per_hour: float = 15.0     # loaded cost per paid agent hour
    paid_hours_per_fte: float = 40.0


def adjusted_volume(fc: pd.DataFrame, s: Scenario) -> pd.DataFrame:
    v = fc[CHANNELS].copy() * (1 + s.volume_change)
    v["chat"] *= 1 - s.ai_deflection
    v["email"] *= 1 - s.ai_deflection
    return v


def staff_plan(fc: pd.DataFrame, s: Scenario = Scenario()) -> dict:
    """Agents per hour per channel, then paid hours, FTE and cost for the forecast window."""
    vol = adjusted_volume(fc, s)
    k = 1 + s.aht_change
    phone_aht, chat_aht, email_aht = s.phone_aht_s * k, s.chat_aht_s * k, s.email_aht_s * k
    plan = pd.DataFrame(index=vol.index)
    plan["phone"] = [agents_needed(v, phone_aht, s.sl_target, s.phone_target_s) for v in vol.phone]
    # Concurrency: an agent on 2 chats behaves like one server with half the handle time.
    plan["chat"] = [agents_needed(v, chat_aht / s.chat_concurrency, s.sl_target, s.chat_target_s) for v in vol.chat]
    # Email: total daily workload spread across business hours.
    open_h, close_h = s.email_hours
    day_work_h = (vol.email * email_aht / 3600).resample("D").sum()
    per_hour = (day_work_h / ((close_h - open_h) * s.email_occupancy)).reindex(vol.index, method="ffill").fillna(0)
    in_hours = (vol.index.hour >= open_h) & (vol.index.hour < close_h)
    plan["email"] = np.where(in_hours, np.ceil(per_hour), 0).astype(int)
    plan["on_floor"] = plan[CHANNELS].sum(axis=1)
    plan["scheduled"] = np.ceil(plan.on_floor / (1 - s.shrinkage)).astype(int)
    hours = int(plan.scheduled.sum())
    weeks = len(plan) / HOURS_PER_WEEK
    return {"plan": plan, "volume": vol, "paid_hours": hours,
            "fte": hours / weeks / s.paid_hours_per_fte,
            "weekly_cost": hours / weeks * s.cost_per_hour,
            "peak": plan.scheduled.idxmax(), "peak_agents": int(plan.scheduled.max())}


def flat_roster_check(fc: pd.DataFrame, s: Scenario = Scenario()) -> dict:
    """Same phone hours spread evenly across the day (8am-10pm) instead of following demand."""
    vol = adjusted_volume(fc, s)
    k = 1 + s.aht_change
    needed = pd.Series([agents_needed(v, s.phone_aht_s * k, s.sl_target, s.phone_target_s) for v in vol.phone], index=vol.index)
    open_mask = (vol.index.hour >= 8) & (vol.index.hour < 22)
    flat = np.zeros(len(vol), dtype=int)
    for day in pd.unique(vol.index.date):
        m = (vol.index.date == day) & open_mask
        flat[m] = math.ceil(needed[(vol.index.date == day)].sum() / m.sum())
    sl_flat = np.array([service_level(v, s.phone_aht_s * k, a, s.phone_target_s) for v, a in zip(vol.phone, flat)])
    sl_plan = np.array([service_level(v, s.phone_aht_s * k, a, s.phone_target_s) for v, a in zip(vol.phone, needed)])
    w = vol.phone.to_numpy()
    return {"flat_agent_hours": int(flat.sum()), "plan_agent_hours": int(needed.sum()),
            "flat_sl": float((sl_flat * w).sum() / w.sum()), "plan_sl": float((sl_plan * w).sum() / w.sum()),
            "flat_hours_below_target": int(((sl_flat < s.sl_target) & (w > 0)).sum())}


# ---- 5. AI copilot --------------------------------------------------------------------
BRIEF_PROMPT = """You are a workforce planning analyst for a customer support centre.
Write for operations managers using ONLY the facts JSON provided. Never invent numbers.
Be concrete: name days, hours, channels, agent counts and money. Keep it under 220 words.
If the user asks something the facts can't answer, say what data would be needed.
Return JSON: {"answer": str}"""


def plan_facts(hist: pd.DataFrame, fc: pd.DataFrame, base: dict, scen: dict, s: Scenario,
               bt: dict, anoms: pd.DataFrame) -> dict:
    last_week = hist.iloc[-HOURS_PER_WEEK:]
    daily = base["volume"].sum(axis=1).resample("D").sum()
    return {
        "forecast_next_7_days": {ch: int(fc[ch].sum()) for ch in CHANNELS},
        "last_7_days_actual": {ch: int(last_week[ch].sum()) for ch in CHANNELS},
        "busiest_forecast_day": f"{daily.idxmax():%a %d %b} ({int(daily.max())} contacts)",
        "quietest_forecast_day": f"{daily.idxmin():%a %d %b} ({int(daily.min())} contacts)",
        "peak_hour": f"{base['peak']:%a %d %b %H:00}", "peak_scheduled_agents": base["peak_agents"],
        "baseline": {"fte": round(base["fte"], 1), "weekly_cost": round(base["weekly_cost"])},
        "scenario": {"settings": {k: v for k, v in asdict(s).items() if k in ("volume_change", "aht_change", "ai_deflection", "sl_target", "shrinkage", "cost_per_hour")},
                     "fte": round(scen["fte"], 1), "weekly_cost": round(scen["weekly_cost"]),
                     "weekly_saving_vs_baseline": round(base["weekly_cost"] - scen["weekly_cost"])},
        "forecast_accuracy": {"model_wape": round(bt["model_wape"], 3), "naive_wape": round(bt["naive_wape"], 3)},
        "recent_anomalies": [{"date": str(r.date), "change_pct": round(r.change * 100), "contacts": r.contacts}
                             for r in anoms.itertuples()],
    }


def template_brief(f: dict) -> str:
    fc, lw = f["forecast_next_7_days"], f["last_7_days_actual"]
    total_fc, total_lw = sum(fc.values()), sum(lw.values())
    sc = f["scenario"]
    lines = [
        f"**Next 7 days:** {total_fc:,} contacts forecast ({total_fc / total_lw - 1:+.0%} vs last week): "
        f"phone {fc['phone']:,}, chat {fc['chat']:,}, email {fc['email']:,}.",
        f"**Busiest day:** {f['busiest_forecast_day']}. Peak hour {f['peak_hour']} needs {f['peak_scheduled_agents']} agents scheduled.",
        f"**Baseline plan:** {f['baseline']['fte']} FTE, about {f['baseline']['weekly_cost']:,} per week.",
    ]
    if sc["weekly_saving_vs_baseline"]:
        lines.append(f"**This scenario:** {sc['fte']} FTE, {sc['weekly_cost']:,} per week, "
                     f"a {'saving' if sc['weekly_saving_vs_baseline'] > 0 else 'cost increase'} of "
                     f"{abs(sc['weekly_saving_vs_baseline']):,} per week (~{abs(sc['weekly_saving_vs_baseline']) * 52:,} per year).")
    if f["recent_anomalies"]:
        a = ", ".join(f"{x['date']} ({x['change_pct']:+d}%)" for x in f["recent_anomalies"])
        lines.append(f"**Unusual days in the history:** {a}. Check what caused them before the next similar event.")
    acc = f["forecast_accuracy"]
    lines.append(f"**Forecast accuracy:** {acc['model_wape']:.0%} hourly error vs {acc['naive_wape']:.0%} for a same-hour-last-week guess.")
    return "\n\n".join(lines)


def brief(facts: dict, question: str | None = None, llm: LLM | None = None) -> tuple[str, str]:
    """Returns (text, mode). Falls back to the template when no LLM is available."""
    llm = llm or LLM()
    if llm.live:
        ask = question or "Write this week's staffing brief: volume outlook, staffing and cost, risks, and one recommendation."
        try:
            data = llm.chat_json(BRIEF_PROMPT, f"Facts:\n{json.dumps(facts, default=str)}\n\nRequest: {ask}")
            return str(data.get("answer", "")).strip(), llm.label
        except Exception as exc:
            print(f"[llm] call failed, using fallback: {exc!r}", flush=True)
    if question:
        return ("Free-form questions need an LLM key. Here is the standard brief instead:\n\n" + template_brief(facts), "demo mode (template)")
    return template_brief(facts), "demo mode (template)"
