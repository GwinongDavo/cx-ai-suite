"""SLA metrics and an escalation-risk model over the ticket dataset.

The risk model only uses fields known when a ticket is created (priority, tier,
channel, category, country, queue load, hour, weekday), so it can score open
tickets before they breach, not explain breaches after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import DATA_DIR

NOW = pd.Timestamp("2026-10-20 18:00")
CATEGORICAL = ["priority", "customer_tier", "channel", "category", "country"]
NUMERIC = ["queue_load", "hour", "weekday"]


def load_tickets() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "tickets.csv", parse_dates=["created_at", "first_response_at", "resolved_at"])
    df["hour"] = df.created_at.dt.hour
    df["weekday"] = df.created_at.dt.weekday
    end = df.resolved_at.fillna(NOW)
    df["age_h"] = (end - df.created_at).dt.total_seconds() / 3600
    df["frt_h"] = (df.first_response_at - df.created_at).dt.total_seconds() / 3600
    df["resolution_breached"] = df.age_h > df.sla_resolution_h
    df["frt_breached"] = df.frt_h.fillna((NOW - df.created_at).dt.total_seconds() / 3600) > df.sla_first_response_h
    df["due_at"] = df.created_at + pd.to_timedelta(df.sla_resolution_h, unit="h")
    df["week"] = df.created_at.dt.to_period("W").dt.start_time
    return df


def matured(df: pd.DataFrame) -> pd.DataFrame:
    """Tickets whose resolution deadline has passed, open or not.

    Measuring breach rate on resolved tickets only would be survivorship bias: recent
    weeks would look great because only the quick tickets have closed so far.
    """
    return df[df.due_at <= NOW]


def kpis(df: pd.DataFrame) -> dict:
    resolved = df[df.status == "resolved"]
    return {
        "tickets": len(df),
        "open": int((df.status == "open").sum()),
        "resolution_sla_met": 1 - matured(df).resolution_breached.mean(),
        "first_response_sla_met": 1 - df.frt_breached.mean(),
        "median_resolution_h": float(resolved.age_h.median()),
        "csat": float(pd.to_numeric(df.csat, errors="coerce").mean()),
        "reopen_rate": float(df.reopened.mean()),
    }


def breach_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    r = matured(df)
    out = r.groupby(col).agg(tickets=("ticket_id", "count"), breach_rate=("resolution_breached", "mean")).reset_index()
    return out.sort_values("breach_rate", ascending=False)


@dataclass
class RiskModel:
    pipeline: Pipeline
    auc: float
    avg_precision: float
    base_rate: float
    top_drivers: pd.DataFrame

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df[CATEGORICAL + NUMERIC])[:, 1]


def train_risk_model(df: pd.DataFrame, seed: int = 0) -> RiskModel:
    data = df[df.status == "resolved"]
    X, y = data[CATEGORICAL + NUMERIC], data.resolution_breached.astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    pipe = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                                    ("num", StandardScaler(), NUMERIC)])),
        ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
    ])
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    drivers = (pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "weight": coefs})
               .assign(abs=lambda d: d.weight.abs()).sort_values("abs", ascending=False).head(10).drop(columns="abs"))
    pipe.fit(X, y)  # refit on everything once the holdout numbers are recorded
    return RiskModel(pipe, float(roc_auc_score(y_te, p)), float(average_precision_score(y_te, p)), float(y.mean()), drivers)


def open_risk(df: pd.DataFrame, model: RiskModel) -> pd.DataFrame:
    o = df[df.status == "open"].copy()
    o["risk"] = model.score(o)
    o["hours_to_due"] = (o.due_at - NOW).dt.total_seconds() / 3600
    o["state"] = np.select([o.hours_to_due < 0, o.hours_to_due < 12, o.risk >= 0.5],
                           ["Breached", "Due < 12h", "High risk"], "On track")
    return o.sort_values(["hours_to_due"])
