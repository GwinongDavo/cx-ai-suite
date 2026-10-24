import math

import pytest

from cxcore.llm import LLM
from cxcore.workforce import (CHANNELS, Scenario, agents_needed, anomalies, backtest, brief, erlang_c, flat_roster_check,
                              forecast, load_demand, plan_facts, service_level, staff_plan)


@pytest.fixture(scope="module")
def hist():
    return load_demand()


def test_erlang_c_matches_textbook_value():
    # 10 Erlangs of traffic on 12 agents: P(wait) is about 0.449 in standard Erlang C tables.
    assert erlang_c(10, 12) == pytest.approx(0.4494, abs=1e-3)
    assert erlang_c(10, 10) == 1.0  # not enough agents: everyone waits


def test_more_agents_means_better_service():
    sls = [service_level(100, 180, n, 20) for n in range(6, 12)]
    assert sls == sorted(sls) and sls[-1] > 0.95
    n = agents_needed(100, 180, 0.8, 20)
    assert service_level(100, 180, n, 20) >= 0.8 > service_level(100, 180, n - 1, 20)


def test_forecast_beats_seasonal_naive_and_band_is_calibrated(hist):
    bt = backtest(hist)
    assert bt["model_wape"] < bt["naive_wape"]
    assert bt["table"].band_coverage.between(0.7, 0.9).all()


def test_forecast_shape(hist):
    fc = forecast(hist, weeks=1)
    assert len(fc) == 168 and fc.index[0] == hist.index[-1] + (hist.index[1] - hist.index[0])
    assert (fc[CHANNELS] >= 0).all().all()


def test_anomalies_find_the_planted_events(hist):
    found = {str(d) for d in anomalies(hist).date}
    assert {"2026-08-31", "2026-09-14"} <= found
    assert len(found) <= 5  # no flood of false alarms


def test_ai_deflection_lowers_cost_and_volume_raises_it(hist):
    fc = forecast(hist)
    base = staff_plan(fc, Scenario())
    assert staff_plan(fc, Scenario(ai_deflection=0.2))["weekly_cost"] < base["weekly_cost"]
    assert staff_plan(fc, Scenario(volume_change=0.2))["weekly_cost"] > base["weekly_cost"]
    assert base["plan"].scheduled.ge(base["plan"].on_floor).all()


def test_flat_roster_serves_worse_than_hourly_plan(hist):
    fr = flat_roster_check(forecast(hist))
    assert fr["plan_sl"] > fr["flat_sl"]


def test_brief_falls_back_without_llm(hist):
    fc, bt, an = forecast(hist), backtest(hist), anomalies(hist)
    b = staff_plan(fc)
    text, mode = brief(plan_facts(hist, fc, b, b, Scenario(), bt, an), llm=LLM("mock"))
    assert "Next 7 days" in text and "demo mode" in mode
    assert not math.isnan(b["fte"])
