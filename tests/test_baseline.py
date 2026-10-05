"""Tests for baseline comparisons."""
from __future__ import annotations

import pandas as pd
import pytest

from src.baseline import (
    apply_baseline_to_result,
    compute_equal_baseline,
    compute_historical_baseline,
    compute_lift,
)
from src.optimizer import OptimResult, predicted_conversions


@pytest.fixture
def params():
    return {
        "google_paid_search": {"a": 100.0, "b": 0.001},
        "meta_facebook": {"a": 80.0, "b": 0.002},
    }


@pytest.fixture
def channels():
    return ["google_paid_search", "meta_facebook"]


@pytest.fixture
def spend_df():
    return pd.DataFrame(
        {
            "google_paid_search_adstock": [100.0, 200.0, 300.0],
            "META_FACEBOOK_SPEND": [50.0, 50.0, 100.0],
        }
    )


def test_compute_equal_baseline_splits_budget(channels):
    alloc = compute_equal_baseline(10_000.0, channels)
    assert alloc == {"google_paid_search": 5000.0, "meta_facebook": 5000.0}
    assert sum(alloc.values()) == pytest.approx(10_000.0)


def test_compute_equal_baseline_rejects_invalid_budget(channels):
    with pytest.raises(ValueError, match="budget"):
        compute_equal_baseline(0, channels)


def test_compute_historical_baseline_uses_spend_proportions(spend_df, channels):
    config = {
        "column_map": {
            "google_paid_search": "GOOGLE_PAID_SEARCH_SPEND",
            "meta_facebook": "META_FACEBOOK_SPEND",
        }
    }
    alloc = compute_historical_baseline(spend_df, channels, 1000.0, config)
    # google mean 200, meta mean 66.67 → ~75% / ~25%
    assert alloc["google_paid_search"] == pytest.approx(750.0, rel=0.01)
    assert alloc["meta_facebook"] == pytest.approx(250.0, rel=0.01)
    assert sum(alloc.values()) == pytest.approx(1000.0)


def test_compute_historical_baseline_falls_back_to_equal_when_no_spend(channels):
    empty = pd.DataFrame({"x": [1, 2, 3]})
    config = {"column_map": {"google_paid_search": "MISSING", "meta_facebook": "ALSO_MISSING"}}
    alloc = compute_historical_baseline(empty, channels, 100.0, config)
    assert alloc == compute_equal_baseline(100.0, channels)


def test_compute_lift_higher_when_optimized_prefers_stronger_channel(params, channels):
    optimized = {"google_paid_search": 10_000.0, "meta_facebook": 0.0}
    baseline = {"google_paid_search": 0.0, "meta_facebook": 10_000.0}
    lift = compute_lift(optimized, baseline, params, channels)
    assert lift["baseline_conversions"] > 0
    assert lift["optimized_conversions"] > lift["baseline_conversions"]
    assert lift["lift_pct"] > 0


def test_compute_lift_zero_when_baseline_conversions_zero(params, channels):
    zero_baseline = {"google_paid_search": 0.0, "meta_facebook": 0.0}
    optimized = {"google_paid_search": 5000.0, "meta_facebook": 5000.0}
    lift = compute_lift(optimized, zero_baseline, params, channels)
    assert lift["baseline_conversions"] == 0.0
    assert lift["lift_pct"] == 0.0


def test_apply_baseline_to_result_fills_optim_result(spend_df, params, channels):
    budget = 1000.0
    optim = OptimResult(
        allocation={"google_paid_search": 700.0, "meta_facebook": 300.0},
        predicted_conversions=predicted_conversions([700.0, 300.0], params, channels),
        total_spent=1000.0,
        status="ok",
        kkt_status="pass",
        lambda_budget=0.1,
        success=True,
    )
    config = {
        "column_map": {
            "google_paid_search": "GOOGLE_PAID_SEARCH_SPEND",
            "meta_facebook": "META_FACEBOOK_SPEND",
        }
    }
    out = apply_baseline_to_result(
        optim, params, channels, budget, spend_df, config=config, method="historical"
    )
    assert out.baseline_allocation
    assert sum(out.baseline_allocation.values()) == pytest.approx(budget)
    assert out.baseline_conversions > 0
    assert isinstance(out.lift_pct, float)
