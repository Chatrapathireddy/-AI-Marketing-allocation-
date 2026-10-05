
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_prep import load_config
from src.mmm_model import (
    aggregate_portfolio,
    evaluate_on_test,
    export_params,
    fit_all_channels,
    fit_channel,
    resolve_mmm_freq,
    saturation_curve,
)

CHANNELS = [
    "google_paid_search",
    "google_shopping",
    "google_pmax",
    "meta_facebook",
    "meta_instagram",
]
RAW = {
    "google_paid_search": "GOOGLE_PAID_SEARCH_SPEND",
    "google_shopping": "GOOGLE_SHOPPING_SPEND",
    "google_pmax": "GOOGLE_PMAX_SPEND",
    "meta_facebook": "META_FACEBOOK_SPEND",
    "meta_instagram": "META_INSTAGRAM_SPEND",
}
# (a_true, b_true, mean monthly spend) — independent channel variation for identifiability
TRUE = {
    "google_paid_search": (40000.0, 1.2e-6, 1_000_000.0),
    "google_shopping": (25000.0, 1.6e-6, 700_000.0),
    "google_pmax": (18000.0, 1.9e-6, 500_000.0),
    "meta_facebook": (22000.0, 1.7e-6, 650_000.0),
    "meta_instagram": (9000.0, 3.0e-6, 200_000.0),
}
BASELINE = 5000.0


def _curve(x, a, b):
    return a * (1.0 - np.exp(-b * x))


@pytest.fixture
def synth_daily() -> pd.DataFrame:
    """18 months of daily portfolio data (2 timeseries) with known a, b, baseline.

    Monthly portfolio spend per channel varies independently; daily rows are split
    so that summing across timeseries then resampling to month reproduces the
    planted monthly spend exactly.
    """
    rng = np.random.default_rng(7)
    months = pd.date_range("2022-01-01", periods=18, freq="MS")
    rows = []
    for m in months:
        days = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="D")
        x_month = {c: TRUE[c][2] * rng.uniform(0.5, 1.5) for c in CHANNELS}
        y_month = BASELINE + sum(
            _curve(x_month[c], TRUE[c][0], TRUE[c][1]) for c in CHANNELS
        )
        y_month *= rng.normal(1.0, 0.02)
        n_ts = 2
        n_cells = len(days) * n_ts
        for d in days:
            for ts in range(n_ts):
                row = {"DATE_DAY": d.strftime("%Y-%m-%d"), "MMM_TIMESERIES_ID": f"TS{ts}"}
                for c in CHANNELS:
                    row[RAW[c]] = x_month[c] / n_cells
                row["y"] = y_month / n_cells
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# saturation_curve
# --------------------------------------------------------------------------- #
def test_saturation_curve_zero_at_zero_spend():
    assert saturation_curve(0.0, 100.0, 1e-3) == 0.0


def test_saturation_curve_monotonic_concave_below_ceiling():
    x = np.linspace(0, 1_000_000, 60)
    y = saturation_curve(x, 30000.0, 1.5e-6)
    d1 = np.diff(y)
    d2 = np.diff(d1)
    assert np.all(d1 > 0)        # strictly increasing -> more spend, more conversions
    assert np.all(d2 < 0)        # strictly concave    -> diminishing returns
    assert y[-1] < 30000.0       # stays below the ceiling a


def test_saturation_curve_vectorized_matches_scalar():
    a, b = 12000.0, 2e-6
    xs = [0.0, 1e5, 5e5]
    vec = saturation_curve(np.array(xs), a, b)
    for i, x in enumerate(xs):
        assert vec[i] == pytest.approx(float(saturation_curve(x, a, b)))


# --------------------------------------------------------------------------- #
# aggregate_portfolio
# --------------------------------------------------------------------------- #
def test_aggregate_portfolio_shape_and_columns(synth_daily):
    cfg = load_config()
    agg = aggregate_portfolio(synth_daily, cfg, CHANNELS, target="y", freq="monthly")
    assert list(agg.columns) == CHANNELS + ["y"]
    assert len(agg) >= 12                       # most of 18 full months retained
    assert (agg[CHANNELS] >= 0).to_numpy().all()
    assert (agg["y"] > 0).all()


def test_aggregate_portfolio_sums_across_timeseries(synth_daily):
    cfg = load_config()
    agg = aggregate_portfolio(synth_daily, cfg, CHANNELS, target="y", freq="monthly")
    # total aggregated spend equals total raw spend (sum is conserved)
    for ch in CHANNELS:
        assert agg[ch].sum() == pytest.approx(synth_daily[RAW[ch]].sum(), rel=1e-6)


# --------------------------------------------------------------------------- #
# fit_all_channels
# --------------------------------------------------------------------------- #
def test_fit_all_channels_positive_params_and_keys(synth_daily):
    cfg = load_config()
    params, baseline = fit_all_channels(
        synth_daily, CHANNELS, config=cfg, freq="monthly", return_baseline=True
    )
    assert set(params) == set(CHANNELS)
    for ch in CHANNELS:
        assert params[ch]["a"] > 0
        assert params[ch]["b"] > 0
    assert baseline >= 0


def test_fit_all_channels_high_in_sample_r2(synth_daily):
    """Model + fitted baseline should explain the planted monthly series well."""
    cfg = load_config()
    params, baseline = fit_all_channels(
        synth_daily, CHANNELS, config=cfg, freq="monthly", return_baseline=True
    )
    agg = aggregate_portfolio(synth_daily, cfg, CHANNELS, freq="monthly")
    pred = baseline + sum(
        saturation_curve(agg[ch].to_numpy(), params[ch]["a"], params[ch]["b"])
        for ch in CHANNELS
    )
    y = agg["y"].to_numpy()
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.9


def test_fit_channel_returns_positive(synth_daily):
    cfg = load_config()
    out = fit_channel(synth_daily, "google_paid_search", config=cfg, freq="monthly")
    assert set(out) == {"a", "b"}
    assert out["a"] > 0 and out["b"] > 0


# --------------------------------------------------------------------------- #
# evaluate_on_test
# --------------------------------------------------------------------------- #
def test_evaluate_on_test_returns_metrics(synth_daily):
    cfg = load_config()
    params = fit_all_channels(synth_daily, CHANNELS, config=cfg, freq="monthly")
    metrics = evaluate_on_test(synth_daily, params, config=cfg, freq="monthly")
    assert metrics["n"] >= 12
    for key in ("r2", "rmse", "mae"):
        assert key in metrics
    assert metrics["rmse"] >= 0


# --------------------------------------------------------------------------- #
# export_params (Meghna's contract)
# --------------------------------------------------------------------------- #
def test_export_params_roundtrip(tmp_path):
    params = {
        ch: {"a": (i + 1) * 1000.0, "b": (i + 1) * 1e-6}
        for i, ch in enumerate(CHANNELS)
    }
    out = tmp_path / "channel_params.json"
    export_params(params, str(out))
    data = json.loads(out.read_text())
    assert list(data.keys()) == CHANNELS
    for ch in CHANNELS:
        assert set(data[ch].keys()) == {"a", "b"}
        assert isinstance(data[ch]["a"], float) and isinstance(data[ch]["b"], float)
        assert data[ch]["a"] > 0 and data[ch]["b"] > 0


def test_resolve_mmm_freq_reads_config_default():
    assert resolve_mmm_freq({"mmm": {"freq": "weekly"}}) == "weekly"
    assert resolve_mmm_freq({}) == "monthly"
    assert resolve_mmm_freq({"mmm": {"freq": "weekly"}}, "daily") == "daily"


# --------------------------------------------------------------------------- #
# weekly support (Task B) — activates automatically once `weekly` is added
# --------------------------------------------------------------------------- #
def test_weekly_frequency_when_supported(synth_daily):
    from src.mmm_model import _FREQ_RULE

    if "weekly" not in _FREQ_RULE:
        pytest.skip("weekly support not added yet (Day-0 Task B)")
    cfg = load_config()
    agg = aggregate_portfolio(synth_daily, cfg, CHANNELS, target="y", freq="weekly")
    assert list(agg.columns) == CHANNELS + ["y"]
    assert len(agg) > 12        # many weeks over 18 months
    assert (agg[CHANNELS] >= 0).to_numpy().all()
