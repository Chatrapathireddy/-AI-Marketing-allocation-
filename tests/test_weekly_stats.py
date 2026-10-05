"""Tests for src/weekly_stats.py — uses synthetic DataFrames only."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_prep import load_config
from src.weekly_stats import (
    B_SCENARIO_ACTIVATION,
    B_TARGET,
    DEFAULT_KAPPA,
    KAPPA_SUM,
    compute_uc_ceilings,
    compute_weekly_stats,
    print_uc_ceilings,
    print_verification,
    scale_decision,
    verify_pipeline_outputs,
    weekly_portfolio,
    write_handoff,
)

MODELED = [
    "google_paid_search",
    "google_shopping",
    "google_pmax",
    "meta_facebook",
    "meta_instagram",
]


@pytest.fixture
def config() -> dict:
    return load_config()


@pytest.fixture
def pipeline_like_df(config) -> pd.DataFrame:
    """Synthetic frame shaped like run_pipeline()'s train_df output."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    column_map = config["column_map"]
    df = pd.DataFrame({"DATE_DAY": dates})
    for ch in MODELED:
        df[column_map[ch]] = rng.uniform(100, 1000, size=len(dates))
        df[f"{ch}_adstock"] = rng.uniform(100, 1200, size=len(dates))
    df["y"] = rng.uniform(5000, 20000, size=len(dates))
    df["CURRENCY_CODE"] = "USD"
    df["MMM_TIMESERIES_ID"] = "ts_1"
    return df


@pytest.fixture
def test_like_df(pipeline_like_df) -> pd.DataFrame:
    out = pipeline_like_df.copy()
    out["DATE_DAY"] = out["DATE_DAY"] + pd.Timedelta(days=70)
    return out.head(21)


# ---------------------------------------------------------------- DEFAULT_KAPPA
def test_kappa_sum_is_75k():
    assert KAPPA_SUM == 75_000
    assert set(DEFAULT_KAPPA) == set(MODELED)


# ----------------------------------------------- verify_pipeline_outputs
def test_verify_pipeline_outputs_success(pipeline_like_df, test_like_df, config, capsys):
    out = verify_pipeline_outputs(pipeline_like_df, test_like_df, config)
    assert out["pipeline_verified"] is True
    assert out["missing_columns"] == []
    assert len(out["adstock_cols_present"]) == 5
    assert out["train_rows"] == len(pipeline_like_df)
    assert out["test_rows"] == len(test_like_df)
    print_verification(out, config)
    assert "All USD" in capsys.readouterr().out


def test_verify_pipeline_outputs_flags_missing_columns_and_non_usd(
    pipeline_like_df, test_like_df, config
):
    broken = pipeline_like_df.drop(columns=["y"]).copy()
    broken.loc[broken.index[:5], "CURRENCY_CODE"] = "EUR"
    out = verify_pipeline_outputs(broken, test_like_df, config)
    assert out["pipeline_verified"] is False
    assert "y" in out["missing_columns"]
    assert any("non-USD" in w for w in out["warnings"])


def test_verify_pipeline_outputs_warns_on_high_nulls(pipeline_like_df, test_like_df, config):
    df = pipeline_like_df.copy()
    df.loc[df.index[:20], "y"] = np.nan  # ~29% nulls
    out = verify_pipeline_outputs(df, test_like_df, config)
    assert any(w.startswith("y:") for w in out["warnings"])


# ----------------------------------------------------- weekly_portfolio
def test_weekly_portfolio_resamples_to_weeks(pipeline_like_df, config):
    spend_cols = [config["column_map"][ch] for ch in MODELED]
    weekly = weekly_portfolio(pipeline_like_df, spend_cols, config)
    assert 10 <= len(weekly) <= 11  # 70 days
    # Weekly totals must preserve total spend
    assert weekly[spend_cols[0]].sum() == pytest.approx(
        pipeline_like_df[spend_cols[0]].sum()
    )


def test_weekly_portfolio_ignores_missing_columns(pipeline_like_df, config):
    weekly = weekly_portfolio(pipeline_like_df, ["NOT_A_COLUMN"], config)
    assert "NOT_A_COLUMN" not in weekly.columns
    assert "y" in weekly.columns


# -------------------------------------------------- compute_weekly_stats
def test_compute_weekly_stats_structure(pipeline_like_df, test_like_df, config):
    stats = compute_weekly_stats(pipeline_like_df, test_like_df, config)
    assert set(stats["per_channel_weekly"]) == set(MODELED)
    for st in stats["per_channel_weekly"].values():
        assert st["min"] <= st["median"] <= st["max"]
    assert stats["B_raw"] > 0
    assert stats["train_weeks"] >= 10
    assert stats["holdout_weeks"] >= 3
    assert stats["weekly_y_mean"] > 0
    assert stats["train_rows"] == len(pipeline_like_df)
    assert stats["test_rows"] == len(test_like_df)


# ---------------------------------------------------------- scale_decision
def test_scale_decision_scales_down_large_spend():
    decision = scale_decision(b_raw=900_000.0)
    assert decision["scale_down"] is True
    assert decision["scale_factor"] == pytest.approx(B_TARGET / 900_000.0)
    assert decision["B_recommended"] == pytest.approx(B_TARGET)


def test_scale_decision_keeps_full_scale_for_small_spend():
    decision = scale_decision(b_raw=100_000.0)
    assert decision["scale_down"] is False
    assert decision["scale_factor"] == 1.0
    assert decision["B_recommended"] == 100_000.0


def test_scale_decision_handles_zero_spend():
    decision = scale_decision(b_raw=0.0)
    assert decision["scale_factor"] == 1.0
    assert decision["B_recommended"] == 0.0


def test_scale_decision_uses_custom_kappa_sum():
    # custom kappa dict with sum = 150_000
    # b_raw = 400_000 -> ratio 2.7x -> scale_down must be False
    # (would be True against the default 75_000 sum)
    custom_kappa = {ch: 30_000.0 for ch in DEFAULT_KAPPA}
    assert sum(custom_kappa.values()) == 150_000.0

    default_decision = scale_decision(b_raw=400_000.0)
    assert default_decision["scale_down"] is True

    custom_decision = scale_decision(b_raw=400_000.0, kappa=custom_kappa)
    assert custom_decision["scale_down"] is False
    assert custom_decision["kappa_sum"] == 150_000.0


# ------------------------------------------------------- compute_uc_ceilings
def test_compute_uc_ceilings_formula_and_no_warnings():
    per_channel = {ch: {"min": 0.0, "median": 50_000.0, "max": 100_000.0} for ch in DEFAULT_KAPPA}
    out = compute_uc_ceilings(per_channel)
    assert set(out["uc_ceilings"]) == set(DEFAULT_KAPPA)
    for ch in DEFAULT_KAPPA:
        assert out["uc_ceilings"][ch] == pytest.approx(1.5 * 100_000.0)
    assert out["uc_warnings"] == []


def test_compute_uc_ceilings_flags_channel_below_kappa():
    per_channel = {ch: {"min": 0.0, "median": 50_000.0, "max": 100_000.0} for ch in DEFAULT_KAPPA}
    # meta_instagram default kappa = 12_000; max weekly 5_000 -> u_c = 7_500 < kappa
    per_channel["meta_instagram"] = {"min": 0.0, "median": 2_000.0, "max": 5_000.0}
    out = compute_uc_ceilings(per_channel)
    assert out["uc_ceilings"]["meta_instagram"] == pytest.approx(7_500.0)
    assert any(w.startswith("meta_instagram") for w in out["uc_warnings"])


def test_compute_uc_ceilings_respects_custom_kappa():
    # channel max weekly = 10_000 -> u_c = 15_000
    # default kappa for google_paid_search is 18_000 -> would flag
    # custom kappa of 9_000 -> must NOT flag
    per_channel = {ch: {"min": 0.0, "median": 8_000.0, "max": 10_000.0} for ch in DEFAULT_KAPPA}

    default_out = compute_uc_ceilings(per_channel)
    assert any(w.startswith("google_paid_search") for w in default_out["uc_warnings"])

    custom_kappa = {ch: 9_000.0 for ch in DEFAULT_KAPPA}
    custom_out = compute_uc_ceilings(per_channel, kappa=custom_kappa)
    assert custom_out["uc_ceilings"]["google_paid_search"] == pytest.approx(15_000.0)
    assert custom_out["uc_warnings"] == []
    assert custom_out["kappa"] == custom_kappa


def test_compute_uc_ceilings_missing_channel_defaults_to_zero():
    out = compute_uc_ceilings({})
    assert all(v == 0.0 for v in out["uc_ceilings"].values())
    assert len(out["uc_warnings"]) == len(DEFAULT_KAPPA)


def test_print_uc_ceilings_prints_table_and_flags(capsys):
    per_channel = {ch: {"min": 0.0, "median": 50_000.0, "max": 100_000.0} for ch in DEFAULT_KAPPA}
    per_channel["meta_instagram"] = {"min": 0.0, "median": 2_000.0, "max": 5_000.0}
    print_uc_ceilings(compute_uc_ceilings(per_channel))
    out = capsys.readouterr().out
    assert "Channel Ceilings u_c" in out
    assert "[FLAG] meta_instagram" in out


# ------------------------------------------------------------ write_handoff
def test_write_handoff_creates_valid_json(
    pipeline_like_df, test_like_df, config, tmp_path
):
    verification = verify_pipeline_outputs(pipeline_like_df, test_like_df, config)
    stats = compute_weekly_stats(pipeline_like_df, test_like_df, config)
    uc_result = compute_uc_ceilings(stats["per_channel_weekly"])

    out_path = tmp_path / "handoff.json"
    handoff = write_handoff(stats, uc_result, config, verification, out_path=out_path)

    assert out_path.exists()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded == handoff
    assert loaded["pipeline_verified"] is True
    assert loaded["kappa"]["google_paid_search"] == 18_000
    assert loaded["train_rows"] == len(pipeline_like_df)
    assert set(loaded["per_channel_weekly"]) == set(MODELED)
    # u_c / B fields for Meghna
    assert set(loaded["uc_ceilings"]) == set(MODELED)
    for ch in MODELED:
        expected_uc = 1.5 * loaded["per_channel_weekly"][ch]["max"]
        assert loaded["uc_ceilings"][ch] == pytest.approx(expected_uc, rel=1e-3)
    assert loaded["B_portfolio"] == pytest.approx(loaded["B_raw"])
    assert loaded["B_scenario_activation"] == B_SCENARIO_ACTIVATION == 90_000
    assert isinstance(loaded["uc_warnings"], list)
    # weekly_stats.json is written next to the handoff
    assert (tmp_path / "weekly_stats.json").exists()
