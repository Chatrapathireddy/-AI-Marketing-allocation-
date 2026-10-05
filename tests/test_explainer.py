

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.explainer import (
    _escape_dollar_signs,
    activation_status,
    compare_models,
    diagnose_allocation,
    generate_comparison_explanation,
    generate_explanation,
    plot_activation_status,
    plot_adstock_decay,
    plot_allocation_bar,
    plot_baseline_lift,
    plot_curve_drift,
    plot_saturation_curves,
    plot_sensitivity_tornado,
    plot_shadow_price_trend,
    run_sensitivity,
)
from src.optimizer import OptimResult


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def channel_params():
    return {
        "google_paid_search": {"a": 1.0e-6, "b": 1.0e-8},
        "google_shopping": {"a": 50_000.0, "b": 1.5e-6},
        "google_pmax": {"a": 120_000.0, "b": 4.0e-7},
        "meta_facebook": {"a": 95_000.0, "b": 5.0e-7},
        "meta_instagram": {"a": 1.0e-6, "b": 1.0e-8},
    }


@pytest.fixture
def channel_params_C():
    """Re-fitted parameters under adstock for Model C."""
    return {
        "google_paid_search": {"a": 1.0e-6, "b": 1.0e-8},
        "google_shopping": {"a": 55_000.0, "b": 1.4e-6},
        "google_pmax": {"a": 125_000.0, "b": 4.5e-7},
        "meta_facebook": {"a": 92_000.0, "b": 6.0e-7},
        "meta_instagram": {"a": 1.0e-6, "b": 1.0e-8},
    }


@pytest.fixture
def channels(channel_params):
    return list(channel_params.keys())


@pytest.fixture
def thresholds():
    return {
        "google_paid_search": 18_000.0,
        "google_shopping": 15_000.0,
        "google_pmax": 18_000.0,
        "meta_facebook": 12_000.0,
        "meta_instagram": 12_000.0,
    }


@pytest.fixture
def lambdas():
    return {
        "google_paid_search": 0.30,
        "google_shopping": 0.45,
        "google_pmax": 0.20,
        "meta_facebook": 0.70,
        "meta_instagram": 0.55,
    }


@pytest.fixture
def result_A():
    return OptimResult(
        allocation={
            "google_paid_search": 0.0,
            "google_shopping": 900_000.0,
            "google_pmax": 1_400_000.0,
            "meta_facebook": 1_200_000.0,
            "meta_instagram": 0.0,
        },
        predicted_conversions=82_000.0,
        total_spent=3_500_000.0,
        status="KKT pass · solver converged",
        kkt_status="pass",
        lambda_budget=0.0234,
        success=True,
        baseline_allocation={
            "google_paid_search": 700_000.0,
            "google_shopping": 700_000.0,
            "google_pmax": 700_000.0,
            "meta_facebook": 700_000.0,
            "meta_instagram": 700_000.0,
        },
        baseline_conversions=68_000.0,
        lift_pct=20.6,
    )


@pytest.fixture
def result_B(thresholds):
    return OptimResult(
        allocation={
            "google_paid_search": 0.0,
            "google_shopping": 15_000.0,  # at kappa
            "google_pmax": 1_500_000.0,
            "meta_facebook": 1_485_000.0,
            "meta_instagram": 0.0,
        },
        predicted_conversions=78_000.0,
        total_spent=3_000_000.0,
        status="KKT pass · solver converged",
        kkt_status="pass",
        lambda_budget=0.0260,
        success=True,
    )


@pytest.fixture
def result_C(thresholds, lambdas):
    return OptimResult(
        allocation={
            "google_paid_search": 0.0,
            "google_shopping": 800_000.0,
            "google_pmax": 1_300_000.0,
            "meta_facebook": 1_400_000.0,  # higher under adstock
            "meta_instagram": 0.0,
        },
        predicted_conversions=88_000.0,
        total_spent=3_500_000.0,
        status="KKT pass · solver converged",
        kkt_status="pass",
        lambda_budget=0.0290,
        success=True,
    )


# -----------------------------------------------------------------------------
# _escape_dollar_signs  (LaTeX rendering bug fix)
# -----------------------------------------------------------------------------
def test_escape_dollar_signs_basic():
    out = _escape_dollar_signs("Allocate $50,000 to channel X")
    assert "\\$50,000" in out
    assert "$50,000" not in out.replace("\\$", "")


def test_escape_dollar_signs_idempotent():
    out1 = _escape_dollar_signs("Cost is \\$1,000")
    out2 = _escape_dollar_signs(out1)
    assert out1 == out2


def test_escape_dollar_signs_empty():
    assert _escape_dollar_signs("") == ""
    assert _escape_dollar_signs(None) is None


# -----------------------------------------------------------------------------
# generate_explanation
# -----------------------------------------------------------------------------
def test_generate_explanation_returns_nonempty_string(result_A, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("API_Key", raising=False)
    text = generate_explanation(result_A, params={"budget": 3_500_000})
    assert isinstance(text, str)
    assert len(text) > 100


def test_generate_explanation_no_bare_dollar_signs(result_A, monkeypatch):
    """Output must not contain bare $ that Streamlit would render as LaTeX."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("API_Key", raising=False)
    text = generate_explanation(result_A, params={"budget": 3_500_000})
    # Every $ should be preceded by a backslash.
    for i, ch in enumerate(text):
        if ch == "$":
            assert i > 0 and text[i - 1] == "\\", f"Unescaped $ at index {i}"


# -----------------------------------------------------------------------------
# plot_saturation_curves
# -----------------------------------------------------------------------------
def test_plot_saturation_curves_returns_figure(channel_params):
    fig = plot_saturation_curves(channel_params)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(channel_params)


def test_plot_saturation_curves_empty_params():
    fig = plot_saturation_curves({})
    assert isinstance(fig, go.Figure)


# -----------------------------------------------------------------------------
# plot_allocation_bar
# -----------------------------------------------------------------------------
def test_plot_allocation_bar_returns_figure(result_A):
    fig = plot_allocation_bar(result_A.allocation, baseline=result_A.baseline_allocation)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_allocation_bar_without_baseline(result_A):
    fig = plot_allocation_bar(result_A.allocation)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


# -----------------------------------------------------------------------------
# run_sensitivity
# -----------------------------------------------------------------------------
def test_run_sensitivity_with_stub_optimizer(channel_params, channels):
    def stub_optimizer(params, chans, budget):
        per = budget / len(chans)
        return {"allocation": {c: per for c in chans}, "predicted_conversions": budget * 0.001}

    df = run_sensitivity(
        channel_params,
        budget=3_500_000,
        channels=channels,
        optimizer_fn=stub_optimizer,
        multipliers=[0.5, 1.0, 1.5],
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


# -----------------------------------------------------------------------------
# plot_sensitivity_tornado
# -----------------------------------------------------------------------------
def test_plot_sensitivity_tornado_returns_figure():
    df = pd.DataFrame(
        {
            "multiplier": [0.5, 1.0, 1.5],
            "budget": [5000, 10000, 15000],
            "predicted_conversions": [80.0, 100.0, 115.0],
            "allocation": [{}, {}, {}],
        }
    )
    fig = plot_sensitivity_tornado(df)
    assert isinstance(fig, go.Figure)


# -----------------------------------------------------------------------------
# plot_baseline_lift
# -----------------------------------------------------------------------------
def test_plot_baseline_lift_returns_figure(result_A):
    fig = plot_baseline_lift(result_A.baseline_allocation, result_A.allocation)
    assert isinstance(fig, go.Figure)


# -----------------------------------------------------------------------------
# diagnose_allocation
# -----------------------------------------------------------------------------
def test_diagnose_allocation_flags_zero_channels(result_A, channel_params):
    diagnostics = diagnose_allocation(result_A.allocation, channel_params=channel_params)
    zero_warnings = [d for d in diagnostics if d.get("level") == "warn"]
    assert len(zero_warnings) >= 2


def test_diagnose_allocation_no_caveats_when_clean(channel_params):
    clean_alloc = {c: 100_000.0 for c in channel_params}
    healthy_params = {c: {"a": 50_000.0, "b": 1e-5} for c in clean_alloc}
    diagnostics = diagnose_allocation(clean_alloc, channel_params=healthy_params)
    assert diagnostics == []


# -----------------------------------------------------------------------------
# activation_status  (NEW)
# -----------------------------------------------------------------------------
def test_activation_status_off(thresholds):
    alloc = {"google_shopping": 0.0, "meta_facebook": 0.0}
    status = activation_status(alloc, thresholds=thresholds)
    assert all(s["status"] == "OFF" for s in status)


def test_activation_status_at_kappa(thresholds):
    alloc = {"google_shopping": 15_000.0, "meta_facebook": 12_000.0}
    status = activation_status(alloc, thresholds=thresholds)
    assert all(s["status"] == "AT_KAPPA" for s in status)


def test_activation_status_on_interior(thresholds):
    alloc = {"google_shopping": 100_000.0, "meta_facebook": 200_000.0}
    status = activation_status(alloc, thresholds=thresholds)
    assert all(s["status"] == "ON_INTERIOR" for s in status)


def test_activation_status_no_thresholds():
    alloc = {"x": 5.0, "y": 0.0}
    status = activation_status(alloc, thresholds=None)
    assert {s["status"] for s in status} == {"ON_INTERIOR", "OFF"}


# -----------------------------------------------------------------------------
# plot_adstock_decay  (NEW)
# -----------------------------------------------------------------------------
def test_plot_adstock_decay_returns_figure(lambdas):
    fig = plot_adstock_decay(lambdas)
    assert isinstance(fig, go.Figure)


def test_plot_adstock_decay_empty():
    fig = plot_adstock_decay({})
    assert isinstance(fig, go.Figure)


# -----------------------------------------------------------------------------
# plot_curve_drift  (NEW)
# -----------------------------------------------------------------------------
def test_plot_curve_drift_returns_figure(channel_params, channel_params_C, lambdas):
    fig = plot_curve_drift(channel_params, channel_params_C, lambdas=lambdas)
    assert isinstance(fig, go.Figure)


def test_plot_curve_drift_missing_inputs():
    fig = plot_curve_drift({}, {})
    assert isinstance(fig, go.Figure)


# -----------------------------------------------------------------------------
# plot_shadow_price_trend  (NEW)
# -----------------------------------------------------------------------------
def test_plot_shadow_price_trend_returns_figure():
    fig = plot_shadow_price_trend({"Model A": 0.02, "Model B": 0.026, "Model C": 0.029})
    assert isinstance(fig, go.Figure)


def test_plot_shadow_price_trend_empty():
    fig = plot_shadow_price_trend({})
    assert isinstance(fig, go.Figure)


# -----------------------------------------------------------------------------
# plot_activation_status  (NEW)
# -----------------------------------------------------------------------------
def test_plot_activation_status_returns_figure(result_B, thresholds):
    status = activation_status(result_B.allocation, thresholds=thresholds)
    fig = plot_activation_status(status)
    assert isinstance(fig, go.Figure)


# -----------------------------------------------------------------------------
# compare_models  (NEW)
# -----------------------------------------------------------------------------
def test_compare_models_all_present(result_A, result_B, result_C, thresholds, lambdas):
    metadata = {
        "A": {"name": "Base"},
        "B": {"name": "Activation", "thresholds": thresholds},
        "C": {"name": "Adstock + Activation", "thresholds": thresholds, "lambdas": lambdas},
    }
    df = compare_models(result_A, result_B, result_C, metadata=metadata)
    assert len(df) == 3
    assert "Predicted Conversions" in df.columns
    assert "Shadow Price (λ_budget)" in df.columns
    assert "KKT Status" in df.columns


def test_compare_models_partial(result_A):
    df = compare_models(result_A, None, None)
    assert len(df) == 3
    assert df.iloc[1]["Notes"] == "not run yet"
    assert df.iloc[2]["Notes"] == "not run yet"


def test_compare_models_all_missing():
    df = compare_models(None, None, None)
    assert len(df) == 3
    assert (df["Notes"] == "not run yet").all()


# -----------------------------------------------------------------------------
# generate_comparison_explanation  (NEW)
# -----------------------------------------------------------------------------
def test_generate_comparison_explanation_template_fallback(
    result_A, result_B, result_C, thresholds, lambdas, monkeypatch
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("API_Key", raising=False)
    metadata = {
        "A": {"name": "Base"},
        "B": {"name": "Activation", "thresholds": thresholds},
        "C": {"name": "Adstock + Activation", "thresholds": thresholds, "lambdas": lambdas},
    }
    text = generate_comparison_explanation(result_A, result_B, result_C, metadata)
    assert isinstance(text, str)
    # The template covers all 8 questions:
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"):
        assert q in text


def test_generate_comparison_explanation_handles_missing_models(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("API_Key", raising=False)
    text = generate_comparison_explanation(None, None, None)
    assert isinstance(text, str)
    assert len(text) > 100
