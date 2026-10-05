"""Tests for optimization_pipeline."""
from __future__ import annotations

from src.data_prep import load_config
from src.optimization_pipeline import (
    apply_optimization_to_session,
    load_activation_thresholds,
    load_model_c_inputs,
    optimizer_fn_for_sensitivity,
    run_model_c,
    run_optimization_pipeline,
)
from src.optimizer import OptimResult


def test_run_optimization_pipeline_returns_result():
    params = {
        "google_paid_search": {"a": 100.0, "b": 0.001},
        "google_shopping": {"a": 80.0, "b": 0.002},
        "google_pmax": {"a": 90.0, "b": 0.0015},
        "meta_facebook": {"a": 70.0, "b": 0.003},
        "meta_instagram": {"a": 60.0, "b": 0.004},
    }
    optim, out_params, budget, model_b, model_c = run_optimization_pipeline(
        confirmed_budget=10_000.0,
        channel_params=params,
    )
    assert out_params == params
    assert budget == 10_000.0
    assert optim.total_spent <= budget + 1e-6
    assert optim.allocation
    assert model_b is not None
    assert model_b.result.total_spent <= budget + 1e-3


def test_optimizer_fn_for_sensitivity_shape():
    params = {"ch_a": {"a": 50.0, "b": 0.01}}
    out = optimizer_fn_for_sensitivity(params, ["ch_a"], 1000.0)
    assert "allocation" in out and "predicted_conversions" in out


def test_load_activation_thresholds_from_config():
    thresholds = load_activation_thresholds()
    config = load_config()
    modeled = list(config["channels"]["modeled"])
    assert set(thresholds) == set(modeled)
    assert thresholds["meta_facebook"] == 12_000.0
    assert thresholds["google_paid_search"] == 18_000.0


def test_load_activation_thresholds_custom_config():
    config = {
        "channels": {"modeled": ["alpha", "beta"]},
        "activation": {
            "thresholds": {"alpha": 5000.0, "beta": 7000.0},
            "ceilings": {"alpha": 50_000.0, "beta": 60_000.0},
        },
    }
    assert load_activation_thresholds(config) == {"alpha": 5000.0, "beta": 7000.0}


def test_run_optimization_pipeline_fits_weekly(monkeypatch):
    captured: dict = {}

    def fake_fitting(*, freq=None, **kwargs):
        captured["freq"] = freq
        return {"params": {"ch_a": {"a": 10.0, "b": 0.01}}}

    monkeypatch.setattr("src.optimization_pipeline.run_fitting", fake_fitting)
    monkeypatch.setattr(
        "src.optimization_pipeline.load_config",
        lambda: {
            "channels": {"modeled": ["ch_a"]},
            "optimization": {"default_budget": 1000},
            "mmm": {"freq": "weekly"},
            "data": {"train_path": "missing.csv"},
            "activation": {"thresholds": {}},
        },
    )
    monkeypatch.setattr("src.optimization_pipeline.load_bo_params", lambda _cfg: None)
    monkeypatch.setattr("src.optimization_pipeline.resolve_project_path", lambda p: p)

    run_optimization_pipeline(confirmed_budget=500.0)
    assert captured["freq"] == "weekly"


def test_run_optimization_pipeline_runs_model_b_with_config():
    params = {
        "google_paid_search": {"a": 100.0, "b": 0.001},
        "google_shopping": {"a": 80.0, "b": 0.002},
        "google_pmax": {"a": 90.0, "b": 0.0015},
        "meta_facebook": {"a": 70.0, "b": 0.003},
        "meta_instagram": {"a": 60.0, "b": 0.004},
    }
    optim, _, budget, model_b, model_c = run_optimization_pipeline(
        confirmed_budget=100_000.0,
        channel_params=params,
    )
    assert budget == 100_000.0
    assert model_b is not None
    assert model_b.result.total_spent <= budget + 1e-3


def test_run_model_c_solves_at_portfolio_budget():
    # Greg's Model C inputs live in data/processed; load + solve at a budget that
    # clears the κ thresholds so several channels can turn ON.
    config = load_config()
    channels = list(config["channels"]["modeled"])
    params_c, lambdas = load_model_c_inputs(config)
    assert params_c is not None and lambdas is not None
    assert lambdas["google_paid_search"] == 0.3  # exercises the f(s/(1-λ)) bridge

    model_c = run_model_c(831_000.0, channels, config=config)
    assert model_c is not None
    assert model_c.result.total_spent <= 831_000.0 + 1.0


def test_apply_optimization_to_session_sets_activation_thresholds():
    class FakeSession:
        pass

    session = FakeSession()
    optim = OptimResult(
        allocation={"ch_a": 100.0},
        predicted_conversions=10.0,
        total_spent=100.0,
        status="ok",
        kkt_status="pass",
        lambda_budget=0.1,
        success=True,
    )
    apply_optimization_to_session(session, optim, {"ch_a": {"a": 1.0, "b": 0.1}})
    assert session.optim_result is optim
    assert session.activation_thresholds["meta_facebook"] == 12_000.0
    assert session.optim_result_B is None
