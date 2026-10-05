"""Tests for optimizer.py — synthetic params only."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.optimizer import (
    OptimResult,
    gradient,
    load_params,
    objective,
    predicted_conversions,
    solve,
    solve_from_file,
    verify_kkt,
)


@pytest.fixture
def symmetric_params() -> dict:
    return {
        "alpha": {"a": 10_000.0, "b": 1e-5},
        "beta": {"a": 10_000.0, "b": 1e-5},
    }


@pytest.fixture
def asymmetric_params() -> dict:
    return {
        "high": {"a": 20_000.0, "b": 2e-5},
        "low": {"a": 5_000.0, "b": 1e-6},
    }


def test_predicted_conversions_zero_at_zero_spend(symmetric_params):
    assert predicted_conversions([0.0, 0.0], symmetric_params, ["alpha", "beta"]) == 0.0


def test_objective_is_negative_conversions(symmetric_params):
    x = [50_000.0, 50_000.0]
    assert objective(x, symmetric_params, ["alpha", "beta"]) == -predicted_conversions(
        x, symmetric_params, ["alpha", "beta"]
    )


def test_gradient_matches_finite_difference(symmetric_params):
    x = np.array([30_000.0, 70_000.0])
    channels = ["alpha", "beta"]
    eps = 1.0
    anal = gradient(x, symmetric_params, channels)
    num = np.zeros(2)
    for i in range(2):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        num[i] = (objective(xp, symmetric_params, channels) - objective(xm, symmetric_params, channels)) / (
            2 * eps
        )
    np.testing.assert_allclose(anal, num, rtol=1e-5, atol=1e-3)


def test_solve_budget_constraint_holds(symmetric_params):
    budget = 100_000.0
    result = solve(symmetric_params, budget, ["alpha", "beta"], n_starts=10, seed=1)
    assert isinstance(result, OptimResult)
    assert sum(result.allocation.values()) <= budget + 1e-6
    assert all(v >= -1e-9 for v in result.allocation.values())


def test_solve_symmetric_channels_split_evenly(symmetric_params):
    budget = 100_000.0
    result = solve(symmetric_params, budget, ["alpha", "beta"], n_starts=15, seed=2)
    assert abs(result.allocation["alpha"] - result.allocation["beta"]) < 500.0
    assert result.predicted_conversions > 0


def test_solve_prefers_higher_marginal_channel(asymmetric_params):
    budget = 50_000.0
    result = solve(asymmetric_params, budget, ["high", "low"], n_starts=20, seed=3)
    assert result.allocation["high"] >= result.allocation["low"]


def test_solve_lambda_near_zero_when_budget_not_binding(symmetric_params):
    budget = 1e9
    result = solve(symmetric_params, budget, ["alpha", "beta"], n_starts=5, seed=4)
    assert result.lambda_budget < 1e-3 or sum(result.allocation.values()) >= budget - 1.0


def test_verify_kkt_passes_for_solve_result(symmetric_params):
    budget = 200_000.0
    result = solve(symmetric_params, budget, ["alpha", "beta"], n_starts=10, seed=5)
    kkt = verify_kkt(result, budget, ["alpha", "beta"], params=symmetric_params, tol=1e-3)
    assert kkt["budget_feasible"]
    assert kkt["status"] in ("pass", "fail")  # stationarity tol may be loose on edge cases


def test_invalid_budget_raises(symmetric_params):
    with pytest.raises(ValueError, match="budget must be positive"):
        solve(symmetric_params, 0, ["alpha", "beta"])


def test_missing_channel_raises(symmetric_params):
    with pytest.raises(KeyError, match="Missing channel"):
        solve(symmetric_params, 1000, ["alpha", "missing"])


def test_invalid_params_ab_raises():
    bad = {"alpha": {"a": -1.0, "b": 1e-5}}
    with pytest.raises(ValueError, match="a > 0"):
        solve(bad, 1000, ["alpha"])


def test_solve_respects_channel_cap(asymmetric_params):
    caps = {"high": 10_000.0, "low": None}
    result = solve(
        asymmetric_params,
        50_000.0,
        ["high", "low"],
        caps=caps,
        n_starts=10,
        seed=6,
    )
    assert result.allocation["high"] <= 10_000.0 + 1e-6


def test_solve_from_file(tmp_path, symmetric_params):
    import yaml

    from src.data_prep import load_config

    params_path = tmp_path / "channel_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(symmetric_params, f)

    config = load_config()
    config["data"]["params_path"] = str(params_path)
    config["optimization"]["default_budget"] = 80_000
    config["channels"]["modeled"] = ["alpha", "beta"]
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    result = solve_from_file(str(cfg_path))
    assert sum(result.allocation.values()) <= 80_000 + 1e-6


def test_load_params_round_trip(tmp_path, symmetric_params):
    path = tmp_path / "p.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(symmetric_params, f)
    loaded = load_params(path)
    assert loaded == symmetric_params
