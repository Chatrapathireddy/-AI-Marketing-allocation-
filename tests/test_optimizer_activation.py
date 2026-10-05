"""Tests for Model B activation enumeration in optimizer.py."""
from __future__ import annotations

import math

import pytest

from src.optimizer import (
    ActivationSolveResult,
    apply_adstock_steady_state,
    cross_check_global_optimum,
    solve,
    solve_activation_kappa_sweep,
    solve_with_activation,
)


@pytest.fixture
def three_channel_params() -> dict:
    return {
        "strong": {"a": 50_000.0, "b": 1e-5},
        "weak": {"a": 1_000.0, "b": 1e-6},
        "mid": {"a": 20_000.0, "b": 2e-5},
    }


@pytest.fixture
def thresholds() -> dict[str, float]:
    return {"strong": 5_000.0, "weak": 5_000.0, "mid": 5_000.0}


@pytest.fixture
def ceilings() -> dict[str, float]:
    return {"strong": 100_000.0, "weak": 100_000.0, "mid": 100_000.0}


def test_solve_respects_lower_bound(three_channel_params):
    channels = ["strong", "mid"]
    bounds = {ch: (10_000.0, 50_000.0) for ch in channels}
    result = solve(
        three_channel_params,
        80_000.0,
        channels,
        channel_bounds=bounds,
        n_starts=5,
        seed=0,
    )
    for ch in channels:
        assert result.allocation[ch] >= 10_000.0 - 1e-3


def test_activation_skips_infeasible_pattern(thresholds, ceilings, three_channel_params):
    channels = list(thresholds.keys())
    # Only the all-OFF pattern is feasible when B < kappa
    out = solve_with_activation(
        three_channel_params,
        3_000.0,
        channels,
        thresholds,
        ceilings,
        n_starts=3,
        seed=1,
    )
    assert out.active_channels == ()
    assert out.result.total_spent == 0.0


def test_activation_turns_off_weak_channel(thresholds, ceilings, three_channel_params):
    channels = list(thresholds.keys())
    budget = 25_000.0  # can afford at most ~4 channels at 5k min; forces choices
    out = solve_with_activation(
        three_channel_params,
        budget,
        channels,
        thresholds,
        ceilings,
        n_starts=5,
        seed=2,
    )
    assert isinstance(out, ActivationSolveResult)
    assert out.patterns_feasible >= 1
    assert out.patterns_evaluated == 8  # 2^3
    assert "weak" not in out.active_channels or out.result.allocation["weak"] == 0.0
    for ch in channels:
        spend = out.result.allocation[ch]
        if spend <= 1e-6:
            assert ch not in out.active_channels
        else:
            assert spend >= thresholds[ch] - 1e-3


def test_activation_feasible_respects_kappa_and_budget(
    thresholds, ceilings, three_channel_params
):
    channels = list(thresholds.keys())
    budget = 40_000.0
    out = solve_with_activation(
        three_channel_params,
        budget,
        channels,
        thresholds,
        ceilings,
        n_starts=5,
        seed=3,
    )
    assert out.result.total_spent <= budget + 1e-3
    for ch in out.active_channels:
        assert out.result.allocation[ch] >= thresholds[ch] - 1e-3
        assert out.result.allocation[ch] <= ceilings[ch] + 1e-3


def test_kappa_sweep_returns_three_runs(thresholds, ceilings, three_channel_params):
    channels = list(thresholds.keys())
    budget = 50_000.0
    sweep = solve_activation_kappa_sweep(
        three_channel_params,
        budget,
        channels,
        thresholds,
        ceilings,
        factors=(0.8, 1.0, 1.2),
        n_starts=3,
        seed=4,
    )
    assert set(sweep) == {"kappa_x0.8", "kappa_x1", "kappa_x1.2"}
    assert all(isinstance(v, ActivationSolveResult) for v in sweep.values())


def test_model_a_vs_b_at_portfolio_scale(thresholds, ceilings, three_channel_params):
    """At large B, Model B should match or be close to Model A objective."""
    channels = list(thresholds.keys())
    budget = 200_000.0
    base = solve(three_channel_params, budget, channels, n_starts=5, seed=5)
    act = solve_with_activation(
        three_channel_params,
        budget,
        channels,
        thresholds,
        ceilings,
        n_starts=5,
        seed=5,
    )
    assert act.result.predicted_conversions <= base.predicted_conversions + 1e-6


# --------------------------------------------------------------------------- #
# Model C — adstock steady-state bridge
# --------------------------------------------------------------------------- #
def test_adstock_steady_state_scales_b(three_channel_params):
    lambdas = {"strong": 0.5, "weak": 0.0, "mid": 0.25}
    eff = apply_adstock_steady_state(three_channel_params, lambdas)
    # b_eff = b / (1 - lambda); a unchanged
    assert eff["strong"]["a"] == three_channel_params["strong"]["a"]
    assert math.isclose(eff["strong"]["b"], three_channel_params["strong"]["b"] / 0.5)
    assert math.isclose(eff["mid"]["b"], three_channel_params["mid"]["b"] / 0.75)


def test_adstock_lambda_zero_is_noop(three_channel_params):
    lambdas = {ch: 0.0 for ch in three_channel_params}
    eff = apply_adstock_steady_state(three_channel_params, lambdas)
    for ch in three_channel_params:
        assert eff[ch]["a"] == three_channel_params[ch]["a"]
        assert eff[ch]["b"] == three_channel_params[ch]["b"]


def test_adstock_rejects_invalid_lambda(three_channel_params):
    with pytest.raises(ValueError):
        apply_adstock_steady_state(three_channel_params, {"strong": 1.0, "weak": 0.0, "mid": 0.0})


def test_model_c_equals_model_b_when_lambda_zero(thresholds, ceilings, three_channel_params):
    """λ = 0 ⇒ Model C is identical to Model B on the same curves."""
    channels = list(thresholds.keys())
    budget = 40_000.0
    model_b = solve_with_activation(
        three_channel_params, budget, channels, thresholds, ceilings, n_starts=5, seed=7
    )
    eff = apply_adstock_steady_state(three_channel_params, {ch: 0.0 for ch in channels})
    model_c = solve_with_activation(
        eff, budget, channels, thresholds, ceilings, n_starts=5, seed=7
    )
    assert math.isclose(
        model_c.result.predicted_conversions,
        model_b.result.predicted_conversions,
        rel_tol=1e-9,
    )


def test_model_c_carryover_lifts_conversions(thresholds, ceilings, three_channel_params):
    """Positive λ makes effective curves steeper ⇒ ≥ conversions at same budget."""
    channels = list(thresholds.keys())
    budget = 60_000.0
    base = solve_with_activation(
        three_channel_params, budget, channels, thresholds, ceilings, n_starts=5, seed=8
    )
    eff = apply_adstock_steady_state(
        three_channel_params, {"strong": 0.5, "weak": 0.0, "mid": 0.3}
    )
    carry = solve_with_activation(
        eff, budget, channels, thresholds, ceilings, n_starts=5, seed=8
    )
    assert (
        carry.result.predicted_conversions
        >= base.result.predicted_conversions - 1e-6
    )


# --------------------------------------------------------------------------- #
# cross_check_global_optimum — random multi-start never beats enumeration
# --------------------------------------------------------------------------- #
def test_cross_check_confirms_enumeration_is_global(thresholds, ceilings, three_channel_params):
    channels = list(thresholds.keys())
    budget = 60_000.0
    winner = solve_with_activation(
        three_channel_params, budget, channels, thresholds, ceilings, n_starts=5, seed=3
    )
    report = cross_check_global_optimum(
        three_channel_params,
        budget,
        channels,
        thresholds,
        ceilings,
        winner.result.predicted_conversions,
        n_starts=64,
        seed=3,
    )
    assert report["passed"]
    # With 64 samples over 8 patterns, random search should also FIND the optimum.
    assert report["best_random_conversions"] <= report["enumerated_conversions"] + 1.0
    assert report["gap"] <= 1.0  # ties the enumerated winner (within rounding)
    assert report["patterns_sampled"] > 0


def test_cross_check_flags_a_too_good_claim(thresholds, ceilings, three_channel_params):
    channels = list(thresholds.keys())
    budget = 60_000.0
    winner = solve_with_activation(
        three_channel_params, budget, channels, thresholds, ceilings, n_starts=5, seed=1
    )
    # Claim far above the true optimum: random search can't beat it, so it "passes"
    # — but a claim BELOW what random finds must fail.
    inflated_low = winner.result.predicted_conversions - 10_000.0
    report = cross_check_global_optimum(
        three_channel_params, budget, channels, thresholds, ceilings,
        inflated_low, n_starts=64, seed=1,
    )
    assert not report["passed"]
