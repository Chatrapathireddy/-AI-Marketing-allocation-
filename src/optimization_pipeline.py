
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.baseline import apply_baseline_to_result
from src.bo_mmm_tuning import load_bo_params
from src.data_prep import load_config, resolve_project_path
from src.mmm_model import resolve_mmm_freq, run_fitting
from src.optimizer import (
    ActivationSolveResult,
    OptimResult,
    apply_adstock_steady_state,
    load_activation_from_config,
    solve,
    solve_with_activation,
)


def load_activation_thresholds(config: dict | None = None) -> dict[str, float]:
    """Load per-channel activation κ (USD/week) from config for Models B/C."""
    thresholds, _ = load_activation_from_config(config)
    return thresholds


def load_activation_ceilings(config: dict | None = None) -> dict[str, float]:
    """Load per-channel activation u_c (USD/week) from config."""
    _, ceilings = load_activation_from_config(config)
    return ceilings


def run_model_b(
    params: dict,
    budget: float,
    channels: list[str],
    config: dict | None = None,
) -> ActivationSolveResult:
    """Model B: activation-threshold enumeration at portfolio budget."""
    cfg = config or load_config()
    thresholds, ceilings = load_activation_from_config(cfg)
    return solve_with_activation(params, budget, channels, thresholds, ceilings)


def load_model_c_inputs(
    config: dict | None = None,
) -> tuple[dict | None, dict[str, float] | None]:
    """Load Greg's adstock curves + per-channel λ for Model C.

    Returns ``(channel_params_C, adstock_lambdas)`` or ``(None, None)`` when the
    files are not present yet (Model C stays disabled, no crash).
    """
    cfg = config or load_config()
    mc = cfg.get("model_c") or {}
    params_path = mc.get("channel_params_path", "data/processed/channel_params_C.json")
    lambdas_path = mc.get("lambdas_path", "data/processed/adstock_lambdas.json")
    p_params = resolve_project_path(params_path)
    p_lambdas = resolve_project_path(lambdas_path)
    if not (Path(p_params).exists() and Path(p_lambdas).exists()):
        return None, None
    with open(p_params, encoding="utf-8") as f:
        params_c = json.load(f)
    with open(p_lambdas, encoding="utf-8") as f:
        lambdas = {k: float(v) for k, v in json.load(f).items()}
    return params_c, lambdas


def run_model_c(
    budget: float,
    channels: list[str],
    config: dict | None = None,
) -> ActivationSolveResult | None:
    """Model C: adstock + activation.

    Folds carryover into effective curves (``b_eff = b / (1 - λ)``) and reuses
    the same 32-pattern activation enumeration as Model B. Returns ``None`` if
    Greg's Model C inputs are not available yet.
    """
    cfg = config or load_config()
    params_c, lambdas = load_model_c_inputs(cfg)
    if params_c is None or lambdas is None:
        return None
    effective = apply_adstock_steady_state(params_c, lambdas)
    thresholds, ceilings = load_activation_from_config(cfg)
    return solve_with_activation(effective, budget, channels, thresholds, ceilings)


def optimizer_fn_for_sensitivity(params: dict, channels: list[str], budget: float) -> dict:
    """Return dict shape expected by explainer.run_sensitivity."""
    result = solve(params, budget, channels)
    return {
        "allocation": result.allocation,
        "predicted_conversions": result.predicted_conversions,
    }


def run_optimization_pipeline(
    *,
    confirmed_budget: float | None = None,
    detected_budget: float | None = None,
    channel_params: dict | None = None,
    train_df: pd.DataFrame | None = None,
    run_model_b_solve: bool = True,
    run_model_c_solve: bool = True,
) -> tuple[
    OptimResult,
    dict[str, Any],
    float,
    ActivationSolveResult | None,
    ActivationSolveResult | None,
]:
    """Fit MMM (if needed), run Models A + B (+ C), attach baseline lift."""
    config = load_config()
    mmm_freq = resolve_mmm_freq(config)
    if not channel_params:
        bo_params = load_bo_params(config)
        if bo_params is not None:
            channel_params = bo_params
        else:
            fitting = run_fitting(freq=mmm_freq)
            channel_params = fitting["params"]

    channels = list(config["channels"]["modeled"])
    budget = float(
        confirmed_budget
        or detected_budget
        or config["optimization"]["default_budget"]
    )
    optim = solve(channel_params, budget, channels)

    model_b: ActivationSolveResult | None = None
    if run_model_b_solve:
        try:
            model_b = run_model_b(channel_params, budget, channels, config=config)
        except (KeyError, ValueError, RuntimeError):
            model_b = None

    model_c: ActivationSolveResult | None = None
    if run_model_c_solve and config.get("model_c", {}).get("enabled", True):
        try:
            model_c = run_model_c(budget, channels, config=config)
        except (KeyError, ValueError, RuntimeError):
            model_c = None

    if train_df is None:
        train_path = resolve_project_path(config["data"]["train_path"])
        if Path(train_path).exists():
            train_df = pd.read_csv(train_path)

    if train_df is not None and len(train_df) > 0:
        optim = apply_baseline_to_result(
            optim, channel_params, channels, budget, train_df, config=config
        )

    return optim, channel_params, budget, model_b, model_c


def apply_optimization_to_session(
    session_state: Any,
    optim: OptimResult,
    channel_params: dict,
    model_b: ActivationSolveResult | None = None,
    model_c: ActivationSolveResult | None = None,
) -> None:
    """Persist optimizer outputs on Streamlit session_state."""
    config = load_config()
    session_state.channel_params = channel_params
    session_state.optim_result = optim
    session_state.optimizer_fn = optimizer_fn_for_sensitivity
    session_state.activation_thresholds = load_activation_thresholds(config)
    try:
        session_state.activation_ceilings = load_activation_ceilings(config)
    except KeyError:
        session_state.activation_ceilings = {}
    session_state.optim_result_B = model_b.result if model_b is not None else None
    session_state.optim_result_C = model_c.result if model_c is not None else None
    # Model C inputs for Vikhyat's Page 6 (adstock decay viz + comparison).
    params_c, lambdas = load_model_c_inputs(config)
    if params_c is not None:
        session_state.channel_params_C = params_c
    if lambdas is not None:
        session_state.adstock_lambdas = lambdas
    session_state.optimization_complete = True
    session_state.phase = "explore"


def _session_value(session_state: Any, key: str, default=None):
    if hasattr(session_state, "get"):
        try:
            return session_state.get(key, default)
        except TypeError:
            pass
    return getattr(session_state, key, default)


def maybe_resolve(session_state: Any, *, force: bool = False) -> OptimResult | None:
    """Re-solve Models A + B + C from CURRENT session-state parameters.

    Called by the app after the agent mutates κ/u_c/B/λ (params_dirty). Uses the
    already-fitted ``channel_params`` (κ/B/u_c/λ changes do not require an MMM
    refit) and the parameterized solvers — no module constants in the solve path.
    """
    if not force and not _session_value(session_state, "params_dirty", False):
        return None

    channel_params = _session_value(session_state, "channel_params", None)
    if not channel_params:
        return None

    config = load_config()
    channels = list(config["channels"]["modeled"])
    budget = float(
        _session_value(session_state, "confirmed_budget", None)
        or config["optimization"]["default_budget"]
    )

    optim = solve(channel_params, budget, channels)

    thresholds = _session_value(session_state, "activation_thresholds", None) or {}
    ceilings = _session_value(session_state, "activation_ceilings", None) or {}
    activation_ready = bool(thresholds and ceilings and all(c in ceilings for c in channels))

    model_b: ActivationSolveResult | None = None
    if activation_ready:
        try:
            model_b = solve_with_activation(
                channel_params, budget, channels, thresholds, ceilings
            )
        except (KeyError, ValueError, RuntimeError):
            model_b = None

    # Model C: re-solve on adstock-adjusted curves using current λ (which the
    # agent may have changed). Prefer session inputs, fall back to Greg's files.
    params_c = _session_value(session_state, "channel_params_C", None)
    lambdas = _session_value(session_state, "adstock_lambdas", None)
    if not params_c or not lambdas:
        file_params_c, file_lambdas = load_model_c_inputs(config)
        params_c = params_c or file_params_c
        lambdas = lambdas or file_lambdas
    model_c: ActivationSolveResult | None = None
    if activation_ready and params_c and lambdas:
        try:
            effective = apply_adstock_steady_state(params_c, lambdas)
            model_c = solve_with_activation(
                effective, budget, channels, thresholds, ceilings
            )
        except (KeyError, ValueError, RuntimeError):
            model_c = None

    session_state.optim_result = optim
    session_state.optim_result_B = model_b.result if model_b is not None else None
    session_state.optim_result_C = model_c.result if model_c is not None else None
    session_state.params_dirty = False
    return optim
