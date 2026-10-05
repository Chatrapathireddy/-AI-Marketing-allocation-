
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data_prep import load_config, resolve_project_path
from src.optimizer import OptimResult, load_params, predicted_conversions, solve

logger = logging.getLogger(__name__)


def _spend_column(df: pd.DataFrame, channel: str, config: dict) -> str | None:
    """Resolve adstock or raw spend column for a modeled channel key."""
    adstock = f"{channel}_adstock"
    if adstock in df.columns:
        return adstock
    raw = config.get("column_map", {}).get(channel)
    if raw and raw in df.columns:
        return raw
    return None


def compute_historical_baseline(
    df: pd.DataFrame,
    channels: list[str],
    budget: float,
    config: dict | None = None,
) -> dict[str, float]:
    """Historical average spend mix scaled to ``budget`` (proportional baseline)."""
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if not channels:
        raise ValueError("channels must be a non-empty list")

    config = config or load_config()
    means: dict[str, float] = {}
    for ch in channels:
        col = _spend_column(df, ch, config)
        means[ch] = float(df[col].fillna(0).mean()) if col else 0.0

    total = sum(means.values())
    if total <= 0:
        logger.warning("Historical spend sums to zero — falling back to equal split baseline")
        return compute_equal_baseline(budget, channels)

    return {ch: (means[ch] / total) * budget for ch in channels}


def compute_equal_baseline(budget: float, channels: list[str]) -> dict[str, float]:
    """Equal split: budget / N per modeled channel."""
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if not channels:
        raise ValueError("channels must be a non-empty list")
    share = budget / len(channels)
    return {ch: share for ch in channels}


def compute_lift(
    optimized: dict[str, float],
    baseline: dict[str, float],
    params: dict,
    channels: list[str],
) -> dict[str, float]:
    """Compare predicted conversions at optimized vs baseline allocations."""
    if not channels:
        raise ValueError("channels must be a non-empty list")

    opt_x = [float(optimized.get(ch, 0.0)) for ch in channels]
    base_x = [float(baseline.get(ch, 0.0)) for ch in channels]
    optimized_conversions = predicted_conversions(opt_x, params, channels)
    baseline_conversions = predicted_conversions(base_x, params, channels)

    if baseline_conversions > 0:
        lift_pct = (optimized_conversions - baseline_conversions) / baseline_conversions * 100.0
    else:
        lift_pct = 0.0

    return {
        "optimized_conversions": optimized_conversions,
        "baseline_conversions": baseline_conversions,
        "lift_pct": lift_pct,
    }


def apply_baseline_to_result(
    optim: OptimResult,
    params: dict,
    channels: list[str],
    budget: float,
    df: pd.DataFrame,
    *,
    config: dict | None = None,
    method: str = "historical",
) -> OptimResult:
    """Fill baseline fields on ``OptimResult`` using train/historical data."""
    config = config or load_config()
    if method == "equal":
        baseline_alloc = compute_equal_baseline(budget, channels)
    elif method == "historical":
        baseline_alloc = compute_historical_baseline(df, channels, budget, config)
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    lift = compute_lift(optim.allocation, baseline_alloc, params, channels)
    optim.baseline_allocation = baseline_alloc
    optim.baseline_conversions = lift["baseline_conversions"]
    optim.lift_pct = lift["lift_pct"]
    return optim


def run_all_baselines(config_path: str = "config.yaml") -> dict:
    """Run historical and equal baselines; compare to default-budget optimizer."""
    config = load_config(config_path)
    channels = list(config["channels"]["modeled"])
    budget = float(config["optimization"]["default_budget"])

    train_path = resolve_project_path(config["data"]["train_path"])
    if not Path(train_path).exists():
        raise FileNotFoundError(f"Train split not found: {train_path}")

    params_path = resolve_project_path(config["data"]["params_path"])
    if not Path(params_path).exists():
        raise FileNotFoundError(f"Channel params not found: {params_path}")

    train_df = pd.read_csv(train_path)
    params = load_params(params_path)
    optim = solve(params, budget, channels)

    historical_alloc = compute_historical_baseline(train_df, channels, budget, config)
    equal_alloc = compute_equal_baseline(budget, channels)

    return {
        "budget": budget,
        "channels": channels,
        "optimizer": {
            "allocation": optim.allocation,
            **compute_lift(optim.allocation, historical_alloc, params, channels),
        },
        "historical": {
            "allocation": historical_alloc,
            **compute_lift(optim.allocation, historical_alloc, params, channels),
        },
        "equal": {
            "allocation": equal_alloc,
            **compute_lift(optim.allocation, equal_alloc, params, channels),
        },
    }
