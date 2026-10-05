
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

try:  # works whether imported as src.mmm_model or run from notebooks/
    from .data_prep import load_config, resolve_project_path
except ImportError:  # pragma: no cover
    from data_prep import load_config, resolve_project_path

logger = logging.getLogger(__name__)

# Spend is fitted in MONTHLY USD by default. Meghna's budget B is monthly (~$3.5M).
DEFAULT_FREQ = "monthly"  # "monthly" or "daily"
_FREQ_RULE = {"monthly": "MS", "weekly": "W", "daily": "D"}
# Minimum covered days to keep a resampled period (drop partial edge periods).
_MIN_DAYS = {"monthly": 20, "weekly": 5, "daily": 1}

# Light log-space ridge weight on the (scaled) saturation rate b, as a
# fraction of RMS target. Breaks flat directions in the joint fit; ~0 bias.
REG_B_WEIGHT = 0.05


# --------------------------------------------------------------------------- #
# Core model
# --------------------------------------------------------------------------- #
def saturation_curve(spend, a: float, b: float):
    """Saturation response: a * (1 - exp(-b * spend)).

    Works on scalars or numpy arrays. Monotonically increasing and concave in
    spend for a > 0, b > 0, so more spend always means more predicted
    conversions (the optimizer's monotonicity requirement).
    """
    spend = np.asarray(spend, dtype="float64")
    return a * (1.0 - np.exp(-b * spend))


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _resolve_spend_columns(
    df: pd.DataFrame,
    config: dict,
    channels: list[str],
    spend_mode: str,
) -> dict[str, str]:
    """Pick raw or adstock column per channel based on availability."""
    column_map = config.get("column_map", {})
    out: dict[str, str] = {}
    for ch in channels:
        if spend_mode == "adstock":
            adstock_col = f"{ch}_adstock"
            if adstock_col in df.columns:
                out[ch] = adstock_col
                continue
        raw = column_map.get(ch, ch.upper())
        out[ch] = raw
    return out


def aggregate_portfolio(
    df: pd.DataFrame,
    config: dict,
    channels: list[str],
    target: str = "y",
    freq: str = DEFAULT_FREQ,
    spend_mode: str = "raw",
) -> pd.DataFrame:
    """Collapse the per-timeseries train frame into ONE portfolio time series.

    Steps:
        1. Sum each channel's spend and the target across all timeseries per
           calendar day (portfolio = all brands/series combined).
        2. Resample to ``freq`` (monthly by default) summing spend and target,
           so x and the budget B share a time unit.

    Returns a frame indexed by period with one column per channel (the channel
    KEY, not the CSV column) plus the target column.
    """
    date_col = config["data"]["date_column"]
    spend_cols = _resolve_spend_columns(df, config, channels, spend_mode)

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])

    needed = list(spend_cols.values()) + [target]
    missing = [c for c in needed if c not in work.columns]
    if missing:
        raise KeyError(
            f"Train frame is missing required columns {missing}. "
            f"Available: {list(work.columns)}"
        )

    # 1) daily portfolio: sum across all timeseries
    daily = work.groupby(date_col)[needed].sum().sort_index()

    # 2) resample to requested frequency
    rule = _FREQ_RULE[freq]
    agg = daily.resample(rule).sum()

    # Drop partial edge periods (a half-month / part-week of coverage would bias
    # the level downward). Keep periods with at least _MIN_DAYS covered days.
    min_days = _MIN_DAYS.get(freq, 1)
    if min_days > 1:
        day_counts = daily.assign(_one=1).resample(rule)["_one"].sum()
        agg = agg[day_counts >= min_days]

    # rename channel spend columns to channel keys
    rename = {raw: ch for ch, raw in spend_cols.items()}
    agg = agg.rename(columns=rename)
    agg = agg.rename(columns={target: "y"})
    return agg[list(channels) + ["y"]]


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_channel(
    train_df: pd.DataFrame,
    channel: str,
    target: str = "y",
    config: dict | None = None,
    freq: str = DEFAULT_FREQ,
) -> dict:
    """Fit saturation parameters for ONE channel (single-channel NLS).

    Used for the quick "fit one channel -> send a sample" smoke test. Because it
    regresses a single curve against TOTAL conversions, its ``a`` is biased high
    (it absorbs other channels' contribution). For the real hand-off use
    ``fit_all_channels`` (joint), which keeps the five curves on the same scale.
    """
    config = config or load_config()
    agg = aggregate_portfolio(train_df, config, [channel], target=target, freq=freq)
    x = agg[channel].to_numpy(dtype="float64")
    y = agg["y"].to_numpy(dtype="float64")
    a, b = _fit_one(x, y)
    return {"a": float(a), "b": float(b)}


def _fit_one(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Single-channel nonlinear least squares with positivity + internal scaling."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    x_scale = max(np.nanmean(x), 1.0)  # keep b ~ O(1) for the solver
    xs = x / x_scale

    a0 = max(np.nanmax(y), 1.0)
    p0 = [a0, 1.0]
    try:
        popt, _ = curve_fit(
            lambda xv, a, b: a * (1.0 - np.exp(-b * xv)),
            xs, y, p0=p0,
            bounds=([1e-9, 1e-9], [np.inf, np.inf]),
            maxfev=20000,
        )
        a, b_scaled = popt
    except Exception as exc:  # fall back to a robust default
        logger.warning("curve_fit failed (%s); using fallback init", exc)
        a, b_scaled = a0, 1.0
    b = b_scaled / x_scale  # convert back to per-USD units
    return float(a), float(b)


def fit_all_channels(
    train_df: pd.DataFrame,
    channels: list[str],
    target: str = "y",
    config: dict | None = None,
    freq: str = DEFAULT_FREQ,
    fit_baseline: bool = True,
    return_baseline: bool = False,
    reg_b_weight: float | None = None,
    spend_mode: str = "raw",
):
    """Fit all channels JOINTLY and return {channel: {"a", "b"}}.

    Solves
        min over {c>=0, a_i, b_i}  sum_t ( c + sum_i a_i(1 - e^{-b_i x_{i,t}}) - y_t )^2
                                    + light regularization (see below)
        s.t. a_i > 0, b_i > 0
    so the five saturation curves *together* (plus a baseline) reproduce total
    monthly conversions.

    BASELINE (important): ``y`` = ALL_PURCHASES includes a large non-paid base
    (organic, direct, email, branded search) that the five paid channels do NOT
    cause. We fit a non-negative intercept ``c`` to absorb it so the curves only
    explain the spend-driven *increment*. ``c`` is NOT exported: it is constant
    with respect to the allocation, so it does not change Meghna's argmax
    (maximizing sum_i a_i(1-e^{-b_i x_i}) s.t. sum x_i <= B). Without the
    intercept the curves are forced to explain the baseline and the per-channel
    a/b are badly distorted (some channels collapse to a ~ 0).

    Why joint (not 5 independent fits): ``y`` is total, not per channel; fitting
    each channel against total y would push every ceiling toward the portfolio
    total and overstate conversions ~5x.

    Identifiability: with few monthly points and partly-correlated channel spend,
    an unconstrained joint fit has flat directions it escapes into (the
    degenerate "linear" basin: one channel a -> huge, b -> ~0). We tame it with
    internal per-channel spend scaling (b ~ O(1) for the solver), bounds (a
    capped, b floored), a light log-space ridge pulling each scaled b toward 1,
    and multi-start. Per-channel a/b should still be read as directional, not
    precise, when channel spends move together.
    """
    config = config or load_config()
    agg = aggregate_portfolio(
        train_df, config, channels, target=target, freq=freq, spend_mode=spend_mode
    )
    X = agg[channels].to_numpy(dtype="float64")        # (T, n)
    y = agg["y"].to_numpy(dtype="float64")             # (T,)
    n = len(channels)

    # Per-channel spend scaling keeps each b ~ O(1) during optimization.
    x_scale = np.maximum(X.mean(axis=0), 1.0)          # (n,)
    Xs = X / x_scale

    y_rms = float(np.sqrt(np.mean(y**2))) or 1.0
    w_b = (reg_b_weight if reg_b_weight is not None else REG_B_WEIGHT) * y_rms

    def split(p):
        if fit_baseline:
            return p[0], p[1:1 + n], p[1 + n:]
        return 0.0, p[:n], p[n:]

    def residuals(p):
        c, a, b = split(p)
        pred = c + (a * (1.0 - np.exp(-b * Xs))).sum(axis=1)
        data_res = pred - y
        reg_res = w_b * np.log(b)          # pull scaled b toward 1 (log b -> 0)
        return np.concatenate([data_res, reg_res])

    # Bounds: cap ceilings, floor the (scaled) saturation rate. b ~ O(1) here.
    y_peak = max(float(np.nanmax(y)), 1.0)
    a_max = 2.0 * y_peak
    lb_ab = np.array([1e-6] * n + [1e-2] * n)
    ub_ab = np.array([a_max] * n + [1e2] * n)
    if fit_baseline:
        lb = np.concatenate([[0.0], lb_ab])
        ub = np.concatenate([[y_peak], ub_ab])
    else:
        lb, ub = lb_ab, ub_ab

    # Multi-start: several inits, keep the lowest-cost convergent fit.
    rng = np.random.default_rng(0)
    starts = []
    base0 = float(np.median(y)) * 0.5
    even = max(y_peak / n, 1.0)

    def pack(c, a, b):
        return np.concatenate([[c], a, b]) if fit_baseline else np.concatenate([a, b])

    starts.append(pack(base0, np.full(n, even), np.ones(n)))
    share = X.mean(axis=0) / max(X.mean(axis=0).sum(), 1.0)
    starts.append(pack(base0, np.clip(y_peak * share, 1.0, a_max), np.full(n, 0.8)))
    for _ in range(14):
        a_init = np.clip(rng.uniform(0.3, 1.2, n) * even, 1.0, a_max)
        b_init = rng.uniform(0.3, 2.5, n)
        c_init = rng.uniform(0.0, 0.8) * y_peak if fit_baseline else 0.0
        starts.append(pack(c_init, a_init, b_init))

    best = None
    for p0 in starts:
        p0 = np.clip(p0, lb + 1e-9, ub - 1e-9)
        try:
            sol = least_squares(residuals, p0, bounds=(lb, ub), max_nfev=50000)
        except Exception as exc:  # pragma: no cover
            logger.warning("least_squares start failed: %s", exc)
            continue
        if best is None or sol.cost < best.cost:
            best = sol

    if best is None:
        raise RuntimeError("Joint fit failed for all starts")

    c_fit, a_fit, b_scaled = split(best.x)
    b_fit = b_scaled / x_scale  # back to per-USD

    params = {
        ch: {"a": float(a_fit[i]), "b": float(b_fit[i])}
        for i, ch in enumerate(channels)
    }
    logger.info("Joint fit (multi-start) converged=%s cost=%.4g baseline=%.0f",
                best.success, best.cost, float(c_fit))
    if return_baseline:
        return params, float(c_fit)
    return params


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_params(params: dict, output_path: str) -> str:
    """Write channel parameters to JSON in Meghna's exact format. Returns path."""
    path = resolve_project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        ch: {"a": float(v["a"]), "b": float(v["b"])} for ch, v in params.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    logger.info("Wrote %d channel params -> %s", len(clean), path)
    return str(path)


# --------------------------------------------------------------------------- #
# Quality checks
# --------------------------------------------------------------------------- #
def quality_checks(
    params: dict,
    train_df: pd.DataFrame,
    channels: list[str],
    config: dict | None = None,
    freq: str = DEFAULT_FREQ,
    spend_mode: str = "raw",
) -> dict:
    """Run Meghna's pre-handoff checks. Returns a report dict (also logs)."""
    config = config or load_config()
    agg = aggregate_portfolio(
        train_df, config, channels, target="y", freq=freq, spend_mode=spend_mode
    )
    report: dict = {"channels": {}, "flags": [], "all_positive": True, "monotonic": True}

    for ch in channels:
        a = params[ch]["a"]
        b = params[ch]["b"]
        x = agg[ch].to_numpy(dtype="float64")
        x_typ = float(np.nanmedian(x[x > 0])) if np.any(x > 0) else 0.0
        # saturation level at typical spend: fraction of ceiling reached
        sat = float(1.0 - np.exp(-b * x_typ)) if x_typ > 0 else 0.0
        contribution = float(a * sat)  # predicted conversions at typical spend
        ok_pos = a > 0 and b > 0
        report["all_positive"] = report["all_positive"] and ok_pos
        # extreme-b heuristic: b*x_typ in a healthy band (not ~linear, not instant)
        bx = b * x_typ
        extreme = bx < 1e-3 or bx > 50
        report["channels"][ch] = {
            "a": a, "b": b, "x_typical": x_typ,
            "saturation_at_typical": sat, "contribution_at_typical": contribution,
            "b_times_x_typical": bx, "extreme_b": extreme,
        }
        if extreme:
            report["flags"].append(
                f"{ch}: b*x_typical={bx:.3g} outside healthy band [1e-3, 50]"
            )

    # weak-channel + under-saturation flags. With collinear channel spend, the
    # joint fit can drive a correlated channel's contribution toward zero (its
    # effect is absorbed by a correlated neighbour) -> low share. A near-linear
    # fit (b*x_typical small) means the ceiling is an extrapolation, so the
    # optimizer will load that channel up; a channel cap may be warranted.
    total_contrib = sum(v["contribution_at_typical"] for v in report["channels"].values()) or 1.0
    for ch, v in report["channels"].items():
        share = v["contribution_at_typical"] / total_contrib
        v["contribution_share"] = share
        v["under_saturated"] = bool(v["b_times_x_typical"] < 0.2)
        if share < 0.02 or v["a"] < 1.0:
            report["flags"].append(
                f"{ch} weak: contribution share {share:.2%} (likely collinear / "
                f"low incremental) - consider dropping or capping in config"
            )
        elif v["under_saturated"]:
            report["flags"].append(
                f"{ch} under-saturated: b*x_typical={v['b_times_x_typical']:.2f} "
                f"(near-linear; ceiling a={v['a']:,.0f} is an extrapolation) - "
                f"consider a channel cap"
            )
    return report


# --------------------------------------------------------------------------- #
# Evaluation on holdout
# --------------------------------------------------------------------------- #
def evaluate_on_test(
    test_df: pd.DataFrame,
    params: dict,
    config: dict | None = None,
    freq: str = DEFAULT_FREQ,
    spend_mode: str = "raw",
) -> dict:
    """Evaluate fitted model on the holdout test split (portfolio, monthly).

    Predicts total conversions per period as sum_i a_i(1 - e^{-b_i x_i}) and
    compares to actual y. Returns R^2, RMSE, MAE, MAPE and n periods.
    """
    config = config or load_config()
    channels = list(params.keys())
    agg = aggregate_portfolio(
        test_df, config, channels, target="y", freq=freq, spend_mode=spend_mode
    )
    if len(agg) == 0:
        return {"n": 0, "note": "no test periods after aggregation"}

    pred = np.zeros(len(agg))
    for ch in channels:
        pred += saturation_curve(agg[ch].to_numpy(), params[ch]["a"], params[ch]["b"])
    y = agg["y"].to_numpy(dtype="float64")

    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-9
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    mask = y != 0
    mape = float(np.mean(np.abs(resid[mask] / y[mask])) * 100) if mask.any() else float("nan")
    return {
        "n": int(len(agg)),
        "r2": float(1 - ss_res / ss_tot),
        "rmse": rmse,
        "mae": mae,
        "mape_pct": mape,
        "freq": freq,
    }


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def resolve_mmm_freq(config: dict, freq: str | None = None) -> str:
    """Return explicit ``freq`` or ``mmm.freq`` from config (default monthly)."""
    if freq is not None:
        return freq
    return str(config.get("mmm", {}).get("freq", DEFAULT_FREQ))


def run_fitting(
    config_path: str = "config.yaml",
    freq: str | None = None,
    *,
    reg_b_weight: float | None = None,
    adstock_decay_overrides: dict[str, float] | None = None,
    params_output_path: str | None = None,
) -> dict:
    """Run the full MMM fitting pipeline and write channel_params JSON.

    Optional BO hooks:
    - ``reg_b_weight``: ridge on log-scaled b in joint fit
    - ``adstock_decay_overrides``: re-apply adstock on train frame before fit
      (uses adstock spend columns when overrides are provided)
    - ``params_output_path``: override default JSON path (e.g. channel_params_bo.json)
    """
    config = load_config(config_path)
    freq = resolve_mmm_freq(config, freq)
    channels = config["channels"]["modeled"]
    target = "y"

    train_path = resolve_project_path(config["data"]["train_path"])
    train_df = pd.read_csv(train_path)

    spend_mode = "raw"
    if adstock_decay_overrides:
        from src.data_prep import apply_adstock

        decay_rates = {**config["adstock"]["decay_rates"], **adstock_decay_overrides}
        train_df = apply_adstock(train_df.copy(), decay_rates, config)
        spend_mode = "adstock"

    params, baseline = fit_all_channels(
        train_df,
        channels,
        target=target,
        config=config,
        freq=freq,
        return_baseline=True,
        reg_b_weight=reg_b_weight,
        spend_mode=spend_mode,
    )
    out_path = export_params(
        params,
        params_output_path or config["data"]["params_path"],
    )
    report = quality_checks(
        params, train_df, channels, config=config, freq=freq, spend_mode=spend_mode
    )

    result = {
        "params": params,
        "params_path": out_path,
        "freq": freq,
        "quality_report": report,
        "baseline_monthly": baseline,
        "reg_b_weight": reg_b_weight if reg_b_weight is not None else REG_B_WEIGHT,
        "spend_mode": spend_mode,
    }

    test_path = resolve_project_path(config["data"]["test_path"])
    if Path(test_path).exists():
        try:
            test_df = pd.read_csv(test_path)
            if adstock_decay_overrides:
                from src.data_prep import apply_adstock

                decay_rates = {**config["adstock"]["decay_rates"], **adstock_decay_overrides}
                test_df = apply_adstock(test_df.copy(), decay_rates, config)
            result["test_metrics"] = evaluate_on_test(
                test_df, params, config=config, freq=freq, spend_mode=spend_mode
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("evaluate_on_test skipped: %s", exc)
    return result


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    res = run_fitting()
    print(json.dumps(res["params"], indent=2))
