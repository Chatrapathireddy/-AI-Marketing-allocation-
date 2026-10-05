
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from src.data_prep import load_config, resolve_project_path

logger = logging.getLogger(__name__)


@dataclass
class OptimResult:
    """Optimization output — shared contract for optimizer, explainer, and Streamlit pages."""

    allocation: dict[str, float]
    predicted_conversions: float
    total_spent: float
    status: str
    kkt_status: str
    lambda_budget: float
    success: bool
    baseline_allocation: dict[str, float] = field(default_factory=dict)
    baseline_conversions: float = 0.0
    lift_pct: float = 0.0
    message: str = ""


@dataclass
class ActivationSolveResult:
    """Model B output: best activation pattern + convex subproblem solution."""

    result: OptimResult
    active_channels: tuple[str, ...]
    patterns_evaluated: int
    patterns_feasible: int
    pattern_mask: tuple[bool, ...]


def _format_status(kkt_status: str, success: bool, message: str) -> str:
    """Human-readable status for Streamlit (page 3 caption).

    Multistart keeps the best point found, so once KKT verifies optimality the
    per-start SLSQP exit flag/message ("Positive directional derivative for
    linesearch", etc.) is not meaningful to the user and only looks alarming.
    """
    if kkt_status == "pass":
        return "KKT pass · optimal allocation found"
    solver = "solver converged" if success else "solver did not fully converge"
    detail = message.strip() if message else ""
    base = f"KKT {kkt_status} · {solver}"
    return f"{base} ({detail})" if detail else base


def _validate_inputs(params: dict, budget: float, channels: list[str]) -> None:
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if not channels:
        raise ValueError("channels must be a non-empty list")
    for ch in channels:
        if ch not in params:
            raise KeyError(f"Missing channel '{ch}' in params")
        a, b = params[ch]["a"], params[ch]["b"]
        if a <= 0 or b <= 0:
            raise ValueError(
                f"Channel '{ch}' requires a > 0 and b > 0; got a={a}, b={b}"
            )


def predicted_conversions(x: list[float] | np.ndarray, params: dict, channels: list[str]) -> float:
    """Total predicted conversions (maximize this)."""
    total = 0.0
    for i, ch in enumerate(channels):
        a = params[ch]["a"]
        b = params[ch]["b"]
        xi = max(float(x[i]), 0.0)
        total += a * (1.0 - np.exp(-b * xi))
    return float(total)


def objective(x: list[float] | np.ndarray, params: dict, channels: list[str]) -> float:
    """Negative total predicted conversions (minimize for SciPy)."""
    return -predicted_conversions(x, params, channels)


def gradient(x: list[float] | np.ndarray, params: dict, channels: list[str]) -> np.ndarray:
    """Gradient of objective (negative marginal conversions per channel)."""
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros(len(channels), dtype=np.float64)
    for i, ch in enumerate(channels):
        a = params[ch]["a"]
        b = params[ch]["b"]
        xi = max(float(x[i]), 0.0)
        grad[i] = -a * b * np.exp(-b * xi)
    return grad


def marginal_conversions(x: float, a: float, b: float) -> float:
    """d/dx [a * (1 - exp(-b*x))] = a * b * exp(-b*x)."""
    return a * b * np.exp(-b * max(x, 0.0))


def _estimate_lambda(
    x: np.ndarray,
    params: dict,
    channels: list[str],
    budget: float,
    caps: dict[str, float | None] | None,
    tol: float,
) -> float:
    """Shadow price of the budget constraint from active-channel marginals."""
    spend = float(np.sum(x))
    if spend >= budget - tol:
        active = []
        for i, ch in enumerate(channels):
            if x[i] <= tol:
                continue
            cap = caps.get(ch) if caps else None
            if cap is not None and x[i] >= cap - tol:
                continue
            active.append(marginal_conversions(x[i], params[ch]["a"], params[ch]["b"]))
        if active:
            return float(np.median(active))
    return 0.0


def verify_kkt(
    result: Any,
    budget: float,
    channels: list[str],
    params: dict | None = None,
    caps: dict[str, float | None] | None = None,
    tol: float = 1e-6,
) -> dict:
    """
    Numeric KKT checks for the budget allocation problem.

    ``result`` may be an OptimResult, a scipy OptimizeResult, or a raw spend vector.
    When ``params`` is omitted, only primal feasibility is checked.
    """
    if hasattr(result, "allocation"):
        x = np.array([result.allocation[ch] for ch in channels], dtype=np.float64)
        lam = result.lambda_budget
    elif hasattr(result, "x"):
        x = np.asarray(result.x, dtype=np.float64)
        lam = None
    else:
        x = np.asarray(result, dtype=np.float64)
        lam = None

    spend = float(np.sum(x))
    # Dollar-scale slack for feasibility. A fixed 1e-9 is tighter than the
    # rounding noise of summing spends at $100k+ budgets, which flags optimal
    # allocations as "over budget". Scale the slack with budget magnitude
    # (≈$0.83 on an $831k budget) while keeping the small tol for tiny budgets.
    budget_tol = max(tol, 1e-6 * max(1.0, budget))
    checks: dict[str, Any] = {
        "budget_feasible": spend <= budget + budget_tol,
        "non_negative": bool(np.all(x >= -tol)),
        "spend_total": spend,
        "budget": budget,
    }

    if caps:
        cap_ok = True
        for i, ch in enumerate(channels):
            cap = caps.get(ch)
            if cap is not None and x[i] > cap + budget_tol:
                cap_ok = False
        checks["caps_feasible"] = cap_ok

    if params is not None:
        if lam is None:
            lam = _estimate_lambda(x, params, channels, budget, caps, tol)
        checks["lambda_budget"] = lam

        binding = spend >= budget - tol
        checks["budget_binding"] = binding

        if binding and lam is not None:
            stationarity_errors = []
            for i, ch in enumerate(channels):
                if x[i] <= tol:
                    continue
                cap = caps.get(ch) if caps else None
                if cap is not None and x[i] >= cap - tol:
                    continue
                mc = marginal_conversions(x[i], params[ch]["a"], params[ch]["b"])
                stationarity_errors.append(abs(mc - lam))
            checks["stationarity_max_error"] = (
                float(max(stationarity_errors)) if stationarity_errors else 0.0
            )
            checks["stationarity_ok"] = checks["stationarity_max_error"] <= max(tol, 1e-3)
        else:
            checks["stationarity_ok"] = lam <= tol if not binding else True
            checks["complementary_slackness_ok"] = (not binding) or (lam > tol)

        if not binding:
            checks["complementary_slackness_ok"] = lam <= max(tol, 1e-3)

    primal_ok = checks["budget_feasible"] and checks["non_negative"]
    if caps and not checks.get("caps_feasible", True):
        primal_ok = False

    if params is None:
        status = "pass" if primal_ok else "fail"
    else:
        kkt_ok = primal_ok and checks.get("stationarity_ok", True)
        if caps:
            kkt_ok = kkt_ok and checks.get("caps_feasible", True)
        if not binding:
            kkt_ok = kkt_ok and checks.get("complementary_slackness_ok", True)
        status = "pass" if kkt_ok else "fail"

    checks["status"] = status
    return checks


def _build_bounds(
    channels: list[str],
    budget: float,
    caps: dict[str, float | None] | None,
    channel_bounds: dict[str, tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    if channel_bounds is not None:
        return [channel_bounds[ch] for ch in channels]
    bounds = []
    for ch in channels:
        upper = budget
        if caps and caps.get(ch) is not None:
            upper = min(upper, float(caps[ch]))
        bounds.append((0.0, upper))
    return bounds


def _initial_guess_bounded(
    bounds: list[tuple[float, float]],
    budget: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Feasible starting point respecting per-channel lower/upper bounds."""
    lows = np.array([b[0] for b in bounds], dtype=np.float64)
    highs = np.array([b[1] for b in bounds], dtype=np.float64)
    x = lows.copy()
    min_commit = float(lows.sum())
    if min_commit > budget + 1e-9:
        return np.clip(lows, lows, highs)
    remaining = budget - min_commit
    if remaining <= 1e-9:
        return x
    slack = np.maximum(highs - lows, 0.0)
    total_slack = float(slack.sum())
    if total_slack <= 1e-12:
        return x
    add = min(remaining, total_slack)
    weights = rng.dirichlet(np.maximum(slack, 1e-12))
    x = lows + weights * add
    return np.clip(x, lows, highs)


def _initial_guess(n: int, budget: float, rng: np.random.Generator) -> np.ndarray:
    weights = rng.dirichlet(np.ones(n))
    return weights * budget


def iter_activation_patterns(n_channels: int) -> list[tuple[bool, ...]]:
    """All 2^n on/off subsets for activation enumeration."""
    return [
        tuple(bool((mask >> i) & 1) for i in range(n_channels))
        for mask in range(2**n_channels)
    ]


def _min_commitment(
    channels: list[str],
    on_mask: tuple[bool, ...],
    thresholds: dict[str, float],
) -> float:
    return sum(thresholds[ch] for ch, on in zip(channels, on_mask) if on)


def _bounds_for_activation_pattern(
    channels: list[str],
    budget: float,
    on_mask: tuple[bool, ...],
    thresholds: dict[str, float],
    ceilings: dict[str, float],
) -> dict[str, tuple[float, float]] | None:
    """Per-channel bounds for one activation pattern; None if infeasible."""
    bounds: dict[str, tuple[float, float]] = {}
    for ch, on in zip(channels, on_mask):
        if not on:
            bounds[ch] = (0.0, 0.0)
            continue
        lo = float(thresholds[ch])
        hi = min(float(ceilings[ch]), budget)
        if lo > hi + 1e-6:
            return None
        bounds[ch] = (lo, hi)
    return bounds


def _pattern_is_feasible(
    channels: list[str],
    on_mask: tuple[bool, ...],
    budget: float,
    thresholds: dict[str, float],
    ceilings: dict[str, float],
) -> bool:
    if _min_commitment(channels, on_mask, thresholds) > budget + 1e-6:
        return False
    return _bounds_for_activation_pattern(channels, budget, on_mask, thresholds, ceilings) is not None


def solve(
    params: dict,
    budget: float,
    channels: list[str],
    caps: dict | None = None,
    channel_bounds: dict[str, tuple[float, float]] | None = None,
    n_starts: int | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
    seed: int = 0,
) -> OptimResult:
    """Run multistart SLSQP to maximize predicted conversions under budget."""
    _validate_inputs(params, budget, channels)
    config = load_config()
    opt_cfg = config.get("optimization", {})
    n_starts = n_starts if n_starts is not None else int(opt_cfg.get("n_starts", 50))
    tol = tol if tol is not None else float(opt_cfg.get("tol", 1e-9))
    max_iter = max_iter if max_iter is not None else int(opt_cfg.get("max_iter", 1000))

    n = len(channels)
    bounds = _build_bounds(channels, budget, caps, channel_bounds=channel_bounds)
    rng = np.random.default_rng(seed)

    def budget_constraint(x: np.ndarray) -> float:
        return budget - float(np.sum(x))

    def budget_jac(x: np.ndarray) -> np.ndarray:
        return -np.ones_like(x)

    constraints = (
        {"type": "ineq", "fun": budget_constraint, "jac": budget_jac},
    )

    starts: list[np.ndarray] = [_initial_guess_bounded(bounds, budget, rng)]
    for _ in range(max(n_starts - 1, 0)):
        starts.append(_initial_guess_bounded(bounds, budget, rng))

    best_x: np.ndarray | None = None
    best_val = np.inf
    best_success = False
    best_message = ""

    for x0 in starts:
        res = minimize(
            objective,
            x0,
            args=(params, channels),
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": tol, "maxiter": max_iter, "disp": False},
        )
        if res.fun < best_val:
            best_val = float(res.fun)
            best_x = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
            best_success = bool(res.success)
            best_message = str(res.message)

    assert best_x is not None
    obj = predicted_conversions(best_x, params, channels)
    effective_caps = caps
    if channel_bounds is not None:
        effective_caps = {
            ch: channel_bounds[ch][1] for ch in channels if channel_bounds[ch][1] < budget
        }
    lam = _estimate_lambda(best_x, params, channels, budget, effective_caps, tol)
    kkt = verify_kkt(best_x, budget, channels, params=params, caps=effective_caps, tol=tol)

    allocation = {ch: float(best_x[i]) for i, ch in enumerate(channels)}
    total_spent = float(sum(allocation.values()))
    kkt_status = str(kkt["status"])
    return OptimResult(
        allocation=allocation,
        predicted_conversions=obj,
        total_spent=total_spent,
        status=_format_status(kkt_status, best_success, best_message),
        kkt_status=kkt_status,
        lambda_budget=lam,
        success=best_success,
        message=best_message,
    )


def solve_with_activation(
    params: dict,
    budget: float,
    channels: list[str],
    thresholds: dict[str, float],
    ceilings: dict[str, float],
    *,
    n_starts: int | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
    seed: int = 0,
) -> ActivationSolveResult:
    """
    Model B: enumerate activation patterns, solve convex subproblem per feasible subset.

    Feasible spend per channel c: {0} union [kappa_c, u_c] when ON.
    """
    _validate_inputs(params, budget, channels)
    for ch in channels:
        if ch not in thresholds or ch not in ceilings:
            raise KeyError(f"Missing activation threshold or ceiling for '{ch}'")
        if thresholds[ch] > ceilings[ch] + 1e-6:
            raise ValueError(
                f"Channel '{ch}': kappa {thresholds[ch]} exceeds ceiling {ceilings[ch]}"
            )

    patterns = iter_activation_patterns(len(channels))
    best: ActivationSolveResult | None = None
    feasible_count = 0

    for pattern_idx, on_mask in enumerate(patterns):
        if not _pattern_is_feasible(channels, on_mask, budget, thresholds, ceilings):
            continue
        feasible_count += 1
        ch_bounds = _bounds_for_activation_pattern(
            channels, budget, on_mask, thresholds, ceilings
        )
        assert ch_bounds is not None
        sub = solve(
            params,
            budget,
            channels,
            channel_bounds=ch_bounds,
            n_starts=n_starts,
            tol=tol,
            max_iter=max_iter,
            seed=seed + pattern_idx,
        )
        active = tuple(ch for ch, on in zip(channels, on_mask) if on)
        candidate = ActivationSolveResult(
            result=sub,
            active_channels=active,
            patterns_evaluated=len(patterns),
            patterns_feasible=0,
            pattern_mask=on_mask,
        )
        if best is None or sub.predicted_conversions > best.result.predicted_conversions:
            best = candidate

    if best is None:
        raise RuntimeError(
            f"No feasible activation pattern for budget {budget}; "
            f"minimum ON commitment exceeds B for all non-empty subsets."
        )

    best.patterns_feasible = feasible_count
    best.patterns_evaluated = len(patterns)
    return best


def solve_activation_kappa_sweep(
    params: dict,
    budget: float,
    channels: list[str],
    thresholds: dict[str, float],
    ceilings: dict[str, float],
    *,
    factors: tuple[float, ...] = (0.8, 1.0, 1.2),
    **solve_kwargs: Any,
) -> dict[str, ActivationSolveResult]:
    """Re-run Model B at scaled kappa values for sensitivity reporting."""
    results: dict[str, ActivationSolveResult] = {}
    for factor in factors:
        scaled = {ch: float(thresholds[ch]) * factor for ch in channels}
        label = f"kappa_x{factor:g}"
        results[label] = solve_with_activation(
            params,
            budget,
            channels,
            scaled,
            ceilings,
            **solve_kwargs,
        )
    return results


def cross_check_global_optimum(
    params: dict,
    budget: float,
    channels: list[str],
    thresholds: dict[str, float],
    ceilings: dict[str, float],
    enumerated_conversions: float,
    *,
    n_starts: int = 64,
    seed: int = 0,
    tol: float | None = None,
    max_iter: int | None = None,
) -> dict[str, Any]:
    """Empirically confirm the enumerated Model B/C winner is globally optimal.

    The activation feasible set ({0} ∪ [κ, u] per channel) is non-convex, so we
    prove optimality by exhaustive enumeration. This is an *independent* sanity
    check: instead of trying all 32 patterns, randomly sample activation patterns,
    locally optimize each with SLSQP, and confirm none beats the enumerated
    objective. A pass means "random multi-start never found a better allocation".

    Returns a small report dict; ``passed`` is True when the enumerated solution
    is at least as good as the best random allocation (within rounding slack).
    """
    rng = np.random.default_rng(seed)
    n = len(channels)
    best_random = 0.0  # all-OFF is always feasible and yields 0 conversions
    patterns_sampled = 0

    for _ in range(max(n_starts, 0)):
        on_mask = tuple(bool(rng.integers(0, 2)) for _ in range(n))
        if not _pattern_is_feasible(channels, on_mask, budget, thresholds, ceilings):
            continue
        ch_bounds = _bounds_for_activation_pattern(
            channels, budget, on_mask, thresholds, ceilings
        )
        assert ch_bounds is not None
        sub = solve(
            params,
            budget,
            channels,
            channel_bounds=ch_bounds,
            n_starts=2,
            tol=tol,
            max_iter=max_iter,
            seed=seed + patterns_sampled + 1,
        )
        patterns_sampled += 1
        if sub.predicted_conversions > best_random:
            best_random = sub.predicted_conversions

    gap = float(enumerated_conversions) - best_random
    denom = max(abs(float(enumerated_conversions)), 1e-9)
    # Slack: a conversion or 1e-6 of the objective, whichever is larger.
    slack = max(1.0, 1e-6 * denom)
    return {
        "enumerated_conversions": float(enumerated_conversions),
        "best_random_conversions": float(best_random),
        "gap": gap,
        "relative_gap": gap / denom,
        "patterns_sampled": patterns_sampled,
        "n_starts": int(n_starts),
        "passed": bool(float(enumerated_conversions) >= best_random - slack),
    }


def load_activation_from_config(config: dict | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """Load kappa and u_c dicts from config activation block."""
    cfg = config or load_config()
    act = cfg.get("activation") or {}
    channels = list(cfg["channels"]["modeled"])
    thresholds_raw = act.get("thresholds") or {}
    ceilings_raw = act.get("ceilings") or {}
    thresholds = {ch: float(thresholds_raw[ch]) for ch in channels if ch in thresholds_raw}
    ceilings = {ch: float(ceilings_raw[ch]) for ch in channels if ch in ceilings_raw}
    missing_t = set(channels) - set(thresholds)
    missing_c = set(channels) - set(ceilings)
    if missing_t:
        raise KeyError(f"Missing activation.thresholds for: {sorted(missing_t)}")
    if missing_c:
        raise KeyError(f"Missing activation.ceilings for: {sorted(missing_c)}")
    return thresholds, ceilings


def apply_adstock_steady_state(
    params: dict,
    lambdas: dict[str, float],
) -> dict:
    """Fold per-channel adstock carryover into effective saturation curves.

    Model C evaluates ``f_c(s_c / (1 - λ_c))`` at steady state. Because the
    decision variable is raw spend ``s_c``, dividing it by ``(1 - λ_c)`` inside
    ``a·(1 - exp(-b·x))`` is identical to keeping raw spend and using a steeper
    rate ``b_eff = b / (1 - λ_c)``. So Model C is just Model A/B run on these
    effective curves — no change to the solver, gradient, KKT, or shadow price.

    λ = 0 is a no-op, so the same params reduce to Model A/B exactly.
    """
    effective: dict = {}
    for ch, p in params.items():
        lam = float(lambdas.get(ch, 0.0))
        if not (0.0 <= lam < 1.0):
            raise ValueError(
                f"Channel '{ch}': adstock lambda must be in [0, 1), got {lam}"
            )
        effective[ch] = {"a": float(p["a"]), "b": float(p["b"]) / (1.0 - lam)}
    return effective


def load_params(path: str | Path) -> dict:
    """Load channel_params.json written by mmm_model."""
    p = resolve_project_path(path)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def solve_from_file(config_path: str = "config.yaml") -> OptimResult:
    """Load params and config paths, then run solve()."""
    config = load_config(config_path)
    params = load_params(config["data"]["params_path"])
    channels = list(config["channels"]["modeled"])
    budget = float(config["optimization"]["default_budget"])
    caps = config["optimization"].get("channel_caps")
    return solve(params, budget, channels, caps=caps)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    out = solve_from_file()
    print(out)
