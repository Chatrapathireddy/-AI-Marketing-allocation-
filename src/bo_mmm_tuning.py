
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from src.data_prep import load_config, resolve_project_path
from src.mmm_model import run_fitting

logger = logging.getLogger(__name__)


@dataclass
class MmmHyperparams:
    """MMM tuning knobs for one BO evaluation."""

    reg_b_weight: float
    adstock_decay: dict[str, float] = field(default_factory=dict)


@dataclass
class BoMmmResult:
    """Output of a completed BO study."""

    best_hyperparams: MmmHyperparams
    best_metric: float
    best_params: dict[str, dict[str, float]]
    params_path: str
    X_history: np.ndarray
    y_history: np.ndarray
    best_so_far: list[float]
    n_init: int
    n_iter: int
    objective_name: str


def expected_improvement(
    X_candidates: np.ndarray,
    y_obs: np.ndarray,
    gp: GaussianProcessRegressor,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement acquisition (maximize y)."""
    mu, sigma = gp.predict(X_candidates, return_std=True)
    f_best = float(np.max(y_obs))
    sigma = np.maximum(sigma, 1e-9)
    Z = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei = np.maximum(ei, 0.0)
    return ei


def _tuning_cfg(config: dict) -> dict:
    return config.get("mmm_tuning", {})


def _search_space(config: dict, channels: list[str]) -> dict:
    tcfg = _tuning_cfg(config)
    space = dict(tcfg.get("search_space", {}))
    reg = space.get("reg_b_weight", [0.01, 0.2])
    decay = space.get("adstock_decay", [0.05, 0.6])
    return {
        "reg_b_weight": (float(reg[0]), float(reg[1])),
        "adstock_decay": (float(decay[0]), float(decay[1])),
        "tune_decays": bool(tcfg.get("tune_decays", False)),
        "channels": channels,
    }


def encode_hyperparams(hyperparams: MmmHyperparams, space: dict) -> np.ndarray:
    """Map hyperparams to unit-box vector for GP (log scale for decays)."""
    lo, hi = space["reg_b_weight"]
    vec = [(hyperparams.reg_b_weight - lo) / max(hi - lo, 1e-12)]
    if space["tune_decays"]:
        dlo, dhi = space["adstock_decay"]
        base = space["channels"]
        decays = hyperparams.adstock_decay or {}
        for ch in base:
            d = decays.get(ch, (dlo * dhi) ** 0.5)
            log_d = np.log10(max(d, 1e-6))
            log_lo, log_hi = np.log10(dlo), np.log10(dhi)
            vec.append((log_d - log_lo) / max(log_hi - log_lo, 1e-12))
    return np.asarray(vec, dtype=np.float64)


def decode_vector(x_vec: np.ndarray, space: dict, base_decays: dict[str, float]) -> MmmHyperparams:
    """Inverse of encode_hyperparams."""
    lo, hi = space["reg_b_weight"]
    reg_b = lo + float(x_vec[0]) * (hi - lo)
    decays = dict(base_decays)
    if space["tune_decays"]:
        dlo, dhi = space["adstock_decay"]
        log_lo, log_hi = np.log10(dlo), np.log10(dhi)
        for i, ch in enumerate(space["channels"], start=1):
            log_d = log_lo + float(x_vec[i]) * (log_hi - log_lo)
            decays[ch] = float(10 ** log_d)
    return MmmHyperparams(reg_b_weight=reg_b, adstock_decay=decays)


def _sample_candidates(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(n, dim))


def evaluate_mmm_hyperparams(
    hyperparams: MmmHyperparams,
    config: dict | None = None,
    *,
    config_path: str = "config.yaml",
) -> tuple[float, dict]:
    """
    Expensive objective: refit MMM and score on holdout.

    Returns (metric to maximize, fitting result dict).
    """
    config = config or load_config(config_path)
    tcfg = _tuning_cfg(config)
    objective = str(tcfg.get("objective", "test_r2"))
    penalty_per_flag = float(tcfg.get("quality_penalty_per_flag", 0.0))

    base_decays = dict(config["adstock"]["decay_rates"])
    decay_overrides = None
    if tcfg.get("tune_decays", False) and hyperparams.adstock_decay:
        decay_overrides = {
            ch: hyperparams.adstock_decay[ch]
            for ch in config["channels"]["modeled"]
            if ch in hyperparams.adstock_decay
        }

    out_path = tcfg.get(
        "output_path",
        "data/processed/channel_params_bo.json",
    )
    fit = run_fitting(
        config_path=config_path,
        reg_b_weight=hyperparams.reg_b_weight,
        adstock_decay_overrides=decay_overrides,
        params_output_path=out_path,
    )

    if "test_metrics" not in fit or fit["test_metrics"].get("n", 0) == 0:
        raise RuntimeError(
            "Holdout test split required for BO — run data pipeline first "
            "(mmm_test.csv missing or empty after aggregation)."
        )

    metrics = fit["test_metrics"]
    if objective == "test_rmse":
        score = -float(metrics["rmse"])
    elif objective == "test_mae":
        score = -float(metrics["mae"])
    else:
        score = float(metrics["r2"])

    n_flags = len(fit.get("quality_report", {}).get("flags", []))
    score -= penalty_per_flag * n_flags
    return score, fit


def bayesian_optimize_mmm(
    objective_fn: Callable[[MmmHyperparams], float] | None = None,
    *,
    config_path: str = "config.yaml",
    n_init: int | None = None,
    n_iter: int | None = None,
    xi: float | None = None,
    n_candidates: int | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> BoMmmResult:
    """
    GP + EI loop over MMM hyperparameters (RetailSense pattern).

    If ``objective_fn`` is None, uses ``evaluate_mmm_hyperparams`` on real data.
    """
    config = load_config(config_path)
    tcfg = _tuning_cfg(config)
    channels = list(config["channels"]["modeled"])
    space = _search_space(config, channels)
    base_decays = dict(config["adstock"]["decay_rates"])

    n_init = n_init if n_init is not None else int(tcfg.get("n_init", 5))
    n_iter = n_iter if n_iter is not None else int(tcfg.get("n_iter", 25))
    xi = xi if xi is not None else float(tcfg.get("xi", 0.01))
    n_candidates = n_candidates if n_candidates is not None else int(
        tcfg.get("n_candidates", 5000)
    )

    dim = 1 + (len(channels) if space["tune_decays"] else 0)
    rng = np.random.default_rng(seed)
    use_real_mmm = objective_fn is None

    if use_real_mmm:
        def _objective(h: MmmHyperparams) -> float:
            score, _ = evaluate_mmm_hyperparams(h, config=config, config_path=config_path)
            return score
    else:
        _objective = objective_fn  # type: ignore[assignment]

    if verbose:
        logger.info(
            "BO MMM tuning: dim=%d, n_init=%d, n_iter=%d, tune_decays=%s",
            dim,
            n_init,
            n_iter,
            space["tune_decays"],
        )

    X_history = _sample_candidates(n_init, dim, rng)
    y_history = np.array([_objective(decode_vector(x, space, base_decays)) for x in X_history])
    best_so_far = [float(y_history.max())]

    length_scale = np.ones(dim)
    kernel = C(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=1e-4)

    for iteration in range(n_iter):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=seed + iteration,
            normalize_y=True,
        )
        gp.fit(X_history, y_history)

        candidates = _sample_candidates(n_candidates, dim, rng)
        ei = expected_improvement(candidates, y_history, gp, xi=xi)
        x_next = candidates[int(np.argmax(ei))]
        y_next = float(_objective(decode_vector(x_next, space, base_decays)))

        X_history = np.vstack([X_history, x_next])
        y_history = np.append(y_history, y_next)
        best_so_far.append(float(y_history.max()))
        if verbose:
            logger.info(
                "BO iter %d/%d — score=%.4f, best=%.4f",
                iteration + 1,
                n_iter,
                y_next,
                best_so_far[-1],
            )

    best_idx = int(np.argmax(y_history))
    best_h = decode_vector(X_history[best_idx], space, base_decays)
    best_metric = float(y_history[best_idx])

    if use_real_mmm:
        _, fit = evaluate_mmm_hyperparams(best_h, config=config, config_path=config_path)
        best_params = fit["params"]
        params_path = str(fit["params_path"])
    else:
        best_params = {}
        params_path = ""

    return BoMmmResult(
        best_hyperparams=best_h,
        best_metric=best_metric,
        best_params=best_params,
        params_path=params_path,
        X_history=X_history,
        y_history=y_history,
        best_so_far=best_so_far,
        n_init=n_init,
        n_iter=n_iter,
        objective_name=str(tcfg.get("objective", "test_r2")),
    )


def run_bo_tuning(config_path: str = "config.yaml", verbose: bool = True) -> BoMmmResult:
    """Run BO study and write best params to ``mmm_tuning.output_path``."""
    config = load_config(config_path)
    if not _tuning_cfg(config).get("enabled", False):
        raise RuntimeError(
            "mmm_tuning.enabled is false in config.yaml — set true to run BO offline."
        )
    result = bayesian_optimize_mmm(config_path=config_path, verbose=verbose)
    meta_path = resolve_project_path(
        _tuning_cfg(config).get(
            "metadata_path",
            "data/processed/channel_params_bo_meta.json",
        )
    )
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "best_metric": result.best_metric,
        "objective": result.objective_name,
        "reg_b_weight": result.best_hyperparams.reg_b_weight,
        "adstock_decay": result.best_hyperparams.adstock_decay,
        "n_init": result.n_init,
        "n_iter": result.n_iter,
        "params_path": result.params_path,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("BO complete — best metric=%.4f, params=%s", result.best_metric, result.params_path)
    return result


def load_bo_params(config: dict | None = None) -> dict | None:
    """Load BO-tuned channel params if enabled and file exists."""
    config = config or load_config()
    tcfg = _tuning_cfg(config)
    if not tcfg.get("use_bo_params", False):
        return None
    path = resolve_project_path(
        tcfg.get("output_path", "data/processed/channel_params_bo.json")
    )
    if not Path(path).is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    run_bo_tuning()
