"""Tests for Bayesian Optimization MMM tuning."""
from __future__ import annotations

import numpy as np
import pytest

from src.bo_mmm_tuning import (
    MmmHyperparams,
    bayesian_optimize_mmm,
    decode_vector,
    encode_hyperparams,
    expected_improvement,
)


def test_expected_improvement_non_negative():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    rng = np.random.default_rng(0)
    X_obs = rng.uniform(0, 1, (8, 2))
    y_obs = np.sin(X_obs[:, 0] * 3).astype(float)
    gp = GaussianProcessRegressor(kernel=RBF(), random_state=0)
    gp.fit(X_obs, y_obs)
    X_c = rng.uniform(0, 1, (100, 2))
    ei = expected_improvement(X_c, y_obs, gp)
    assert ei.shape == (100,)
    assert np.all(ei >= 0)


def test_encode_decode_roundtrip():
    space = {
        "reg_b_weight": (0.01, 0.2),
        "adstock_decay": (0.05, 0.6),
        "tune_decays": False,
        "channels": ["a", "b"],
    }
    h = MmmHyperparams(reg_b_weight=0.07, adstock_decay={})
    x = encode_hyperparams(h, space)
    h2 = decode_vector(x, space, {"a": 0.3, "b": 0.3})
    assert h2.reg_b_weight == pytest.approx(0.07)


def test_bayesian_optimize_mock_objective():
    """BO on a cheap 1D peak — no MMM refit."""
    calls = {"n": 0}
    optimum = 0.35

    def objective(h: MmmHyperparams) -> float:
        calls["n"] += 1
        return -((h.reg_b_weight - optimum) ** 2)

    result = bayesian_optimize_mmm(
        objective_fn=objective,
        config_path="config.yaml",
        n_init=4,
        n_iter=6,
        n_candidates=500,
        seed=1,
        verbose=False,
    )
    assert calls["n"] == 10
    assert result.best_metric > -0.05
    assert len(result.best_so_far) == 7


def test_load_bo_params_disabled():
    from src.bo_mmm_tuning import load_bo_params
    from src.data_prep import load_config

    cfg = load_config()
    assert load_bo_params(cfg) is None
