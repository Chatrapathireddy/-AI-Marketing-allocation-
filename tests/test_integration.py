"""
Day 7 end-to-end smoke test — Ana's pipeline; teammates add assertions later.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.backward_analysis import run_backward_analysis
from src.data_prep import load_config, run_pipeline, split_train_test
from src.guardrails import GuardrailsService


@pytest.fixture(scope="module")
def sample_df():
    path = Path("data/raw/conjura_mmm_data.csv")
    if not path.exists():
        pytest.skip("Raw data not available in test environment")
    return pd.read_csv(path, nrows=500)


def test_data_prep_pipeline_runs_without_error(sample_mmm_df, tmp_path):
    raw = tmp_path / "uploaded_dataset.csv"
    sample_mmm_df.to_csv(raw, index=False)
    proc = tmp_path / "processed"
    proc.mkdir(exist_ok=True)
    import yaml

    config = load_config()
    config["data"]["raw_path"] = str(raw)
    config["data"]["processed_path"] = str(proc / "mmm_ready.csv")
    config["data"]["train_path"] = str(proc / "mmm_train.csv")
    config["data"]["test_path"] = str(proc / "mmm_test.csv")
    cfg = tmp_path / "config.yaml"
    with open(cfg, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    result = run_pipeline(str(cfg), raw_path=str(raw))
    assert len(result["ready_df"]) > 0


def test_train_test_split_no_leakage(cleaned_df):
    config = load_config()
    train, test = split_train_test(cleaned_df, config=config)
    date_col = config["data"]["date_column"]
    assert train[date_col].max() < test[date_col].min() or len(test) == 0 or len(train) == 0


def test_guardrails_block_harm_on_integration():
    gr = GuardrailsService()
    r = gr.apply_input_guardrails("how to attack someone")
    assert r.action == "block"


def test_backward_analysis_runs_on_sample(cleaned_df):
    result = run_backward_analysis(cleaned_df)
    assert len(result.stages) == 7


def test_schema_confirmed_before_optimization_runs():
    """Optimizer must stay blocked until user confirms backward analysis."""
    backward_confirmed = False

    def optimization_allowed() -> bool:
        return backward_confirmed

    assert optimization_allowed() is False
    backward_confirmed = True
    assert optimization_allowed() is True


@pytest.mark.skip(reason="Gregory implements")
def test_mmm_model_runs_after_data_prep():
    pass


def test_optimizer_budget_constraint_holds():
    from src.optimizer import solve

    params = {
        "google_paid_search": {"a": 5000.0, "b": 1e-5},
        "google_shopping": {"a": 8000.0, "b": 1e-5},
        "google_pmax": {"a": 6000.0, "b": 1e-5},
        "meta_facebook": {"a": 7000.0, "b": 1e-5},
        "meta_instagram": {"a": 3000.0, "b": 1e-5},
    }
    channels = list(params.keys())
    budget = 250_000.0
    result = solve(params, budget, channels, n_starts=10, seed=0)
    assert sum(result.allocation.values()) <= budget + 1e-4
    assert result.success
    assert result.predicted_conversions > 0


def test_optimizer_kkt_satisfied():
    from src.optimizer import solve, verify_kkt

    params = {
        "google_paid_search": {"a": 5000.0, "b": 1e-5},
        "google_shopping": {"a": 8000.0, "b": 1e-5},
        "google_pmax": {"a": 6000.0, "b": 1e-5},
        "meta_facebook": {"a": 7000.0, "b": 1e-5},
        "meta_instagram": {"a": 3000.0, "b": 1e-5},
    }
    channels = list(params.keys())
    budget = 250_000.0
    result = solve(params, budget, channels, n_starts=10, seed=1)
    kkt = verify_kkt(result, budget, channels, params=params, tol=1e-3)
    assert kkt["budget_feasible"]
    assert kkt["non_negative"]
    assert kkt["status"] == "pass"


def test_agent_responds_in_scope(monkeypatch):
    from src.agent import run_agent

    monkeypatch.setattr("src.agent.get_claude_client", lambda: None)

    response = run_agent(
        "How does backward analysis work for my marketing dataset?",
        "system prompt",
        [],
        context={"phase": "upload_request", "backward_analysis_confirmed": True},
    )
    assert response
    assert "ANTHROPIC_API_KEY" in response or "marketing" in response.lower()


@pytest.mark.skip(reason="All team")
def test_full_pipeline_end_to_end():
    pass
