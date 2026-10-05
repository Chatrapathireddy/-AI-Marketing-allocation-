
from __future__ import annotations

import pytest

from src.agent import (
    ALL_CHANNELS,
    ParamChange,
    apply_parameter_change,
    detect_parameter_change,
    is_cancellation,
    is_confirmation,
    parse_parameter_change,
    process_parameter_message,
    validate_parameter_change,
)
from src.agent_prompts import summarize_results_context
from src.optimization_pipeline import maybe_resolve


class FakeSession(dict):
    """dict that also supports attribute access (mimics st.session_state)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


# --------------------------------------------------------------------------- #
# parse_parameter_change (regex)
# --------------------------------------------------------------------------- #
def test_parse_kappa_with_k_suffix():
    change = parse_parameter_change("set google_pmax activation to $20k and re-optimize")
    assert change is not None
    assert change.kind == "kappa"
    assert change.channel == "google_pmax"
    assert change.value == 20_000.0


def test_parse_kappa_spoken_channel_and_commas():
    change = parse_parameter_change("raise Meta Facebook activation threshold to $15,000")
    assert change.kind == "kappa"
    assert change.channel == "meta_facebook"
    assert change.value == 15_000.0


def test_parse_lambda():
    change = parse_parameter_change("set instagram decay to 0.5")
    assert change.kind == "lambda"
    assert change.channel == "meta_instagram"
    assert change.value == 0.5


def test_parse_budget_no_channel():
    change = parse_parameter_change("change the total budget to $400k")
    assert change.kind == "budget"
    assert change.channel is None
    assert change.value == 400_000.0


def test_parse_returns_none_for_plain_question():
    assert parse_parameter_change("what does the shadow price mean?") is None


def test_detect_without_llm_fallback_is_regex_only():
    assert detect_parameter_change("hello there", use_llm_fallback=False) is None


def test_question_with_ceiling_and_number_is_not_a_command():
    # Regression: this question previously misfired as a ceiling change.
    q = "if the ceiling is 375K then how did the optimizer allocate 1.4 million to google paid search?"
    assert detect_parameter_change(q, use_llm_fallback=False) is None


def test_question_starter_blocks_detection():
    assert detect_parameter_change(
        "what happens to meta_facebook activation at $20k?", use_llm_fallback=False
    ) is None


def test_explicit_command_still_detected_even_if_invalid():
    change = detect_parameter_change(
        "update threshold for paid search to $1,000,000", use_llm_fallback=False
    )
    assert change is not None
    assert change.kind == "kappa"
    assert change.channel == "google_paid_search"


# --------------------------------------------------------------------------- #
# polite imperatives ("can you change...") and bulk "all channels"
# --------------------------------------------------------------------------- #
def test_polite_request_is_detected_as_command():
    # Regression for the PDF: "can you change..." was wrongly treated as a question.
    change = detect_parameter_change(
        "can you change the meta_facebook threshold to $20k?", use_llm_fallback=False
    )
    assert change is not None
    assert change.kind == "kappa"
    assert change.channel == "meta_facebook"
    assert change.value == 20_000.0


def test_polite_explanation_request_is_not_a_command():
    assert detect_parameter_change(
        "can you explain how the threshold would change the allocation?",
        use_llm_fallback=False,
    ) is None


def test_bulk_all_channels_threshold_parsed():
    change = parse_parameter_change("can you change the channel thresholds to 10 million?")
    assert change is not None
    assert change.kind == "kappa"
    assert change.channel == ALL_CHANNELS
    assert change.value == 10_000_000.0


def test_bulk_kappa_above_all_ceilings_rejected_with_explanation():
    change = ParamChange(kind="kappa", channel=ALL_CHANNELS, value=10_000_000.0)
    ok, msg = validate_parameter_change(change)
    assert not ok
    assert "OFF" in msg or "ceiling" in msg.lower()


def test_bulk_kappa_apply_sets_every_channel(five_channel_params):
    session = FakeSession(activation_thresholds={})
    change = ParamChange(kind="kappa", channel=ALL_CHANNELS, value=5_000.0)
    apply_parameter_change(session, change)
    thresholds = session["activation_thresholds"]
    assert set(thresholds) == set(five_channel_params)
    assert all(v == 5_000.0 for v in thresholds.values())
    assert session["params_dirty"] is True


# --------------------------------------------------------------------------- #
# validate_parameter_change
# --------------------------------------------------------------------------- #
def test_validate_kappa_above_ceiling_rejected():
    # google_paid_search ceiling is 375,516 in config
    change = ParamChange(kind="kappa", channel="google_paid_search", value=500_000.0)
    ok, msg = validate_parameter_change(change)
    assert not ok
    assert "ceiling" in msg.lower()


def test_validate_kappa_ok():
    change = ParamChange(kind="kappa", channel="meta_facebook", value=20_000.0)
    ok, _ = validate_parameter_change(change)
    assert ok


def test_validate_missing_channel_asks_which():
    change = ParamChange(kind="kappa", channel=None, value=10_000.0)
    ok, msg = validate_parameter_change(change)
    assert not ok
    assert "which channel" in msg.lower()


def test_validate_lambda_out_of_range():
    change = ParamChange(kind="lambda", channel="meta_instagram", value=1.5)
    ok, msg = validate_parameter_change(change)
    assert not ok
    assert "λ" in msg or "between 0 and 1" in msg


def test_validate_budget_must_be_positive():
    ok, _ = validate_parameter_change(ParamChange(kind="budget", channel=None, value=-5.0))
    assert not ok


# --------------------------------------------------------------------------- #
# apply + confirmation helpers
# --------------------------------------------------------------------------- #
def test_apply_kappa_sets_session_and_dirty():
    session = FakeSession(activation_thresholds={"meta_facebook": 12_000.0})
    apply_parameter_change(session, ParamChange(kind="kappa", channel="meta_facebook", value=15_000.0))
    assert session["activation_thresholds"]["meta_facebook"] == 15_000.0
    assert session["params_dirty"] is True


def test_apply_budget_sets_confirmed_budget():
    session = FakeSession()
    apply_parameter_change(session, ParamChange(kind="budget", channel=None, value=400_000.0))
    assert session["confirmed_budget"] == 400_000.0
    assert session["params_dirty"] is True


def test_confirmation_and_cancellation_words():
    assert is_confirmation("yes")
    assert is_confirmation("proceed please")
    assert not is_confirmation("not yet")
    assert is_cancellation("cancel")
    assert is_cancellation("no thanks")


# --------------------------------------------------------------------------- #
# process_parameter_message (coordinator)
# --------------------------------------------------------------------------- #
def test_process_detects_then_confirms_then_applies():
    session = FakeSession(
        activation_thresholds={"google_pmax": 18_000.0},
        activation_ceilings={"google_pmax": 1_232_752.0},
    )
    first = process_parameter_message(
        session, "set google pmax activation to $25k", use_llm_fallback=False
    )
    assert first["handled"]
    assert first["needs_resolve"] is False
    assert "proceed" in first["response"].lower()
    assert session.get("pending_param_change") is not None

    second = process_parameter_message(session, "yes", use_llm_fallback=False)
    assert second["handled"]
    assert second["needs_resolve"] is True
    assert session["activation_thresholds"]["google_pmax"] == 25_000.0
    assert session["params_dirty"] is True
    assert session.get("pending_param_change") is None


def test_process_invalid_channel_value_returns_clarification():
    session = FakeSession()
    out = process_parameter_message(
        session, "set google_paid_search activation to $500k", use_llm_fallback=False
    )
    assert out["handled"]
    assert out["needs_resolve"] is False
    assert "ceiling" in out["response"].lower()


def test_process_cancel_clears_pending():
    session = FakeSession(pending_param_change=ParamChange(kind="budget", channel=None, value=100.0))
    out = process_parameter_message(session, "cancel", use_llm_fallback=False)
    assert out["handled"]
    assert out["needs_resolve"] is False
    assert session.get("pending_param_change") is None


def test_process_passthrough_for_non_parameter_message():
    session = FakeSession()
    out = process_parameter_message(
        session, "explain the shadow price", use_llm_fallback=False
    )
    assert out["handled"] is False


# --------------------------------------------------------------------------- #
# maybe_resolve
# --------------------------------------------------------------------------- #
@pytest.fixture
def five_channel_params() -> dict:
    return {
        "google_paid_search": {"a": 100.0, "b": 0.001},
        "google_shopping": {"a": 80.0, "b": 0.002},
        "google_pmax": {"a": 90.0, "b": 0.0015},
        "meta_facebook": {"a": 70.0, "b": 0.003},
        "meta_instagram": {"a": 60.0, "b": 0.004},
    }


def test_maybe_resolve_noop_when_not_dirty(five_channel_params):
    session = FakeSession(params_dirty=False, channel_params=five_channel_params)
    assert maybe_resolve(session) is None


def test_maybe_resolve_reoptimizes_models_a_and_b(five_channel_params):
    thresholds = {
        "google_paid_search": 18_000.0,
        "google_shopping": 15_000.0,
        "google_pmax": 18_000.0,
        "meta_facebook": 12_000.0,
        "meta_instagram": 12_000.0,
    }
    ceilings = {
        "google_paid_search": 375_516.0,
        "google_shopping": 912_392.0,
        "google_pmax": 1_232_752.0,
        "meta_facebook": 1_465_278.0,
        "meta_instagram": 488_104.0,
    }
    session = FakeSession(
        params_dirty=True,
        channel_params=five_channel_params,
        confirmed_budget=100_000.0,
        activation_thresholds=thresholds,
        activation_ceilings=ceilings,
    )
    optim = maybe_resolve(session)
    assert optim is not None
    assert session["params_dirty"] is False
    assert session["optim_result"] is optim
    assert session["optim_result_B"] is not None
    assert session["optim_result_B"].total_spent <= 100_000.0 + 1e-3


# --------------------------------------------------------------------------- #
# summarize_results_context — lets the chatbot explain the charts
# --------------------------------------------------------------------------- #
def test_summarize_results_empty_before_optimization():
    assert summarize_results_context(FakeSession()) == ""


def test_summarize_results_includes_allocation_and_models():
    optim = {
        "allocation": {"google_paid_search": 58_442.0, "google_shopping": 772_558.0},
        "baseline_allocation": {"google_paid_search": 70_000.0, "google_shopping": 200_000.0},
        "predicted_conversions": 94_085.0,
        "baseline_conversions": 69_643.0,
        "lift_pct": 35.1,
        "lambda_budget": 0.0774,
        "total_spent": 831_000.0,
    }
    session = FakeSession(
        optim_result=optim,
        channel_params={"google_shopping": {"a": 296_000.0, "b": 3.4e-7}},
        activation_thresholds={"google_shopping": 15_000.0},
        activation_ceilings={"google_shopping": 912_392.0},
        optim_result_C={"allocation": {"google_shopping": 760_000.0}, "predicted_conversions": 95_076.0},
        adstock_lambdas={"google_paid_search": 0.3, "google_shopping": 0.0},
    )
    digest = summarize_results_context(session)
    assert "Allocation page" in digest
    assert "google_shopping" in digest
    assert "94,085" in digest          # predicted conversions
    assert "Saturation Curves page" in digest
    assert "Model C" in digest
    assert "95,076" in digest          # Model C conversions
