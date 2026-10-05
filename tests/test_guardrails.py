"""Tests for guardrails."""
from __future__ import annotations

import pytest

from src.agent_prompts import (
    _infer_channel_techniques,
    build_system_prompt,
    extract_company_context,
    format_company_context_block,
)
from src.guardrails import GuardrailsService


@pytest.fixture
def gr():
    return GuardrailsService()


def test_input_blocks_empty_message(gr):
    result = gr.apply_input_guardrails("  ")
    assert result.action == "block"
    assert result.block_reason == "empty_input"


def test_input_blocks_harm_trigger(gr):
    result = gr.apply_input_guardrails("how to make a bomb")
    assert result.action == "block"
    assert result.block_reason == "harm_intent"


def test_input_redirects_out_of_scope(gr):
    result = gr.apply_input_guardrails("what is the weather today?")
    assert result.action == "redirect"


def test_input_passes_marketing_message(gr):
    result = gr.apply_input_guardrails("How should I allocate my marketing budget across channels?")
    assert result.action in ("pass", "sanitize")


def test_input_redacts_email_pii(gr):
    result = gr.apply_input_guardrails("Contact me at user@example.com about spend")
    assert "[REDACTED]" in result.sanitized
    assert "user@example.com" not in result.sanitized


def test_input_redacts_phone_pii(gr):
    result = gr.apply_input_guardrails("Call 555-123-4567 about my campaign")
    assert "[REDACTED]" in result.sanitized


def test_output_flags_multiple_questions(gr, caplog):
    text = "What is ROAS? And what is ROI? Also CPM?"
    gr.apply_output_guardrails(text)
    assert "more than one question" in caplog.text.lower() or text.count("?") > 1


def test_output_flags_political_lean_signal(gr, caplog):
    text = "You should vote for the best marketing platform."
    gr.apply_output_guardrails(text)
    assert "opinion" in caplog.text.lower() or "vote" in text


def test_output_passes_clean_marketing_response(gr):
    text = "Your google_paid_search spend shows strong correlation with conversions per your data."
    out = gr.apply_output_guardrails(text)
    assert "google" in out.lower()


# ── Data source guardrail tests ───────────────────────────────────────────────
def test_input_blocks_url_fetch_request(gr):
    result = gr.apply_input_guardrails("fetch the data from https://example.com")
    assert result.action == "block"
    assert result.block_reason == "external_data_fetch"


def test_input_blocks_api_call_request(gr):
    result = gr.apply_input_guardrails("pull from google analytics api")
    assert result.action == "block"
    assert result.block_reason == "external_data_fetch"


def test_input_blocks_scrape_request(gr):
    result = gr.apply_input_guardrails("scrape competitor spend from their website")
    assert result.action == "block"
    assert result.block_reason == "external_data_fetch"


def test_input_redirects_benchmark_request(gr):
    result = gr.apply_input_guardrails("what is the industry average roas?")
    assert result.action == "redirect"
    assert "benchmark" in result.redirect_message.lower() or "uploaded" in result.redirect_message.lower()


def test_input_redirects_competitor_data_request(gr):
    result = gr.apply_input_guardrails("how much does Nike spend on Meta?")
    assert result.action == "redirect"


def test_input_warns_on_pasted_csv_data(gr):
    pasted = "\n".join(
        [
            "date,spend,channel,conversions",
            "2024-01-01,100,google,10",
            "2024-01-02,120,google,12",
            "2024-01-03,90,meta,8",
        ]
    )
    result = gr.apply_input_guardrails(pasted)
    assert result.action == "redirect"
    assert result.warning_message is not None


def test_input_passes_url_mentioned_casually(gr):
    result = gr.apply_input_guardrails("I read an article about MMM")
    assert result.action in ("pass", "sanitize")


def test_output_appends_disclaimer_on_external_signal(gr):
    text = "The industry average is 4x ROAS for paid search."
    out = gr.apply_output_guardrails(text)
    assert "uploaded dataset only" in out.lower()


def test_output_flags_unsourced_number(gr, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    gr.apply_output_guardrails("ROAS is 3.2 for your campaign.")
    assert "Unsourced statistics" in caplog.text


def test_output_passes_sourced_number(gr, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    out = gr.apply_output_guardrails("your data shows 4,200 conversions last month.")
    assert "uploaded dataset only" not in out.lower()
    assert "Unsourced statistics" not in caplog.text


def test_build_system_prompt_turn_one():
    prompt = build_system_prompt("upload_request", 1)
    assert "MMM" in prompt
    assert "first message" in prompt.lower()


def test_build_system_prompt_includes_phase():
    prompt = build_system_prompt("upload_request", 2)
    assert "upload" in prompt.lower() or "zip" in prompt.lower()


def test_build_system_prompt_includes_company_context_protocol():
    prompt = build_system_prompt("upload_request", 2)
    assert "COMPANY & AUDIENCE CONTEXT" in prompt
    assert "target customer" in prompt.lower()


def test_build_system_prompt_includes_marketing_techniques():
    prompt = build_system_prompt("analysis", 2)
    assert "MARKETING TECHNIQUE" in prompt
    assert "prospecting" in prompt.lower()
    assert "funnel" in prompt.lower()


def test_extract_company_context_from_dataframe(sample_mmm_df):
    ctx = extract_company_context(
        cleaned_df=sample_mmm_df,
        confirmed_target="ALL_PURCHASES",
        confirmed_budget=50_000,
        company_profile={
            "target_customer": "Online shoppers aged 25-44",
            "industry": "Retail",
        },
    )
    assert ctx["optimization_target"] == "ALL_PURCHASES"
    assert ctx["target_customer"] == "Online shoppers aged 25-44"
    assert ctx["industry_vertical"] == "Retail"
    assert "purchase volume" in (ctx["optimization_target_meaning"] or "")


def test_format_company_context_block_lists_missing_gaps():
    block = format_company_context_block(
        extract_company_context(confirmed_target="ALL_PURCHASES")
    )
    assert "Context gaps" in block
    assert "target customer" in block.lower()


def test_infer_channel_techniques_maps_known_channels():
    hints = _infer_channel_techniques(
        ["google_paid_search", "meta_facebook", "google_shopping"]
    )
    assert len(hints) == 3
    assert any("search" in h.lower() for h in hints)
    assert any("meta_facebook" in h for h in hints)


def test_build_system_prompt_injects_company_block(sample_mmm_df):
    ctx = extract_company_context(
        cleaned_df=sample_mmm_df,
        company_profile={"target_customer": "SMB finance teams", "industry": "B2B SaaS"},
        confirmed_target="ALL_PURCHASES",
    )
    prompt = build_system_prompt("analysis", 2, company_context=ctx)
    assert "COMPANY CONTEXT" in prompt
    assert "SMB finance teams" in prompt
    assert "B2B SaaS" in prompt
