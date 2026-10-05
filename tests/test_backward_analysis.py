"""Tests for backward_analysis."""
from __future__ import annotations

import pandas as pd
import pytest

from src.backward_analysis import (
    CHART_SAMPLE_SIZE,
    run_backward_analysis,
    stage_1_outcome_identification,
)


def test_backward_analysis_returns_7_stages(cleaned_df):
    result = run_backward_analysis(cleaned_df, user_budget=50000)
    assert len(result.stages) == 7


def test_stage_1_identifies_target_column(cleaned_df):
    stage = stage_1_outcome_identification(cleaned_df)
    assert stage.stage_id == "outcome_identification"
    assert "y" in stage.finding or "ALL_PURCHASES" in stage.finding


def test_stage_2_lists_config_modeled_and_dropped_channels(cleaned_df):
    result = run_backward_analysis(cleaned_df)
    s2 = result.stages[1]
    assert s2.stage_id == "channel_detection"
    assert "Google Paid Search" in s2.finding or "google_paid_search" in s2.finding
    assert "TikTok" in s2.finding or "tiktok" in s2.finding.lower()
    assert len(result.spend_columns) == 5


def test_stage_5_objective_function_text_not_empty(cleaned_df):
    result = run_backward_analysis(cleaned_df)
    assert len(result.objective_function_text) > 0


def test_stage_6_detects_budget_constraint(cleaned_df):
    result = run_backward_analysis(cleaned_df, user_budget=75000)
    assert result.detected_budget == 75000
    assert any("budget" in c.lower() for c in result.constraint_text)


def test_stage_7_confirmed_false_by_default(cleaned_df):
    result = run_backward_analysis(cleaned_df)
    assert result.confirmed_by_user is False


def test_run_backward_analysis_completes_on_sample_data(cleaned_df):
    result = run_backward_analysis(cleaned_df)
    assert result.target_column
    assert result.stages[-1].stage_id == "user_confirmation"


def test_scatter_charts_are_subsampled_not_full_dataset(cleaned_df):
    """Stage 3/4 must not embed every row — that freezes Streamlit."""
    result = run_backward_analysis(cleaned_df)
    max_pts_per_chart = CHART_SAMPLE_SIZE * 4  # up to 4 channel subplots
    for stage in result.stages:
        if stage.chart is None:
            continue
        pts = sum(
            len(tr.x)
            for tr in stage.chart.data
            if hasattr(tr, "x") and tr.x is not None
        )
        assert pts <= max_pts_per_chart, (
            f"{stage.stage_id} has {pts} points; cap is {max_pts_per_chart}"
        )
