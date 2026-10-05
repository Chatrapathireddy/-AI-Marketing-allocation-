
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.channel_policy import (
    analysis_spend_columns,
    classify_channels,
    display_name,
    format_dropped_channel_sentence,
    format_modeled_channel_sentence,
)
from src.data_prep import load_config

logger = logging.getLogger(__name__)

# Cap scatter points sent to Plotly — full-row charts freeze Streamlit in the browser.
CHART_SAMPLE_SIZE = 2000

# Days per modeling period. The detected budget must use the SAME period as the
# fitted MMM curves and Ana's κ/u_c (config mmm.freq), or budget and thresholds
# end up on different scales (e.g. annual $3.5M budget vs weekly $375K ceilings).
_PERIOD_DAYS = {"daily": 1.0, "weekly": 7.0, "monthly": 365.0 / 12.0}


def _budget_period_days(config: dict) -> float:
    """Days in one modeling period, driven by config ``mmm.freq``."""
    freq = str(config.get("mmm", {}).get("freq", "monthly")).lower()
    return _PERIOD_DAYS.get(freq, 365.0 / 12.0)

TARGET_PRIORITY = [
    "y",
    "ALL_PURCHASES",
    "CONVERSIONS",
    "REVENUE",
    "SALES",
    "PURCHASES",
    "ALL_PURCHASES_ORIGINAL_PRICE",
]


@dataclass
class AnalysisStage:
    stage_id: str
    title: str
    finding: str
    technical_detail: str
    chart: Any | None
    confirmed: bool = False


@dataclass
class BackwardAnalysisResult:
    stages: list[AnalysisStage] = field(default_factory=list)
    target_column: str = ""
    spend_columns: list[str] = field(default_factory=list)
    objective_function_text: str = ""
    objective_function_math: str = ""
    constraint_text: list[str] = field(default_factory=list)
    detected_budget: float | None = None
    confirmed_by_user: bool = False


def _resolve_target(df: pd.DataFrame) -> str:
    for col in TARGET_PRIORITY:
        if col in df.columns and df[col].notna().any():
            return col
    numeric = df.select_dtypes(include=[np.number]).columns
    for col in numeric:
        if "SPEND" not in col.upper() and df[col].notna().mean() > 0.5:
            return col
    raise ValueError("No target column found in dataset.")



def _sample_for_chart(df: pd.DataFrame, n: int = CHART_SAMPLE_SIZE) -> pd.DataFrame:
    """Random subsample for visualization; statistics still use the full frame."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)


def strip_stage_charts(result: BackwardAnalysisResult) -> None:
    """Drop Plotly figures from session state after confirm to speed reruns."""
    for stage in result.stages:
        stage.chart = None


def stage_1_outcome_identification(df: pd.DataFrame) -> AnalysisStage:
    target = _resolve_target(df)
    s = df[target].dropna()
    null_pct = df[target].isna().mean()
    date_col = next((c for c in df.columns if "DATE" in c.upper()), None)

    if date_col:
        daily = df.groupby(date_col)[target].sum().reset_index()
        chart = px.line(daily, x=date_col, y=target, title=f"{target} over time")
    else:
        chart = px.histogram(s, title=f"{target} distribution")

    finding = (
        f"We found your outcome variable: **{target}** — the total metric per day. "
        f"It ranges from {s.min():.0f} to {s.max():.0f} with a mean of {s.mean():.0f} "
        f"and {null_pct:.1%} missing values. Here's how it looked over your data period:"
    )
    return AnalysisStage(
        stage_id="outcome_identification",
        title="Stage 1: Outcome identification",
        finding=finding,
        technical_detail=f"Selected target `{target}` by lowest missing rate among candidates.",
        chart=chart,
    )


def stage_2_channel_detection(df: pd.DataFrame, config: dict | None = None) -> AnalysisStage:
    config = config or load_config()
    modeled_keys, dropped_keys, modeled_raw = classify_channels(df, config)
    policy_dropped = list(config["channels"]["dropped_sparse"])
    column_map = config.get("column_map", {})

    coverage: dict[str, float] = {}
    for ch in modeled_keys:
        raw = column_map.get(ch)
        adstock = f"{ch}_adstock"
        col = adstock if adstock in df.columns else raw
        if col and col in df.columns:
            coverage[display_name(ch)] = 1.0 - float(df[col].isna().mean())

    if coverage:
        chart = px.bar(
            x=list(coverage.values()),
            y=list(coverage.keys()),
            orientation="h",
            labels={"x": "non_null_pct", "y": "channel"},
            title="Modeled channel coverage (% non-null)",
        )
    else:
        chart = go.Figure()

    modeled_sentence = format_modeled_channel_sentence(modeled_keys)
    dropped_sentence = format_dropped_channel_sentence(policy_dropped)
    finding = (
        f"We will model **{len(modeled_keys)}** marketing channels from your data: "
        f"{modeled_sentence}. "
        f"These match our EDA policy (sparse channels are excluded, not filled). "
        f"**Excluded from the model:** {dropped_sentence}. "
        f"Here's coverage for the modeled channels:"
    )
    return AnalysisStage(
        stage_id="channel_detection",
        title="Stage 2: Channel detection",
        finding=finding,
        technical_detail=(
            "Modeled per config: 5 channels with null→0; "
            "Display, Video, Meta Other, TikTok dropped as too sparse."
        ),
        chart=chart,
    )


def stage_3_spend_response_relationship(
    df: pd.DataFrame,
    channels: list[str],
    target: str,
) -> AnalysisStage:
    correlations = {}
    for ch in channels:
        if ch in df.columns and target in df.columns:
            sub = df[[ch, target]].dropna()
            if len(sub) > 2:
                correlations[ch] = sub[ch].corr(sub[target])

    strongest = max(correlations, key=correlations.get) if correlations else (channels[0] if channels else "n/a")
    r = correlations.get(strongest, 0.0)

    n = min(len(channels), 4)
    cols = channels[:n] if channels else []
    if cols:
        fig = make_subplots(rows=(len(cols) + 1) // 2, cols=2, subplot_titles=cols)
        for i, ch in enumerate(cols):
            row, col = i // 2 + 1, i % 2 + 1
            sub = _sample_for_chart(df[[ch, target]].dropna())
            fig.add_trace(
                go.Scatter(x=sub[ch], y=sub[target], mode="markers", name=ch),
                row=row,
                col=col,
            )
        fig.update_layout(title="Spend vs outcome by channel", showlegend=False)
        chart = fig
    else:
        chart = None

    finding = (
        f"We measured how each channel's spend relates to your outcome. "
        f"**{strongest}** shows the strongest relationship (r = {r:.2f}). "
        f"This confirms spend drives conversions — the prerequisite for optimization."
    )
    return AnalysisStage(
        stage_id="spend_response_relationship",
        title="Stage 3: Spend–response relationship",
        finding=finding,
        technical_detail="Pearson correlation between daily spend and target per channel.",
        chart=chart,
    )


def stage_4_saturation_check(
    df: pd.DataFrame,
    channels: list[str],
    target: str,
) -> AnalysisStage:
    n = min(len(channels), 4)
    cols = channels[:n]
    fig = make_subplots(rows=(len(cols) + 1) // 2, cols=2, subplot_titles=cols) if cols else None
    concave_count = 0

    for i, ch in enumerate(cols):
        sub_full = df[[ch, target]].dropna()
        sub_full = sub_full[sub_full[ch] > 0]
        if len(sub_full) < 10:
            continue
        x = np.log1p(sub_full[ch].values)
        y = sub_full[target].values
        coef = np.polyfit(x, y, 2)
        concave = coef[0] < 0
        if concave:
            concave_count += 1
        x_line = np.linspace(x.min(), x.max(), 50)
        y_line = np.polyval(coef, x_line)
        row, col = i // 2 + 1, i % 2 + 1
        sub = _sample_for_chart(sub_full)
        fig.add_trace(go.Scatter(x=sub[ch], y=sub[target], mode="markers", name=ch), row=row, col=col)
        fig.add_trace(
            go.Scatter(x=np.expm1(x_line), y=y_line, mode="lines", name=f"{ch} fit"),
            row=row,
            col=col,
        )

    if fig:
        fig.update_layout(title="Saturation check (log-quadratic fit)", showlegend=False)
        chart = fig
    else:
        chart = None

    finding = (
        f"We tested diminishing returns per channel. "
        f"{concave_count} of {len(cols)} checked channels show concave patterns. "
        f"This supports saturation curves f(x) = a·(1−e^(−b·x)) rather than a linear model."
    )
    return AnalysisStage(
        stage_id="saturation_check",
        title="Stage 4: Saturation check",
        finding=finding,
        technical_detail="Log-transform spend; quadratic fit — negative leading coeff indicates concavity.",
        chart=chart,
    )


def stage_5_objective_formulation(channels: list[str]) -> AnalysisStage:
    ch_list = ", ".join(channels) if channels else "each modeled channel"
    finding = (
        "Based on what we found, your optimization problem is:\n\n"
        "**MAXIMIZE** total predicted conversions across all channels, "
        "where each channel's contribution follows a saturation curve.\n\n"
        f"Channels included: {ch_list}."
    )
    math = (
        "max  Σᵢ aᵢ·(1−exp(−bᵢ·xᵢ))\n"
        "where xᵢ is budget allocated to channel i, and aᵢ, bᵢ are curve parameters from your data."
    )
    return AnalysisStage(
        stage_id="objective_formulation",
        title="Stage 5: Objective formulation",
        finding=finding,
        technical_detail=math,
        chart=None,
    )


def stage_6_constraint_identification(
    df: pd.DataFrame,
    channels: list[str],
    user_budget: float | None = None,
) -> AnalysisStage:
    spend_cols = [c for c in channels if c in df.columns]
    if spend_cols:
        daily_total = df[spend_cols].fillna(0).sum(axis=1)
        avg_daily = daily_total.mean()
        B = user_budget if user_budget is not None else float(avg_daily * 260)
        alloc = df[spend_cols].fillna(0).mean()
        chart = px.bar(x=alloc.index, y=alloc.values, title="Average historical spend by channel")
    else:
        B = user_budget or 50000.0
        chart = None

    soft_notes = []
    for col in spend_cols:
        mx = df[col].max()
        if mx > 0 and df[col].quantile(0.99) < mx * 0.5:
            soft_notes.append(f"{col} rarely exceeds ${df[col].quantile(0.99):,.0f}")

    soft_cap_note = (
        "; ".join(soft_notes) if soft_notes else "No tight historical caps detected."
    )

    finding = (
        f"We identified **3** constraints:\n"
        f"1. **Total budget**: cannot exceed **${B:,.0f}** across all channels "
        f"({'user-provided' if user_budget else 'estimated from avg daily spend × 260 trading days'}).\n"
        f"2. **Non-negativity**: no channel can receive negative spend.\n"
        f"3. **Historical caps**: {soft_cap_note}"
    )
    return AnalysisStage(
        stage_id="constraint_identification",
        title="Stage 6: Constraint identification",
        finding=finding,
        technical_detail="Constraints: Σxᵢ ≤ B, xᵢ ≥ 0, optional soft caps from historical maxima.",
        chart=chart,
    )


def stage_7_user_confirmation(result: BackwardAnalysisResult) -> AnalysisStage:
    constraints = "\n".join(f"- {c}" for c in result.constraint_text) if result.constraint_text else ""
    finding = (
        "**Summary — please confirm before optimization runs:**\n\n"
        f"{result.objective_function_text}\n\n"
        f"**Constraints:**\n{constraints}\n\n"
        f"Estimated budget: **${result.detected_budget:,.0f}**"
        if result.detected_budget
        else ""
    )
    return AnalysisStage(
        stage_id="user_confirmation",
        title="Stage 7: Your confirmation",
        finding=finding,
        technical_detail=result.objective_function_math,
        chart=None,
        confirmed=False,
    )


def run_backward_analysis(
    df: pd.DataFrame,
    user_budget: float | None = None,
    config_path: str = "config.yaml",
) -> BackwardAnalysisResult:
    """Run all 7 backward analysis stages."""
    config = load_config(config_path)
    logger.info("Starting backward analysis")

    target = _resolve_target(df)
    modeled_keys, dropped_keys, _ = classify_channels(df, config)
    spend_cols = analysis_spend_columns(df, config)
    if not spend_cols:
        raise ValueError("No modeled spend columns found for backward analysis.")

    result = BackwardAnalysisResult(target_column=target, spend_columns=spend_cols)

    s1 = stage_1_outcome_identification(df)
    logger.info("Completed stage 1: %s", s1.stage_id)
    s2 = stage_2_channel_detection(df, config)
    channels = spend_cols
    s3 = stage_3_spend_response_relationship(df, channels, target)
    s4 = stage_4_saturation_check(df, channels, target)
    s5 = stage_5_objective_formulation(modeled_keys or channels)
    s6 = stage_6_constraint_identification(df, channels, user_budget)

    spend_cols = [c for c in channels if c in df.columns]
    if spend_cols:
        avg_daily = df[spend_cols].fillna(0).sum(axis=1).mean()
        period_days = _budget_period_days(config)
        result.detected_budget = (
            user_budget
            if user_budget is not None
            else float(avg_daily * period_days)
        )
    else:
        result.detected_budget = user_budget or float(config["optimization"]["default_budget"])

    result.objective_function_text = s5.finding
    result.objective_function_math = s5.technical_detail
    result.constraint_text = [
        f"Total budget ≤ ${result.detected_budget:,.0f}",
        "Non-negativity: xᵢ ≥ 0 for all channels",
        "Soft caps from historical spend patterns (if applicable)",
    ]

    s7 = stage_7_user_confirmation(result)
    result.stages = [s1, s2, s3, s4, s5, s6, s7]
    result.confirmed_by_user = False
    logger.info("Backward analysis complete — awaiting user confirmation")
    return result
