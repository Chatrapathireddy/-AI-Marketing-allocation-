
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

logger = logging.getLogger(__name__)


def project_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a repo-relative path from the project root."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def load_config(path: str = "config.yaml") -> dict:
    """Load project configuration from YAML (works from notebooks/ or repo root)."""
    config_file = resolve_project_path(path)
    with open(config_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _spend_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "SPEND" in c.upper() or c.endswith("_adstock")]


def load_raw(path: str, config: dict | None = None) -> pd.DataFrame:
    """Load raw CSV; parse DATE_DAY; assert non-negative spend."""
    config = config or load_config()
    date_col = config["data"]["date_column"]
    df = pd.read_csv(path)
    logger.info("Loaded raw data shape=%s dtypes=%s", df.shape, df.dtypes.to_dict())

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        valid = df[date_col].dropna()
        if len(valid):
            logger.info("Date range: %s to %s", valid.min(), valid.max())

    spend_cols = [c for c in df.columns if "SPEND" in c.upper()]
    for col in spend_cols:
        neg = df[df[col] < 0]
        if len(neg):
            raise ValueError(
                f"Negative spend in {col}: rows {neg.index.tolist()[:5]}"
            )
    return df


def remove_duplicates(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Check exact and key duplicates."""
    config = config or load_config()
    ts_col = config["data"].get("timeseries_column", "MMM_TIMESERIES_ID")
    date_col = config["data"]["date_column"]

    exact = df.duplicated()
    if exact.any():
        dup_rows = df[exact].head(5)
        raise ValueError(f"Exact duplicate rows found: {len(df[exact])}. Sample:\n{dup_rows}")

    if ts_col in df.columns and date_col in df.columns:
        key_dup = df.duplicated(subset=[ts_col, date_col])
        if key_dup.any():
            raise ValueError(
                f"Duplicate keys on ({ts_col}, {date_col}): {int(key_dup.sum())} rows"
            )

    logger.info("Duplicate check passed: 0 exact duplicates, 0 key duplicates.")
    return df


def normalize_currency(df: pd.DataFrame, fx_rates: dict) -> pd.DataFrame:
    """Convert spend and revenue columns to USD."""
    if "CURRENCY_CODE" not in df.columns:
        logger.warning("No CURRENCY_CODE column; skipping currency normalization.")
        return df

    df = df.copy()
    df["CURRENCY_CODE_ORIGINAL"] = df["CURRENCY_CODE"]
    currencies = df["CURRENCY_CODE"].dropna().unique()
    logger.info("Currencies found: %s", list(currencies))

    numeric_cols = [
        c
        for c in df.columns
        if ("SPEND" in c.upper() or "PURCHASE" in c.upper() or "REVENUE" in c.upper())
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in numeric_cols:
        df[col] = df[col].astype("float64")

    for curr in currencies:
        if curr not in fx_rates:
            raise ValueError(f"Unknown currency code: {curr}")
        rate = fx_rates[curr]
        mask = df["CURRENCY_CODE"] == curr
        n = int(mask.sum())
        for col in numeric_cols:
            df.loc[mask, col] = df.loc[mask, col] * rate
        logger.info("Converted %d rows from %s at rate %s", n, curr, rate)

    df["CURRENCY_CODE"] = "USD"
    return df


def fill_vertical_nulls(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Forward-fill verticals within org, then Unknown."""
    config = config or load_config()
    org_col = config["data"].get("organisation_column", "ORGANISATION_ID")
    date_col = config["data"]["date_column"]
    df = df.copy()

    for col in ("ORGANISATION_VERTICAL", "ORGANISATION_SUBVERTICAL"):
        if col not in df.columns:
            continue
        before = int(df[col].isna().sum())
        if org_col in df.columns and date_col in df.columns:
            df = df.sort_values([org_col, date_col])
            df[col] = df.groupby(org_col)[col].ffill()
        remaining = int(df[col].isna().sum())
        df[col] = df[col].fillna("Unknown")
        logger.info("%s: filled %d via ffill, %d -> Unknown", col, before - remaining, remaining)
    return df


def handle_spend_nulls(
    df: pd.DataFrame,
    modeled_channels: list[str] | None = None,
    dropped_channels: list[str] | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Fill modeled channel nulls with 0; drop sparse columns."""
    config = config or load_config()
    column_map = config.get("column_map", {})
    modeled = modeled_channels or config["channels"]["modeled"]
    dropped_cols = config.get("dropped_columns", [])

    df = df.copy()
    for ch in modeled:
        raw_col = column_map.get(ch, ch.upper())
        if raw_col not in df.columns:
            continue
        before = int(df[raw_col].isna().sum())
        df[raw_col] = df[raw_col].fillna(0)
        logger.info("%s: filled %d nulls with 0", raw_col, before)

    for col in dropped_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.info("Dropped sparse column: %s", col)
    return df


def validate_spend_non_negative(df: pd.DataFrame) -> pd.DataFrame:
    """Assert all spend columns >= 0."""
    spend_cols = [c for c in df.columns if "SPEND" in c.upper()]
    for col in spend_cols:
        neg = df[df[col] < 0]
        if len(neg):
            raise ValueError(f"Negative values in {col} at rows {neg.index.tolist()[:5]}")
    logger.info("Non-negativity check passed.")
    return df


def winsorize_spend(df: pd.DataFrame, percentile: float = 0.99, config: dict | None = None) -> pd.DataFrame:
    """Cap spend at per-(timeseries, channel) percentile."""
    config = config or load_config()
    ts_col = config["data"].get("timeseries_column", "MMM_TIMESERIES_ID")
    column_map = config.get("column_map", {})
    modeled = config["channels"]["modeled"]
    df = df.copy()

    for ch in modeled:
        col = column_map.get(ch, ch)
        if col not in df.columns:
            continue
        affected = 0
        if ts_col in df.columns:
            for _, grp in df.groupby(ts_col):
                cap = grp[col].quantile(percentile)
                mask = df.index.isin(grp.index) & (df[col] > cap)
                affected += int(mask.sum())
                df.loc[mask, col] = cap
        else:
            cap = df[col].quantile(percentile)
            mask = df[col] > cap
            affected = int(mask.sum())
            df.loc[mask, col] = cap
        logger.info("Winsorized %s: %d rows capped at %.2f percentile", col, affected, percentile)
    return df


def fill_date_gaps(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Complete daily date range per timeseries; spend=0, target=NaN on gaps."""
    config = config or load_config()
    ts_col = config["data"].get("timeseries_column", "MMM_TIMESERIES_ID")
    date_col = config["data"]["date_column"]
    target_col = config["data"]["target_column"]
    spend_cols = [c for c in df.columns if "SPEND" in c.upper()]

    if ts_col not in df.columns or date_col not in df.columns:
        return df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    parts = []
    added_total = 0

    for ts_id, grp in df.groupby(ts_col):
        grp = grp.sort_values(date_col)
        idx = pd.date_range(grp[date_col].min(), grp[date_col].max(), freq="D")
        reindexed = grp.set_index(date_col).reindex(idx)
        reindexed[ts_col] = ts_id
        reindexed = reindexed.reset_index()
        if reindexed.columns[0] != date_col:
            reindexed = reindexed.rename(columns={reindexed.columns[0]: date_col})

        new_rows = len(reindexed) - len(grp)
        added_total += new_rows
        for col in spend_cols:
            if col in reindexed.columns:
                reindexed[col] = reindexed[col].fillna(0)
        if target_col in reindexed.columns:
            pass  # leave NaN for imputed dates
        parts.append(reindexed)

    out = pd.concat(parts, ignore_index=True)
    logger.info("Date gap fill: added %d rows across timeseries", added_total)
    return out


def apply_adstock(df: pd.DataFrame, decay_rates: dict, config: dict | None = None) -> pd.DataFrame:
    """Geometric adstock per channel per timeseries."""
    config = config or load_config()
    ts_col = config["data"].get("timeseries_column", "MMM_TIMESERIES_ID")
    column_map = config.get("column_map", {})
    df = df.copy()

    date_col = config["data"]["date_column"]
    for ch, decay in decay_rates.items():
        spend_col = column_map.get(ch, ch)
        if spend_col not in df.columns:
            continue
        adstock_col = f"{ch}_adstock"
        df[adstock_col] = 0.0
        if ts_col in df.columns:
            for _, grp in df.groupby(ts_col, sort=False):
                sort_cols = [date_col] if date_col in grp.columns else []
                ordered = grp.sort_values(sort_cols) if sort_cols else grp
                idx = ordered.index
                s = ordered[spend_col].fillna(0).values
                ad = np.zeros(len(s))
                for i in range(len(s)):
                    ad[i] = s[i] + (decay * ad[i - 1] if i > 0 else 0)
                df.loc[idx, adstock_col] = ad
        else:
            s = df[spend_col].fillna(0).values
            ad = np.zeros(len(s))
            for i in range(len(s)):
                ad[i] = s[i] + (decay * ad[i - 1] if i > 0 else 0)
            df[adstock_col] = ad
        logger.info("Applied adstock to %s (decay=%.2f)", ch, decay)
    return df


def aggregate_channels(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Aggregate google/meta spend; rename target to y."""
    config = config or load_config()
    modeled = config["channels"]["modeled"]
    df = df.copy()

    google_cols = [f"{ch}_adstock" for ch in modeled if ch.startswith("google")]
    meta_cols = [f"{ch}_adstock" for ch in modeled if ch.startswith("meta")]
    existing_google = [c for c in google_cols if c in df.columns]
    existing_meta = [c for c in meta_cols if c in df.columns]

    if existing_google:
        df["google_spend"] = df[existing_google].sum(axis=1)
    if existing_meta:
        df["meta_spend"] = df[existing_meta].sum(axis=1)

    target = config["data"]["target_column"]
    if target in df.columns:
        df["y"] = df[target]
    return df


def export_processed(df: pd.DataFrame, output_path: str) -> bytes:
    """Write CSV and return bytes for Streamlit download."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    csv_bytes = path.read_bytes()
    nulls = df.isna().sum()
    logger.info(
        "Exported shape=%s cols=%s nulls=%s",
        df.shape,
        list(df.columns),
        nulls[nulls > 0].to_dict(),
    )
    return csv_bytes


def generate_eda_report(df: pd.DataFrame, config: dict | None = None) -> dict:
    """Produce EDA summary with Plotly figures."""
    config = config or load_config()
    date_col = config["data"]["date_column"]
    target_col = "y" if "y" in df.columns else config["data"]["target_column"]
    column_map = config.get("column_map", {})
    modeled = config["channels"]["modeled"]

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    summary_stats = df[numeric].describe() if numeric else pd.DataFrame()

    null_pct = df.isna().mean() * 100
    null_heatmap = go.Figure(
        data=go.Heatmap(
            z=[null_pct.values],
            x=list(null_pct.index),
            y=["null_%"],
            colorscale="Reds",
        )
    )
    null_heatmap.update_layout(title="Null % per column")

    spend_cols = [column_map.get(ch, ch) for ch in modeled if column_map.get(ch, ch) in df.columns]
    adstock_cols = [f"{ch}_adstock" for ch in modeled if f"{ch}_adstock" in df.columns]
    plot_cols = adstock_cols or spend_cols

    if date_col in df.columns and plot_cols:
        daily = df.groupby(date_col)[plot_cols].sum().reset_index()
        spend_over_time = px.area(
            daily,
            x=date_col,
            y=plot_cols,
            title="Total daily spend by channel (adstock)",
        )
    else:
        spend_over_time = go.Figure()

    if date_col in df.columns and target_col in df.columns:
        target_daily = df.groupby(date_col)[target_col].sum().reset_index()
        target_over_time = px.line(
            target_daily,
            x=date_col,
            y=target_col,
            title=f"{target_col} over time",
        )
    else:
        target_over_time = go.Figure()

    from src.channel_policy import display_name

    coverage = {}
    for ch in modeled:
        col = column_map.get(ch, ch)
        if col in df.columns:
            coverage[display_name(ch)] = 1.0 - df[col].isna().mean()
    channel_coverage = px.bar(
        x=list(coverage.keys()),
        y=list(coverage.values()),
        labels={"x": "channel", "y": "non_null_pct"},
        title="Modeled channel coverage (non-null %)",
    )

    corr_cols = [c for c in plot_cols + ([target_col] if target_col in df.columns else []) if c in df.columns]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr()
        correlation_matrix = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdBu",
                zmid=0,
            )
        )
        correlation_matrix.update_layout(title="Spend vs target correlation")
    else:
        correlation_matrix = go.Figure()

    if plot_cols:
        spend_distribution = px.box(df, y=plot_cols, title="Spend distribution by channel")
    else:
        spend_distribution = go.Figure()

    currencies = (
        list(df["CURRENCY_CODE_ORIGINAL"].dropna().unique())
        if "CURRENCY_CODE_ORIGINAL" in df.columns
        else (list(df["CURRENCY_CODE"].dropna().unique()) if "CURRENCY_CODE" in df.columns else ["USD"])
    )
    ts_col = config["data"].get("timeseries_column", "MMM_TIMESERIES_ID")
    timeseries_count = df[ts_col].nunique() if ts_col in df.columns else 1
    date_range = ("", "")
    if date_col in df.columns:
        d = pd.to_datetime(df[date_col]).dropna()
        if len(d):
            date_range = (str(d.min().date()), str(d.max().date()))

    return {
        "summary_stats": summary_stats,
        "null_heatmap": null_heatmap,
        "spend_over_time": spend_over_time,
        "target_over_time": target_over_time,
        "channel_coverage": channel_coverage,
        "correlation_matrix": correlation_matrix,
        "spend_distribution": spend_distribution,
        "n_rows": len(df),
        "n_channels": len(modeled),
        "date_range": date_range,
        "currencies_found": [str(c) for c in currencies],
        "timeseries_count": int(timeseries_count),
    }


def split_train_test(
    df: pd.DataFrame,
    test_months: int = 3,
    config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-timeseries holdout: last test_months calendar months."""
    config = config or load_config()
    ts_col = config["data"].get("timeseries_column", "MMM_TIMESERIES_ID")
    date_col = config["data"]["date_column"]
    test_months = test_months or config["data"]["test_months"]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    train_parts, test_parts = [], []

    if ts_col not in df.columns:
        cutoff = df[date_col].max() - pd.DateOffset(months=test_months)
        train_parts.append(df[df[date_col] < cutoff])
        test_parts.append(df[df[date_col] >= cutoff])
    else:
        for _, grp in df.groupby(ts_col):
            grp = grp.sort_values(date_col)
            span_days = (grp[date_col].max() - grp[date_col].min()).days
            if span_days < test_months * 28:
                cutoff = grp[date_col].quantile(0.8)
            else:
                cutoff = grp[date_col].max() - pd.DateOffset(months=test_months)
            train_parts.append(grp[grp[date_col] < cutoff])
            test_parts.append(grp[grp[date_col] >= cutoff])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    logger.info(
        "Train/test split: train=%d test=%d",
        len(train_df),
        len(test_df),
    )
    return train_df, test_df


def run_pipeline(config_path: str = "config.yaml", raw_path: str | None = None) -> dict:
    """Master cleaning pipeline."""
    config = load_config(config_path)
    path = str(resolve_project_path(raw_path or config["data"]["raw_path"]))
    abs_raw = str(Path(path).resolve())

    df = load_raw(path, config)
    df = remove_duplicates(df, config)
    df = normalize_currency(df, config["fx_rates"])
    df = fill_vertical_nulls(df, config)
    df = handle_spend_nulls(df, config=config)
    df = validate_spend_non_negative(df)
    df = winsorize_spend(df, config=config)
    df = fill_date_gaps(df, config)
    df = apply_adstock(df, config["adstock"]["decay_rates"], config)
    df = aggregate_channels(df, config)

    ready_bytes = export_processed(df, str(resolve_project_path(config["data"]["processed_path"])))
    eda_report = generate_eda_report(df, config)
    train_df, test_df = split_train_test(
        df, test_months=config["data"]["test_months"], config=config
    )
    train_bytes = export_processed(train_df, str(resolve_project_path(config["data"]["train_path"])))
    test_bytes = export_processed(test_df, str(resolve_project_path(config["data"]["test_path"])))

    # Weekly handoff (weekly_handoff.json).
    # Imported here to avoid a circular import (weekly_stats imports load_config).
    from src.weekly_stats import (
        compute_uc_ceilings,
        compute_weekly_stats,
        verify_pipeline_outputs,
        write_handoff,
    )

    weekly_stats = compute_weekly_stats(train_df, test_df, config)
    uc_result = compute_uc_ceilings(weekly_stats["per_channel_weekly"])
    verification = verify_pipeline_outputs(train_df, test_df, config)
    handoff = write_handoff(weekly_stats, uc_result, config, verification)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "ready_df": df,
        "ready_csv_bytes": ready_bytes,
        "train_csv_bytes": train_bytes,
        "test_csv_bytes": test_bytes,
        "eda_report": eda_report,
        "raw_path": abs_raw,
        "weekly_stats": weekly_stats,
        "uc_result": uc_result,
        "handoff": handoff,
        "verification": verification,
    }


if __name__ == "__main__":
    run_pipeline()
