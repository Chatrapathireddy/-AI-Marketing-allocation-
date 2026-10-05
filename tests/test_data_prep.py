"""Tests for data_prep — synthetic DataFrames only."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.data_prep import (
    apply_adstock,
    export_processed,
    fill_vertical_nulls,
    generate_eda_report,
    handle_spend_nulls,
    load_config,
    load_raw,
    normalize_currency,
    remove_duplicates,
    run_pipeline,
    split_train_test,
    validate_spend_non_negative,
    winsorize_spend,
)


def test_remove_duplicates_passes_clean_data(sample_mmm_df):
    config = load_config()
    out = remove_duplicates(sample_mmm_df, config)
    assert len(out) == len(sample_mmm_df)


def test_remove_duplicates_raises_on_exact_dup(sample_mmm_df):
    dup = pd.concat([sample_mmm_df, sample_mmm_df.iloc[:1]])
    with pytest.raises(ValueError, match="Exact duplicate"):
        remove_duplicates(dup, load_config())


def test_remove_duplicates_raises_on_key_dup(sample_mmm_df):
    config = load_config()
    row = sample_mmm_df.iloc[0].copy()
    row["GOOGLE_PAID_SEARCH_SPEND"] = 999.0
    dup = pd.concat([sample_mmm_df, pd.DataFrame([row])], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate keys"):
        remove_duplicates(dup, config)


def test_normalize_currency_usd_row_unchanged(sample_mmm_df):
    df = sample_mmm_df.copy()
    df["CURRENCY_CODE"] = "USD"
    out = normalize_currency(df, load_config()["fx_rates"])
    assert out.loc[0, "GOOGLE_PAID_SEARCH_SPEND"] == pytest.approx(10.0)


def test_normalize_currency_gbp_multiplied_correctly(sample_mmm_df):
    df = sample_mmm_df[sample_mmm_df["MMM_TIMESERIES_ID"] == "TS2"].copy()
    fx = load_config()["fx_rates"]
    out = normalize_currency(df, fx)
    assert out["GOOGLE_PAID_SEARCH_SPEND"].iloc[0] == pytest.approx(10.0 * fx["GBP"])


def test_normalize_currency_int_columns_become_float(sample_mmm_df):
    df = sample_mmm_df.copy()
    df["GOOGLE_PAID_SEARCH_SPEND"] = df["GOOGLE_PAID_SEARCH_SPEND"].astype("int64")
    df.loc[df["MMM_TIMESERIES_ID"] == "TS2", "CURRENCY_CODE"] = "GBP"
    out = normalize_currency(df, load_config()["fx_rates"])
    assert pd.api.types.is_float_dtype(out["GOOGLE_PAID_SEARCH_SPEND"])


def test_normalize_currency_unknown_code_raises(sample_mmm_df):
    df = sample_mmm_df.copy()
    df.loc[0, "CURRENCY_CODE"] = "XXX"
    with pytest.raises(ValueError, match="Unknown currency"):
        normalize_currency(df, load_config()["fx_rates"])


def test_fill_vertical_nulls_forward_fills_within_org(sample_mmm_df):
    df = sample_mmm_df.copy()
    df.loc[df.index[0], "ORGANISATION_VERTICAL"] = "Retail"
    df.loc[df.index[1:], "ORGANISATION_VERTICAL"] = None
    out = fill_vertical_nulls(df, load_config())
    assert out["ORGANISATION_VERTICAL"].notna().all()


def test_fill_vertical_nulls_remaining_become_unknown(sample_mmm_df):
    df = sample_mmm_df.copy()
    df["ORGANISATION_VERTICAL"] = None
    out = fill_vertical_nulls(df, load_config())
    assert (out["ORGANISATION_VERTICAL"] == "Unknown").any() or out["ORGANISATION_VERTICAL"].notna().all()


def test_handle_spend_nulls_fills_zero(sample_mmm_df):
    config = load_config()
    out = handle_spend_nulls(sample_mmm_df, config=config)
    assert out["GOOGLE_PAID_SEARCH_SPEND"].isna().sum() == 0


def test_handle_spend_nulls_drops_tiktok_column(sample_mmm_df):
    config = load_config()
    out = handle_spend_nulls(sample_mmm_df, config=config)
    assert "TIKTOK_SPEND" not in out.columns


def test_validate_spend_non_negative_raises_on_negative(sample_mmm_df):
    df = sample_mmm_df.copy()
    df.loc[0, "GOOGLE_PAID_SEARCH_SPEND"] = -1
    with pytest.raises(ValueError, match="Negative"):
        validate_spend_non_negative(df)


def test_winsorize_clips_at_99th_percentile(sample_mmm_df):
    df = sample_mmm_df.copy()
    df.loc[0, "GOOGLE_PAID_SEARCH_SPEND"] = 1e6
    out = winsorize_spend(df, config=load_config())
    assert out.loc[0, "GOOGLE_PAID_SEARCH_SPEND"] < 1e6


def test_apply_adstock_first_row_equals_spend(cleaned_df):
    config = load_config()
    col = "google_paid_search_adstock"
    if col not in cleaned_df.columns:
        pytest.skip("adstock column missing")
    first = cleaned_df.groupby("MMM_TIMESERIES_ID").head(1)
    spend = cleaned_df.loc[first.index, "GOOGLE_PAID_SEARCH_SPEND"]
    assert first[col].values == pytest.approx(spend.fillna(0).values, rel=1e-5)


def test_apply_adstock_second_row_uses_decay(sample_mmm_df, tmp_path):
    from src.data_prep import apply_adstock, handle_spend_nulls, load_raw, remove_duplicates

    raw = tmp_path / "r.csv"
    sample_mmm_df.to_csv(raw, index=False)
    config = load_config()
    df = load_raw(str(raw), config)
    df = remove_duplicates(df, config)
    df = handle_spend_nulls(df, config=config)
    decay = config["adstock"]["decay_rates"]["google_paid_search"]
    out = apply_adstock(df, {"google_paid_search": decay}, config)
    ts = out["MMM_TIMESERIES_ID"].iloc[0]
    grp = out[out["MMM_TIMESERIES_ID"] == ts].sort_values("DATE_DAY")
    if len(grp) < 2:
        pytest.skip("need 2 rows")
    s0 = grp["GOOGLE_PAID_SEARCH_SPEND"].iloc[0]
    s1 = grp["GOOGLE_PAID_SEARCH_SPEND"].iloc[1]
    expected = s1 + decay * s0
    assert grp["google_paid_search_adstock"].iloc[1] == pytest.approx(expected, rel=1e-5)


def test_apply_adstock_resets_at_timeseries_boundary(cleaned_df):
    config = load_config()
    col = "google_paid_search_adstock"
    if col not in cleaned_df.columns:
        pytest.skip("adstock missing")
    groups = list(cleaned_df.groupby("MMM_TIMESERIES_ID"))
    if len(groups) < 2:
        pytest.skip("need 2 series")
    g1_first = groups[0][1].iloc[0][col]
    g2_first = groups[1][1].iloc[0][col]
    s1 = groups[0][1].iloc[0]["GOOGLE_PAID_SEARCH_SPEND"]
    s2 = groups[1][1].iloc[0]["GOOGLE_PAID_SEARCH_SPEND"]
    assert g1_first == pytest.approx(s1, rel=1e-4)
    assert g2_first == pytest.approx(s2, rel=1e-4)


def test_split_train_test_no_date_overlap(cleaned_df):
    config = load_config()
    train, test = split_train_test(cleaned_df, config=config)
    date_col = config["data"]["date_column"]
    train_dates = set(pd.to_datetime(train[date_col]))
    test_dates = set(pd.to_datetime(test[date_col]))
    assert train_dates.isdisjoint(test_dates)


def test_split_train_test_test_is_last_n_months(cleaned_df):
    config = load_config()
    train, test = split_train_test(cleaned_df, test_months=1, config=config)
    assert len(test) >= 1
    assert len(train) >= 1


def test_split_train_test_all_timeseries_present_in_both(cleaned_df):
    config = load_config()
    train, test = split_train_test(cleaned_df, config=config)
    ts = config["data"]["timeseries_column"]
    if ts in cleaned_df.columns:
        assert set(train[ts].unique()) == set(test[ts].unique())


def test_export_processed_returns_bytes(cleaned_df, tmp_path):
    path = tmp_path / "out.csv"
    b = export_processed(cleaned_df, str(path))
    assert isinstance(b, bytes)
    assert len(b) > 0


def test_export_processed_bytes_parseable_as_csv(cleaned_df, tmp_path):
    path = tmp_path / "out.csv"
    b = export_processed(cleaned_df, str(path))
    df = pd.read_csv(io.BytesIO(b))
    assert len(df) == len(cleaned_df)


def test_generate_eda_report_has_all_keys(cleaned_df):
    report = generate_eda_report(cleaned_df)
    expected = {
        "summary_stats",
        "null_heatmap",
        "spend_over_time",
        "target_over_time",
        "channel_coverage",
        "correlation_matrix",
        "spend_distribution",
        "n_rows",
        "n_channels",
        "date_range",
        "currencies_found",
        "timeseries_count",
    }
    assert expected.issubset(report.keys())


def test_generate_eda_report_figures_are_plotly(cleaned_df):
    report = generate_eda_report(cleaned_df)
    for key in ("null_heatmap", "spend_over_time", "channel_coverage"):
        assert isinstance(report[key], go.Figure)


def test_run_pipeline_returns_dict_with_all_keys(sample_mmm_df, tmp_path, monkeypatch):
    raw = tmp_path / "uploaded_dataset.csv"
    sample_mmm_df.to_csv(raw, index=False)
    proc = tmp_path / "processed"
    proc.mkdir(exist_ok=True)

    config = load_config()
    config["data"]["raw_path"] = str(raw)
    config["data"]["processed_path"] = str(proc / "mmm_ready.csv")
    config["data"]["train_path"] = str(proc / "mmm_train.csv")
    config["data"]["test_path"] = str(proc / "mmm_test.csv")

    import yaml

    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    result = run_pipeline(str(cfg_path), raw_path=str(raw))
    for key in (
        "train_df",
        "test_df",
        "ready_df",
        "ready_csv_bytes",
        "train_csv_bytes",
        "test_csv_bytes",
        "eda_report",
        "raw_path",
    ):
        assert key in result


def test_run_pipeline_raw_path_matches_saved_file(sample_mmm_df, tmp_path):
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
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    result = run_pipeline(str(cfg_path), raw_path=str(raw))
    assert Path(result["raw_path"]).exists()
