"""Shared pytest fixtures — synthetic data only."""
from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta

import pandas as pd
import pytest


@pytest.fixture
def sample_mmm_df() -> pd.DataFrame:
    """Minimal conjura-like dataset for pipeline tests."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    rows = []
    for ts in ("TS1", "TS2"):
        for d in dates:
            rows.append(
                {
                    "MMM_TIMESERIES_ID": ts,
                    "ORGANISATION_ID": "ORG1",
                    "ORGANISATION_VERTICAL": None,
                    "ORGANISATION_SUBVERTICAL": None,
                    "DATE_DAY": d.strftime("%Y-%m-%d"),
                    "CURRENCY_CODE": "USD" if ts == "TS1" else "GBP",
                    "ALL_PURCHASES": 100,
                    "GOOGLE_PAID_SEARCH_SPEND": 10.0,
                    "GOOGLE_SHOPPING_SPEND": 5.0,
                    "GOOGLE_PMAX_SPEND": 2.0,
                    "META_FACEBOOK_SPEND": 8.0,
                    "META_INSTAGRAM_SPEND": 4.0,
                    "GOOGLE_DISPLAY_SPEND": None,
                    "TIKTOK_SPEND": None,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_csv_bytes(sample_mmm_df) -> bytes:
    return sample_mmm_df.to_csv(index=False).encode("utf-8")


@pytest.fixture
def sample_zip_bytes(sample_csv_bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("marketing_data.csv", sample_csv_bytes)
    return buf.getvalue()


@pytest.fixture
def sample_zip_with_dict_bytes(sample_csv_bytes) -> bytes:
    dict_df = pd.DataFrame(
        {"column": ["ALL_PURCHASES"], "description": ["Total conversions"]}
    )
    dict_buf = io.BytesIO()
    dict_df.to_excel(dict_buf, index=False)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("marketing_data.csv", sample_csv_bytes)
        zf.writestr("dictionary.xlsx", dict_buf.getvalue())
    return buf.getvalue()


@pytest.fixture
def cleaned_df(sample_mmm_df, tmp_path) -> pd.DataFrame:
    """Run a minimal pipeline on temp raw file."""
    from src.data_prep import (
        aggregate_channels,
        apply_adstock,
        fill_date_gaps,
        fill_vertical_nulls,
        handle_spend_nulls,
        load_config,
        load_raw,
        normalize_currency,
        remove_duplicates,
        validate_spend_non_negative,
        winsorize_spend,
    )

    raw = tmp_path / "raw.csv"
    sample_mmm_df.to_csv(raw, index=False)
    config = load_config()
    df = load_raw(str(raw), config)
    df = remove_duplicates(df, config)
    df = normalize_currency(df, config["fx_rates"])
    df = fill_vertical_nulls(df, config)
    df = handle_spend_nulls(df, config=config)
    df = validate_spend_non_negative(df)
    df = winsorize_spend(df, config=config)
    df = fill_date_gaps(df, config)
    df = apply_adstock(df, config["adstock"]["decay_rates"], config)
    return aggregate_channels(df, config)
