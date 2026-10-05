"""Tests for zip_handler — synthetic uploads only."""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from src.zip_handler import (
    auto_detect_schema,
    confirm_and_save,
    load_upload,
    render_schema_confirmation,
)


def test_load_zip_with_csv_and_dictionary_returns_both(sample_zip_with_dict_bytes):
    result = load_upload(sample_zip_with_dict_bytes, "data.zip")
    assert "csv_files" in result
    assert len(result["csv_files"]) >= 1
    assert result["dictionary"] is not None


def test_dictionary_stored_in_schema_profile(sample_mmm_df, sample_zip_with_dict_bytes):
    result = load_upload(sample_zip_with_dict_bytes, "data.zip")
    df = list(result["csv_files"].values())[0]
    profile = auto_detect_schema(df, "marketing_data.csv", dictionary=result["dictionary"])
    assert profile.data_dictionary is not None


def test_dictionary_enriches_column_role_detection(sample_mmm_df):
    dictionary = pd.DataFrame({"column": ["ALL_PURCHASES"], "description": ["conversions"]})
    profile = auto_detect_schema(sample_mmm_df, "data.csv", dictionary=dictionary)
    assert "ALL_PURCHASES" in profile.target_candidates


def test_load_zip_csv_only_returns_dataframe(sample_zip_bytes):
    result = load_upload(sample_zip_bytes, "data.zip")
    assert len(result["csv_files"]) == 1
    assert isinstance(list(result["csv_files"].values())[0], pd.DataFrame)


def test_load_zip_csv_only_dictionary_is_none(sample_zip_bytes):
    result = load_upload(sample_zip_bytes, "data.zip")
    assert result["dictionary"] is None


def test_load_zip_raises_on_invalid_bytes():
    with pytest.raises(ValueError, match="zip"):
        load_upload(b"not a zip", "bad.zip")


def test_load_zip_raises_if_no_csv_inside():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "hello")
    with pytest.raises(ValueError, match="No CSV"):
        load_upload(buf.getvalue(), "empty.zip")


def test_load_zip_multiple_csvs_returns_all(sample_csv_bytes):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.csv", sample_csv_bytes)
        zf.writestr("b.csv", sample_csv_bytes)
    result = load_upload(buf.getvalue(), "multi.zip")
    assert len(result["csv_files"]) == 2


def test_load_bare_csv_returns_dataframe(sample_csv_bytes):
    result = load_upload(sample_csv_bytes, "data.csv")
    assert len(result["csv_files"]) == 1


def test_load_bare_csv_dictionary_is_none(sample_csv_bytes):
    result = load_upload(sample_csv_bytes, "data.csv")
    assert result["dictionary"] is None


def test_load_bare_csv_raises_on_unparseable_bytes():
    with pytest.raises((ValueError, pd.errors.ParserError)):
        load_upload(b"", "bad.csv")


def test_load_upload_raises_on_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported"):
        load_upload(b"x", "file.parquet")


def test_auto_detect_schema_finds_spend_columns(sample_mmm_df):
    profile = auto_detect_schema(sample_mmm_df, "data.csv")
    assert len(profile.spend_columns) >= 3


def test_auto_detect_schema_flags_sparse_channels(sample_mmm_df):
    profile = auto_detect_schema(sample_mmm_df, "data.csv")
    assert "tiktok" in profile.dropped_channels
    assert "google_display" in profile.dropped_channels
    assert "meta_instagram" in profile.detected_channels
    assert "meta_instagram" not in profile.dropped_channels


def test_auto_detect_schema_detects_date_column(sample_mmm_df):
    profile = auto_detect_schema(sample_mmm_df, "data.csv")
    assert "DATE_DAY" in profile.date_columns


def test_auto_detect_schema_counts_duplicates(sample_mmm_df):
    dup = pd.concat([sample_mmm_df, sample_mmm_df.iloc[:1]])
    profile = auto_detect_schema(dup, "data.csv")
    assert profile.duplicate_count >= 1


def test_render_schema_confirmation_returns_keys(sample_mmm_df):
    profile = auto_detect_schema(sample_mmm_df, "data.csv")
    data = render_schema_confirmation(profile)
    assert "columns_table" in data
    assert "channels_to_model" in data


def test_confirm_and_save_writes_file(sample_mmm_df, tmp_path):
    profile = auto_detect_schema(sample_mmm_df, "data.csv")
    out = tmp_path / "uploaded.csv"
    df, path = confirm_and_save(sample_mmm_df, profile, str(out))
    assert out.exists()
    assert path == str(out.resolve())
