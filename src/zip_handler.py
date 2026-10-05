
from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.channel_policy import (
    DROPPED_CHANNEL_REASONS,
    classify_channels,
    display_name,
)

logger = logging.getLogger(__name__)

TARGET_CANDIDATE_NAMES = {
    "ALL_PURCHASES",
    "CONVERSIONS",
    "REVENUE",
    "PURCHASES",
    "SALES",
    "ALL_PURCHASES_ORIGINAL_PRICE",
}
TIMESERIES_CANDIDATES = {"MMM_TIMESERIES_ID", "TIMESERIES_ID", "SERIES_ID"}
DATE_CANDIDATES = {"DATE_DAY", "DATE", "DAY"}


@dataclass
class SchemaProfile:
    """Full dataset profile shown to user before any cleaning begins."""

    filenames_in_zip: list[str]
    selected_file: str
    n_rows: int
    n_columns: int
    column_names: list[str]
    date_columns: list[str]
    spend_columns: list[str]
    target_candidates: list[str]
    currency_column: str | None
    null_summary: dict[str, float]
    detected_channels: list[str]
    dropped_channels: list[str]
    duplicate_count: int
    key_duplicate_count: int
    data_dictionary: pd.DataFrame | None = None
    confirmation_required: bool = True


def load_upload(upload_bytes: bytes, filename: str) -> dict[str, pd.DataFrame | None]:
    """Single entry point for zip or bare CSV uploads."""
    name = filename.lower()
    if name.endswith(".zip"):
        return _load_from_zip(upload_bytes)
    if name.endswith(".csv"):
        return _load_bare_csv(upload_bytes, filename)
    raise ValueError(
        f"Unsupported file type: {filename}. Accepted formats: .zip, .csv"
    )


def _load_from_zip(upload_bytes: bytes) -> dict:
    """Open zip in memory; extract CSV and optional xlsx dictionary."""
    try:
        with zipfile.ZipFile(io.BytesIO(upload_bytes)) as zf:
            names = zf.namelist()
            csv_files: dict[str, pd.DataFrame] = {}
            dictionary: pd.DataFrame | None = None

            for name in names:
                if name.endswith("/") or name.startswith("__MACOSX"):
                    continue
                lower = name.lower()
                if lower.endswith(".csv"):
                    try:
                        with zf.open(name) as f:
                            csv_files[Path(name).name] = pd.read_csv(f)
                    except Exception as exc:
                        logger.warning("Skipping unparseable CSV %s: %s", name, exc)
                elif lower.endswith(".xlsx") and dictionary is None:
                    try:
                        with zf.open(name) as f:
                            dictionary = pd.read_excel(f)
                    except Exception as exc:
                        logger.warning("Skipping unparseable XLSX %s: %s", name, exc)

            if not csv_files:
                raise ValueError("No CSV files found inside the zip archive.")
            return {"csv_files": csv_files, "dictionary": dictionary}
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid zip archive.") from exc


def _load_bare_csv(upload_bytes: bytes, filename: str) -> dict:
    """Parse raw CSV bytes directly."""
    if not upload_bytes or not upload_bytes.strip():
        raise ValueError(f"Could not parse CSV file '{filename}'.")
    try:
        df = pd.read_csv(io.BytesIO(upload_bytes))
        if df.empty or len(df.columns) == 0:
            raise ValueError(f"Could not parse CSV file '{filename}'.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse CSV file '{filename}'.") from exc
    return {"csv_files": {filename: df}, "dictionary": None}


def _pct_null(series: pd.Series) -> float:
    return float(series.isna().mean())


def _detect_timeseries_col(columns: list[str]) -> str | None:
    for c in columns:
        if c.upper() in TIMESERIES_CANDIDATES or "TIMESERIES" in c.upper():
            return c
    return None


def _detect_date_col(columns: list[str], df: pd.DataFrame) -> str | None:
    for c in columns:
        if "DATE" in c.upper():
            return c
    for c in columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
        try:
            pd.to_datetime(df[c].dropna().head(5), errors="raise")
            return c
        except (ValueError, TypeError):
            continue
    return None


def auto_detect_schema(
    df: pd.DataFrame,
    filename: str,
    dictionary: pd.DataFrame | None = None,
) -> SchemaProfile:
    """Inspect df and build a SchemaProfile."""
    columns = list(df.columns)
    n_rows = len(df)

    date_columns = [c for c in columns if "DATE" in c.upper()]
    spend_columns = [c for c in columns if "SPEND" in c.upper()]

    target_candidates = [
        c for c in columns if c.upper() in TARGET_CANDIDATE_NAMES
    ]
    for c in columns:
        if c in target_candidates:
            continue
        if "SPEND" in c.upper():
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and _pct_null(df[c]) < 0.05:
            target_candidates.append(c)

    currency_column = None
    for c in columns:
        if c.upper() in ("CURRENCY", "CURRENCY_CODE"):
            currency_column = c
            break

    null_summary = {c: _pct_null(df[c]) for c in columns}

    detected_channels, dropped_channels, _ = classify_channels(df)

    ts_col = _detect_timeseries_col(columns)
    date_col = _detect_date_col(columns, df)
    key_duplicate_count = 0
    if ts_col and date_col:
        key_duplicate_count = int(df.duplicated(subset=[ts_col, date_col]).sum())

    if dictionary is not None and not dictionary.empty:
        desc_cols = [c for c in dictionary.columns if "desc" in c.lower() or "column" in c.lower()]
        if desc_cols:
            logger.info("Data dictionary loaded with %d rows", len(dictionary))

    return SchemaProfile(
        filenames_in_zip=[filename],
        selected_file=filename,
        n_rows=n_rows,
        n_columns=len(columns),
        column_names=columns,
        date_columns=date_columns or ([date_col] if date_col else []),
        spend_columns=spend_columns,
        target_candidates=target_candidates or ["ALL_PURCHASES"],
        currency_column=currency_column,
        null_summary=null_summary,
        detected_channels=detected_channels,
        dropped_channels=dropped_channels,
        duplicate_count=int(df.duplicated().sum()),
        key_duplicate_count=key_duplicate_count,
        data_dictionary=dictionary,
        confirmation_required=True,
    )


def render_schema_confirmation(profile: SchemaProfile) -> dict:
    """Convert SchemaProfile into a Streamlit-renderable dict."""
    columns_table = []
    for col in profile.column_names:
        pct = profile.null_summary.get(col, 0.0)
        if col in profile.spend_columns:
            role = "spend"
        elif col in profile.target_candidates:
            role = "target"
        elif col in profile.date_columns:
            role = "date"
        elif col == profile.currency_column:
            role = "currency"
        else:
            role = "other"
        columns_table.append(
            {"column": col, "pct_null": f"{pct:.1%}", "role": role, "description": ""}
        )

    channels_dropped = [
        {
            "channel": display_name(ch),
            "reason": DROPPED_CHANNEL_REASONS.get(ch, "too sparse — excluded from model"),
        }
        for ch in profile.dropped_channels
    ]

    warnings: list[str] = []
    if profile.duplicate_count > 0:
        warnings.append(f"Found {profile.duplicate_count} exact duplicate rows.")
    if profile.key_duplicate_count > 0:
        warnings.append(
            f"Found {profile.key_duplicate_count} duplicate (timeseries, date) keys."
        )

    summary = (
        f"Dataset `{profile.selected_file}`: {profile.n_rows:,} rows, "
        f"{profile.n_columns} columns. "
        f"Modeling {len(profile.detected_channels)} channels; "
        f"excluding {len(profile.dropped_channels)} sparse channels."
    )

    return {
        "summary": summary,
        "columns_table": columns_table,
        "channels_to_model": [display_name(ch) for ch in profile.detected_channels],
        "channels_dropped": channels_dropped,
        "warnings": warnings,
        "confirmation_prompt": (
            "Review the profile above. Confirm the target variable and budget, "
            "then click **Confirm and proceed** to run the cleaning pipeline."
        ),
        "dictionary_preview": (
            profile.data_dictionary.head(20) if profile.data_dictionary is not None else None
        ),
    }


def confirm_and_save(
    df: pd.DataFrame,
    profile: SchemaProfile,
    output_raw_path: str = "data/raw/uploaded_dataset.csv",
) -> tuple[pd.DataFrame, str]:
    """Save confirmed dataset to disk after user confirmation."""
    path = Path(output_raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    abs_path = str(path.resolve())

    if profile.data_dictionary is not None:
        dict_path = path.parent / "data_dictionary.xlsx"
        profile.data_dictionary.to_excel(dict_path, index=False)
        logger.info("Saved data dictionary to %s", dict_path)

    logger.info("User confirmed schema. Saved to %s.", abs_path)
    return df, abs_path
