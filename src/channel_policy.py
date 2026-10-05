
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def _load_config(path: str = "config.yaml") -> dict:
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / path if not Path(path).is_absolute() else Path(path)
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

CHANNEL_DISPLAY_NAMES: dict[str, str] = {
    "google_paid_search": "Google Paid Search",
    "google_shopping": "Google Shopping",
    "google_pmax": "Google PMax",
    "meta_facebook": "Meta Facebook",
    "meta_instagram": "Meta Instagram",
    "google_display": "Google Display",
    "google_video": "Google Video",
    "meta_other": "Meta Other",
    "tiktok": "TikTok",
}

DROPPED_CHANNEL_REASONS: dict[str, str] = {
    "google_display": "85.9% null — too sparse for reliable saturation curves",
    "google_video": "94.3% null — too sparse for reliable saturation curves",
    "meta_other": "82.3% null — too sparse for reliable saturation curves",
    "tiktok": "97.4% null — excluded from the model",
}


def display_name(channel_key: str) -> str:
    return CHANNEL_DISPLAY_NAMES.get(channel_key, channel_key.replace("_", " ").title())


def classify_channels(
    df: pd.DataFrame,
    config: dict | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Classify channels per project policy (not dynamic 90% threshold).

    Returns:
        modeled_keys: logical channel ids to model (present in df)
        dropped_keys: sparse channels to exclude (present in df)
        modeled_raw_columns: CSV spend column names for modeled channels
    """
    config = config or _load_config()
    column_map = config.get("column_map", {})
    modeled_cfg = config["channels"]["modeled"]
    dropped_cfg = config["channels"]["dropped_sparse"]

    modeled_keys: list[str] = []
    modeled_raw: list[str] = []
    for ch in modeled_cfg:
        raw = column_map.get(ch)
        if raw and raw in df.columns:
            modeled_keys.append(ch)
            modeled_raw.append(raw)

    dropped_keys: list[str] = []
    for ch in dropped_cfg:
        raw = column_map.get(ch)
        if raw and raw in df.columns:
            dropped_keys.append(ch)

    return modeled_keys, dropped_keys, modeled_raw


def analysis_spend_columns(df: pd.DataFrame, config: dict | None = None) -> list[str]:
    """Spend columns for backward analysis / correlation (prefer adstock columns)."""
    config = config or _load_config()
    column_map = config.get("column_map", {})
    cols: list[str] = []
    for ch in config["channels"]["modeled"]:
        adstock_col = f"{ch}_adstock"
        raw_col = column_map.get(ch)
        if adstock_col in df.columns:
            cols.append(adstock_col)
        elif raw_col and raw_col in df.columns:
            cols.append(raw_col)
    return cols


def format_modeled_channel_sentence(modeled_keys: list[str]) -> str:
    names = [display_name(k) for k in modeled_keys]
    if not names:
        return "no modeled spend columns found"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def format_dropped_channel_sentence(dropped_keys: list[str]) -> str:
    if not dropped_keys:
        return "none"
    parts = [f"{display_name(k)} ({DROPPED_CHANNEL_REASONS.get(k, 'too sparse')})" for k in dropped_keys]
    return "; ".join(parts)
