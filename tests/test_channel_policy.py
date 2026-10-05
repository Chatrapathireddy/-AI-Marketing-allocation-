"""Tests for channel inclusion policy."""
from __future__ import annotations

from src.channel_policy import classify_channels


def test_classify_channels_modeled_and_dropped(sample_mmm_df):
    modeled, dropped, raw_cols = classify_channels(sample_mmm_df)
    assert "google_paid_search" in modeled
    assert "meta_instagram" in modeled
    assert "tiktok" in dropped
    assert "google_display" in dropped
    assert "meta_instagram" not in dropped
    assert "GOOGLE_PAID_SEARCH_SPEND" in raw_cols
    assert "TIKTOK_SPEND" not in raw_cols
