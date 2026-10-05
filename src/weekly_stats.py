
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow `python src/weekly_stats.py` to resolve the `src` package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_prep import load_config, resolve_project_path  # noqa: E402

logger = logging.getLogger(__name__)

# Professor's example kappa activation thresholds (USD/week) — startup
# defaults only; runtime kappa is passed in as a function argument.
DEFAULT_KAPPA: dict[str, int] = {
    "google_paid_search": 18_000,
    "google_shopping": 15_000,
    "google_pmax": 18_000,
    "meta_facebook": 12_000,
    "meta_instagram": 12_000,
}
KAPPA_SUM = sum(DEFAULT_KAPPA.values())  # 75_000

# Mid-market target for recommended weekly budget B.
B_TARGET = 180_000
# Optional scenario budget just above kappa sum: forces real ON/OFF tradeoffs.
B_SCENARIO_ACTIVATION = 90_000
NULL_WARN_THRESHOLD = 0.05

DEFAULT_RAW_PATH = "data/raw/conjura_mmm_data.csv"


# ----------------------------------------------------------------------
# Pipeline output verification
# ----------------------------------------------------------------------
def verify_pipeline_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict | None = None,
) -> dict:
    """Check required columns, null rates, and currency on pipeline outputs."""
    config = config or load_config()
    date_col = config["data"]["date_column"]
    column_map = config["column_map"]
    modeled = config["channels"]["modeled"]

    spend_cols = [column_map[ch] for ch in modeled]
    adstock_cols = [f"{ch}_adstock" for ch in modeled]

    missing = [
        col for col in [date_col, *spend_cols, *adstock_cols, "y"]
        if col not in train_df.columns
    ]

    warnings: list[str] = []
    null_pct = train_df.isna().mean()
    high_null = null_pct[null_pct > NULL_WARN_THRESHOLD]
    for col, pct in high_null.items():
        warnings.append(f"{col}: {pct:.1%} nulls in train_df")

    currency_ok = True
    if "CURRENCY_CODE" in train_df.columns:
        non_usd = train_df.loc[
            train_df["CURRENCY_CODE"].notna() & (train_df["CURRENCY_CODE"] != "USD")
        ]
        if len(non_usd):
            currency_ok = False
            warnings.append(f"{len(non_usd)} non-USD rows remain in train_df")

    adstock_present = [c for c in adstock_cols if c in train_df.columns]
    return {
        "pipeline_verified": not missing and currency_ok,
        "missing_columns": missing,
        "adstock_cols_present": adstock_present,
        "warnings": warnings,
        "spend_cols": spend_cols,
        "currency_ok": currency_ok,
        "date_col_present": date_col in train_df.columns,
        "y_present": "y" in train_df.columns,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }


# ----------------------------------------------------------------------
# Weekly aggregation
# ----------------------------------------------------------------------
def weekly_portfolio(
    df: pd.DataFrame,
    spend_cols: list[str],
    config: dict | None = None,
) -> pd.DataFrame:
    """Sum daily spend (and y) across all timeseries, resampled to weekly."""
    config = config or load_config()
    date_col = config["data"]["date_column"]
    cols = [c for c in spend_cols if c in df.columns]
    agg_cols = cols + (["y"] if "y" in df.columns else [])
    return (
        df.assign(**{date_col: pd.to_datetime(df[date_col])})
        .set_index(date_col)[agg_cols]
        .resample("W")
        .sum()
    )


def compute_weekly_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict | None = None,
) -> dict:
    """
    Weekly spend stats for Greg (MMM fit) and Meghna (B, u_c, kappa).
    Called automatically by data_prep.run_pipeline().
    Returns per-channel weekly min/median/max, B_raw, week counts, y mean.
    """
    config = config or load_config()
    column_map = config["column_map"]
    modeled = config["channels"]["modeled"]
    spend_cols = [column_map[ch] for ch in modeled]

    weekly_train = weekly_portfolio(train_df, spend_cols, config)
    weekly_test = weekly_portfolio(test_df, spend_cols, config)

    per_channel: dict[str, dict[str, float]] = {}
    for ch in modeled:
        col = column_map[ch]
        if col not in weekly_train.columns:
            per_channel[ch] = {"min": 0.0, "median": 0.0, "max": 0.0}
            continue
        s = weekly_train[col]
        per_channel[ch] = {
            "min": float(s.min()),
            "median": float(s.median()),
            "max": float(s.max()),
        }

    present = [c for c in spend_cols if c in weekly_train.columns]
    total_weekly = weekly_train[present].sum(axis=1)
    b_raw = float(total_weekly.mean()) if len(total_weekly) else 0.0
    weekly_y_mean = float(weekly_train["y"].mean()) if "y" in weekly_train.columns else 0.0

    return {
        "per_channel_weekly": per_channel,
        "B_raw": b_raw,
        "train_weeks": int(len(weekly_train)),
        "holdout_weeks": int(len(weekly_test)),
        "weekly_y_mean": weekly_y_mean,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "_weekly_train": weekly_train,
    }


def compute_uc_ceilings(
    per_channel_weekly: dict[str, dict[str, float]],
    kappa: dict[str, float] | None = None,
) -> dict:
    """
    u_c[c] = 1.5 x max historical weekly spend per channel.
    Flags channels where u_c < kappa (will always be OFF in Model B).
    Called automatically by data_prep.run_pipeline().

    `kappa` defaults to DEFAULT_KAPPA; pass runtime thresholds
    (st.session_state['activation_thresholds']) to override.
    """
    kappa = kappa or DEFAULT_KAPPA
    uc: dict[str, float] = {}
    uc_warnings: list[str] = []
    for ch in kappa:
        weekly_max = per_channel_weekly.get(ch, {}).get("max", 0.0)
        uc[ch] = 1.5 * weekly_max
        if uc[ch] < kappa[ch]:
            uc_warnings.append(f"{ch}: u_c < kappa — will always be OFF")
    return {"uc_ceilings": uc, "uc_warnings": uc_warnings, "kappa": dict(kappa)}


def scale_decision(
    b_raw: float,
    kappa: dict[str, float] | None = None,
) -> dict:
    """
    Decide whether to scale spend down to the mid-market range (D2).

    `kappa` defaults to DEFAULT_KAPPA; the >3x activation-sum check uses
    the sum of the kappa actually provided.
    """
    kappa = kappa or DEFAULT_KAPPA
    kappa_sum = sum(kappa.values())
    scale_factor = min(1.0, B_TARGET / b_raw) if b_raw > 0 else 1.0
    return {
        "B_raw": b_raw,
        "B_recommended": b_raw * scale_factor,
        "scale_factor": scale_factor,
        "scale_down": b_raw > 3 * kappa_sum,
        "kappa_sum": kappa_sum,
    }


def write_handoff(
    weekly_stats: dict,
    uc_result: dict,
    config: dict,
    verification: dict | None = None,
    out_path: Path | None = None,
    kappa: dict[str, float] | None = None,
) -> dict:
    """
    Write data/processed/weekly_handoff.json (+ weekly_stats.json and,
    when scaling applies, weekly_scaled_spend.csv) for Greg and Meghna.

    `kappa` defaults to DEFAULT_KAPPA; the JSON's "kappa" key reflects
    the kappa actually used.
    """
    kappa = kappa or DEFAULT_KAPPA
    verification = verification or {
        "pipeline_verified": None,
        "adstock_cols_present": [],
        "warnings": [],
        "spend_cols": [],
    }
    decision = scale_decision(weekly_stats["B_raw"], kappa)
    out_path = out_path or resolve_project_path("data/processed/weekly_handoff.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    handoff = {
        "train_rows": weekly_stats["train_rows"],
        "test_rows": weekly_stats["test_rows"],
        "train_weeks": weekly_stats["train_weeks"],
        "holdout_weeks": weekly_stats["holdout_weeks"],
        "weekly_y_mean": round(weekly_stats["weekly_y_mean"], 2),
        "B_raw": round(decision["B_raw"], 2),
        "B_recommended": round(decision["B_recommended"], 2),
        "scale_factor": round(decision["scale_factor"], 4),
        "kappa": dict(kappa),
        "per_channel_weekly": {
            ch: {k: round(v, 2) for k, v in st.items()}
            for ch, st in weekly_stats["per_channel_weekly"].items()
        },
        "pipeline_verified": verification["pipeline_verified"],
        "adstock_cols_present": verification["adstock_cols_present"],
        "warnings": verification["warnings"],
        "uc_ceilings": {ch: round(v, 2) for ch, v in uc_result["uc_ceilings"].items()},
        "B_portfolio": round(decision["B_raw"], 2),
        "B_scenario_activation": B_SCENARIO_ACTIVATION,
        "uc_warnings": uc_result["uc_warnings"],
    }
    out_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

    stats_path = out_path.parent / "weekly_stats.json"
    stats_path.write_text(
        json.dumps(
            {k: v for k, v in weekly_stats.items() if not k.startswith("_")},
            indent=2,
        ),
        encoding="utf-8",
    )

    weekly_train = weekly_stats.get("_weekly_train")
    if decision["scale_factor"] < 1.0 and weekly_train is not None:
        spend_cols = [c for c in verification.get("spend_cols", []) if c in weekly_train.columns]
        if spend_cols:
            scaled = weekly_train[spend_cols] * decision["scale_factor"]
            scaled.to_csv(out_path.parent / "weekly_scaled_spend.csv")

    logger.info("Weekly handoff written to %s", out_path)
    return handoff


# ----------------------------------------------------------------------
# Display / CLI report
# ----------------------------------------------------------------------
def print_verification(verification: dict, config: dict) -> None:
    date_col = config["data"]["date_column"]
    print("=== Ana Day-0 Verification ===")
    print(f"[PIPELINE] train={verification['train_rows']:,} rows, test={verification['test_rows']:,} rows")
    date_mark = "OK" if verification["date_col_present"] else "MISSING"
    spend_present = len(verification["spend_cols"]) - sum(
        c in verification["missing_columns"] for c in verification["spend_cols"]
    )
    y_mark = "OK" if verification["y_present"] else "MISSING"
    print(
        f"[COLUMNS]  {date_col}: {date_mark}  |  spend cols: {spend_present}/5"
        f"  |  adstock cols: {len(verification['adstock_cols_present'])}/5  |  y: {y_mark}"
    )
    if verification["missing_columns"]:
        print(f"[COLUMNS]  MISSING: {', '.join(verification['missing_columns'])}")
    print(f"[CURRENCY] {'All USD' if verification['currency_ok'] else 'NON-USD ROWS REMAIN'}")
    print(f"[WARNINGS] {'; '.join(verification['warnings']) if verification['warnings'] else 'none'}")
    print()


def print_weekly_stats(stats: dict) -> None:
    from src.channel_policy import display_name

    print("=== Weekly Spend Stats (USD) ===")
    header = f"{'Channel':<25}| {'Min':>12} | {'Median':>12} | {'Max':>12}"
    print(header)
    print("-" * len(header))
    for ch, st in stats["per_channel_weekly"].items():
        print(
            f"{display_name(ch):<25}| ${st['min']:>11,.0f} | ${st['median']:>11,.0f} | ${st['max']:>11,.0f}"
        )
    print()
    print(f"Total portfolio (mean/wk): ${stats['B_raw']:,.0f}")
    print(f"Train weeks : {stats['train_weeks']}  |  Holdout weeks : {stats['holdout_weeks']}")
    print(f"Weekly y mean: {stats['weekly_y_mean']:,.0f}")
    print()


def print_uc_ceilings(uc_result: dict) -> None:
    from src.channel_policy import display_name

    uc = uc_result["uc_ceilings"]
    kappa_map = uc_result.get("kappa", DEFAULT_KAPPA)
    print("=== Channel Ceilings u_c (for Meghna -> activation.ceilings) ===")
    header = f"{'Channel':<25}| {'kappa (min ON)':>14} | {'u_c (max ON)':>14} | {'Ratio u_c/kappa':>15}"
    print(header)
    print("-" * len(header))
    for ch, kappa in kappa_map.items():
        ratio = uc[ch] / kappa if kappa else 0.0
        print(
            f"{display_name(ch):<25}| ${kappa:>13,.0f} | ${uc[ch]:>13,.0f} | {ratio:>14.1f}x"
        )
    print()
    if uc_result["uc_warnings"]:
        for w in uc_result["uc_warnings"]:
            print(f"  [FLAG] {w}")
    else:
        print("  All channels: u_c >= kappa — every channel CAN activate in Model B.")
    print()


def print_portfolio_b(b_portfolio: float) -> None:
    print("=== Portfolio B Recommendation ===")
    print(f"  Mean weekly portfolio spend (train): ${b_portfolio:,.0f}")
    print("  -> Set optimization.default_budget = this value in config.yaml")
    print("  -> At this B, Model B likely ~= Model A (kappa won't force channels OFF)")
    print()
    print("  Optional scenario B for activation story:")
    print(f"    Sum of all kappa = ${KAPPA_SUM:,.0f}/week")
    print("    A budget just above kappa_sum forces real ON/OFF tradeoffs.")
    print(f"    -> Consider B_scenario = ${B_SCENARIO_ACTIVATION:,.0f} for the Model B write-up.")
    print()


def print_scale_decision(decision: dict) -> None:
    kappa_sum = decision.get("kappa_sum", KAPPA_SUM)
    ratio = decision["B_raw"] / kappa_sum if kappa_sum else 0.0
    verdict = "Scale down to mid-market" if decision["scale_down"] else "Use full scale"
    print("=== Scale Decision (D2) ===")
    print(f"  Raw mean weekly portfolio spend : ${decision['B_raw']:,.0f}")
    print(f"  kappa sum (all 5 channels ON)   : ${kappa_sum:,.0f}")
    print(f"  Ratio                           : {ratio:.1f}x")
    print(f"  Recommended B                   : ${decision['B_recommended']:,.0f}")
    print(f"  Scale factor applied            : {decision['scale_factor']:.2f}")
    print(f"  -> {verdict}")
    print()


def print_adstock_ownership() -> None:
    print("=== Adstock Ownership ===")
    print("  [x] fill_date_gaps()  -> daily continuity (data_prep.py, Ana - done)")
    print("  [x] apply_adstock()   -> uses config.adstock.decay_rates as starting lambda")
    print("                           (data_prep.py, Ana - done)")
    print("  [x] lambda estimation -> bo_mmm_tuning.bayesian_optimize_mmm tunes per-channel")
    print("                           decay from holdout (bo_mmm_tuning.py, Meghna)")
    print("  [x] Weekly holdout    -> chronological last-N-weeks split on aggregated weekly")
    print("                           series owned by Greg inside mmm_model.py")
    print("  [ ] split_train_test  -> NOT changed by Ana; daily per-timeseries split")
    print("                           remains as-is")
    print()


def main(raw_path: str | None = None) -> dict:
    """Run the pipeline (which computes and writes the handoff) and print the report."""
    from src.data_prep import run_pipeline

    try:  # keep currency/marker output safe on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    config = load_config()
    if raw_path is None:
        if resolve_project_path(DEFAULT_RAW_PATH).exists():
            raw_path = DEFAULT_RAW_PATH
        else:
            raw_path = config["data"]["raw_path"]
            print(f"[NOTE] {DEFAULT_RAW_PATH} not found; using config raw_path: {raw_path}\n")

    res = run_pipeline(raw_path=raw_path)

    print_verification(res["verification"], config)
    print_weekly_stats(res["weekly_stats"])
    print_uc_ceilings(res["uc_result"])
    print_portfolio_b(res["weekly_stats"]["B_raw"])
    print_scale_decision(scale_decision(res["weekly_stats"]["B_raw"]))
    print_adstock_ownership()
    print("=== Handoff written -> data/processed/weekly_handoff.json ===")
    return res["handoff"]


if __name__ == "__main__":
    main(raw_path=sys.argv[1] if len(sys.argv) > 1 else None)
