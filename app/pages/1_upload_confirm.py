
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.channel_policy import display_name
from src.data_prep import run_pipeline
from src.zip_handler import (
    auto_detect_schema,
    confirm_and_save,
    load_upload,
    render_schema_confirmation,
)

# Average weeks per month/year — used to convert an entered budget to the
# weekly scale the optimizer (and Ana's κ/u_c) operate on.
_WEEKS_PER_PERIOD = {"Weekly": 1.0, "Monthly": 52.0 / 12.0, "Annual": 52.0}


def _to_weekly_budget(amount: float, period: str) -> float:
    """Convert a budget entered for ``period`` into a weekly budget."""
    return float(amount) / _WEEKS_PER_PERIOD.get(period, 1.0)


def _invalidate_downstream() -> None:
    """Clear cached backward-analysis/optimization results.

    A changed budget (or dataset) must force those stages to recompute —
    otherwise the new budget is silently ignored because they only rerun when
    their cached state is empty/unconfirmed.
    """
    for key in (
        "backward_analysis_result",
        "optim_result",
        "optim_result_B",
        "channel_params",
        "pending_param_change",
    ):
        st.session_state[key] = None
    st.session_state.backward_analysis_confirmed = False
    st.session_state.optimization_complete = False
    st.session_state.params_dirty = False
    st.session_state.activation_ceilings = {}
    st.session_state.adstock_lambdas = {}


st.header("Step 1: Upload your marketing dataset")

# -----------------------------------------------------------------------------
# Live-demo quickstart — for recruiters, reviewers, and first-time visitors
# who don't have a Conjura-style CSV on hand. Loads a small pre-committed
# sample and runs the full pipeline in one click.
# -----------------------------------------------------------------------------
DEMO_CSV = ROOT / "data" / "sample" / "demo.csv"

with st.container():
    st.markdown(
        "** First time here?** Click below to load a small sample of the "
        "Conjura eCommerce dataset and skip straight to the recommendation."
    )
    demo_clicked = st.button(
        "Load demo data and run the full pipeline",
        type="primary",
        use_container_width=False,
        disabled=not DEMO_CSV.exists(),
        help=(
            f"Reads {DEMO_CSV.name} (~5,000 rows) → cleans → fits MMM → "
            "optimizes → populates every page. Takes about 30 seconds."
        ),
    )

    if demo_clicked:
        with st.spinner("Loading demo data and running the full pipeline..."):
            raw_target = ROOT / "data" / "raw" / "demo.csv"
            raw_target.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(DEMO_CSV, raw_target)

            pipeline_result = run_pipeline(raw_path=str(raw_target))

            # Populate session state so downstream pages behave as if a normal
            # upload+confirm just finished.
            st.session_state.raw_path = str(raw_target)
            st.session_state.cleaned_df = pipeline_result["ready_df"]
            st.session_state.train_df = pipeline_result["train_df"]
            st.session_state.test_df = pipeline_result["test_df"]
            st.session_state.eda_report = pipeline_result["eda_report"]
            st.session_state.upload_complete = True
            st.session_state.schema_confirmed = True
            st.session_state.phase = "analysis"
            st.session_state.confirmed_target = "y"
            st.session_state.confirmed_budget = 831_000.0
            st.session_state.confirmed_budget_period = "Weekly"
            st.session_state.demo_mode = True
            _invalidate_downstream()

        st.success(
            "✅ Demo data loaded. Continue to **Step 2: Backward Analysis** "
            "in the sidebar, or jump ahead to the results on any downstream page."
        )
        if st.button("Continue to Step 2: Backward Analysis"):
            st.switch_page("pages/2_backward_analysis.py")

st.divider()

st.markdown(
    """
Or upload your own dataset in any of these formats:
- **`.zip` file** containing your CSV + an optional `.xlsx` data dictionary
- **`.csv` file** directly

The file should include a date column, spend columns per channel, and a conversions column.
"""
)

uploaded_file = st.file_uploader(
    label="Upload dataset (.zip or .csv)",
    type=["zip", "csv"],
    help="Accepted: .zip (with CSV + optional XLSX dictionary) or bare .csv. Maximum 200 MB.",
)

# Already have a dataset loaded? Let the user change just the budget without
# re-uploading or re-cleaning — the cleaned data and splits are reused as-is.
if st.session_state.get("upload_complete") and st.session_state.get("cleaned_df") is not None:
    current = st.session_state.get("confirmed_budget")
    with st.expander("Update budget only (keep the current dataset)", expanded=False):
        st.caption(
            "Use this to re-run with a different budget on the dataset you already "
            "uploaded. No need to upload again."
        )
        if current is not None:
            st.markdown(f"**Current weekly budget:** ${current:,.0f}")
        with st.form("update_budget_only"):
            bo_period = st.radio(
                "Budget period:",
                options=["Weekly", "Monthly", "Annual"],
                index=0,
                horizontal=True,
            )
            bo_amount = st.number_input(
                "New total budget ($) for the selected period:",
                min_value=1000,
                max_value=100_000_000,
                value=int(current) if current else 831_000,
                step=1000,
            )
            bo_submit = st.form_submit_button("Update budget and re-run pipeline")

        if bo_submit:
            new_weekly = _to_weekly_budget(bo_amount, bo_period)
            st.session_state.confirmed_budget = new_weekly
            st.session_state.confirmed_budget_period = bo_period
            _invalidate_downstream()
            st.success(f"Budget updated to a weekly budget of ${new_weekly:,.0f}.")
            if bo_period != "Weekly":
                st.caption(
                    f"Converted {bo_period.lower()} budget ${bo_amount:,.0f} "
                    f"to ${new_weekly:,.0f}/week."
                )
            st.info(
                "Now open **Backward Analysis** and click **Confirm and run "
                "optimization** to recompute results with the new budget."
            )
            if st.button("Go to Backward Analysis"):
                st.switch_page("pages/2_backward_analysis.py")

if uploaded_file is not None:
    with st.spinner("Reading your dataset..."):
        try:
            result = load_upload(uploaded_file.read(), uploaded_file.name)
        except ValueError as e:
            st.error(f"Could not read the file: {e}")
            st.stop()

    csv_files = result["csv_files"]
    dictionary = result["dictionary"]

    if len(csv_files) > 1:
        selected = st.selectbox(
            "Multiple CSV files found in your zip. Which one should we use?",
            options=list(csv_files.keys()),
        )
    else:
        selected = list(csv_files.keys())[0]

    df = csv_files[selected]

    if dictionary is not None:
        st.success(f"Data dictionary found and loaded ({len(dictionary)} entries).")
        with st.expander("Preview data dictionary"):
            st.dataframe(dictionary.head(20), use_container_width=True)
    else:
        st.info("No data dictionary found. Column roles will be auto-detected from column names.")

    profile = auto_detect_schema(df, selected, dictionary=dictionary)
    confirmation_data = render_schema_confirmation(profile)

    st.success(f"Found **{profile.n_rows:,} rows** and **{profile.n_columns} columns** in `{selected}`.")

    st.subheader("Dataset profile")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{profile.n_rows:,}")
    col2.metric("Columns", profile.n_columns)
    col3.metric("Exact duplicates", profile.duplicate_count)

    st.subheader("Column roles detected")
    if confirmation_data.get("columns_table"):
        st.dataframe(
            pd.DataFrame(confirmation_data["columns_table"]),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Marketing channels")
    st.markdown(
        "**Channels to model:** "
        + (", ".join(display_name(ch) for ch in profile.detected_channels) or "None detected")
    )
    if profile.dropped_channels:
        st.markdown(
            "**Channels excluded** (too sparse): "
            + ", ".join(display_name(ch) for ch in profile.dropped_channels)
        )

    if confirmation_data.get("warnings"):
        for w in confirmation_data["warnings"]:
            st.warning(w)

    st.subheader("Confirm before proceeding")
    st.markdown(confirmation_data.get("confirmation_prompt", ""))

    with st.form("confirm_schema"):
        target_col = st.selectbox(
            "Target variable (what we're optimizing for):",
            options=profile.target_candidates,
            index=0,
        )
        budget_period = st.radio(
            "Budget period:",
            options=["Weekly", "Monthly", "Annual"],
            index=0,
            horizontal=True,
            help=(
                "The model optimizes weekly spend. Pick the period your number "
                "is in — we convert it to a weekly budget automatically."
            ),
        )
        budget_input = st.number_input(
            "Total budget ($) for the selected period:",
            min_value=1000,
            max_value=100_000_000,
            value=831_000,
            step=1000,
        )
        confirmed = st.form_submit_button("Confirm and proceed to analysis")

    if confirmed:
        with st.spinner("Saving confirmed dataset..."):
            df, raw_path = confirm_and_save(df, profile)

        with st.spinner("Cleaning dataset..."):
            pipeline_result = run_pipeline(raw_path=raw_path)

        st.session_state.raw_path = raw_path
        st.session_state.cleaned_df = pipeline_result["ready_df"]
        st.session_state.train_df = pipeline_result["train_df"]
        st.session_state.test_df = pipeline_result["test_df"]
        st.session_state.eda_report = pipeline_result["eda_report"]
        st.session_state.schema_profile = profile
        st.session_state.upload_complete = True
        st.session_state.schema_confirmed = True
        st.session_state.phase = "analysis"
        st.session_state.confirmed_target = target_col
        # Model A/B/C, κ and u_c are all fit at weekly scale, so the optimizer
        # expects a WEEKLY budget. Convert the user's entry to weekly.
        weekly_budget = _to_weekly_budget(budget_input, budget_period)
        st.session_state.confirmed_budget = weekly_budget
        st.session_state.confirmed_budget_period = budget_period

        # Re-confirming invalidates any cached results — otherwise a changed
        # budget/target is ignored because backward analysis and optimization
        # are only recomputed when their cached state is empty/unconfirmed.
        _invalidate_downstream()

        st.success("Dataset cleaned and ready.")
        if budget_period != "Weekly":
            st.caption(
                f"Converted {budget_period.lower()} budget ${budget_input:,.0f} "
                f"to a weekly budget of ${weekly_budget:,.0f} for optimization."
            )

        st.subheader("Download your cleaned files")
        c1, c2, c3 = st.columns(3)
        c1.download_button(
            label="Full cleaned dataset (mmm_ready.csv)",
            data=pipeline_result["ready_csv_bytes"],
            file_name="mmm_ready.csv",
            mime="text/csv",
        )
        c2.download_button(
            label="Training split (mmm_train.csv)",
            data=pipeline_result["train_csv_bytes"],
            file_name="mmm_train.csv",
            mime="text/csv",
        )
        c3.download_button(
            label="Holdout split (mmm_test.csv)",
            data=pipeline_result["test_csv_bytes"],
            file_name="mmm_test.csv",
            mime="text/csv",
        )

        st.subheader("EDA — Exploratory Data Analysis")
        eda = pipeline_result["eda_report"]
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Spend over time", "Channel coverage", "Target variable", "Correlation"]
        )
        with tab1:
            st.plotly_chart(eda["spend_over_time"], use_container_width=True)
        with tab2:
            st.plotly_chart(eda["channel_coverage"], use_container_width=True)
            st.plotly_chart(eda["null_heatmap"], use_container_width=True)
        with tab3:
            st.plotly_chart(eda["target_over_time"], use_container_width=True)
        with tab4:
            st.plotly_chart(eda["correlation_matrix"], use_container_width=True)
            st.plotly_chart(eda["spend_distribution"], use_container_width=True)

        with st.expander("Summary statistics"):
            st.dataframe(eda["summary_stats"], use_container_width=True)

        st.info(
            f"Dataset: **{eda['n_rows']:,} rows** · "
            f"**{eda['n_channels']} channels** · "
            f"**{eda['timeseries_count']} time series** · "
            f"Date range: {eda['date_range'][0]} → {eda['date_range'][1]}"
        )

        st.divider()
        if st.button("Proceed to Step 2: Backward Analysis"):
            st.switch_page("pages/2_backward_analysis.py")
