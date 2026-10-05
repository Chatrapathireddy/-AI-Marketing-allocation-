

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.explainer import (
    diagnose_allocation,
    generate_explanation,
    plot_allocation_bar,
    plot_baseline_lift,
)
from src.optimization_pipeline import apply_optimization_to_session, run_optimization_pipeline

st.title("📊 Recommended Allocation")

with st.expander("🟢 In plain English — what is this page?"):
    st.markdown(
        """
This is the agent's suggested way to **split your budget** across channels.

- **Total Budget Allocated** — how much of your budget actually gets spent.
- **Predicted Conversions** — the results we expect from this split.
- **Baseline Conversions** — what you'd expect if you spent in the *same
  proportions* you did historically.
- **Lift vs Baseline** — how much *better* this recommendation is than that
  baseline.

The bar chart compares the recommended spend to the baseline, and *"Where the
Agent Moved Money"* shows which channels gained or lost dollars and why.
"""
    )

optim_result = st.session_state.get("optim_result")

if optim_result is None:
    if not st.session_state.get("schema_confirmed"):
        st.warning("Complete Step 1 (upload and confirm schema) first.")
        if st.button("Go to Step 1"):
            st.switch_page("pages/1_upload_confirm.py")
        st.stop()

    if not st.session_state.get("backward_analysis_result"):
        st.warning("Complete Step 2 (backward analysis) first.")
        if st.button("Go to Step 2"):
            st.switch_page("pages/2_backward_analysis.py")
        st.stop()

    if not st.session_state.get("backward_analysis_confirmed"):
        st.warning(
            "You have reviewed backward analysis but have not **confirmed** it yet. "
            "On Step 2, scroll to the bottom and click **Confirm and run optimization**."
        )
        if st.button("Go to Step 2"):
            st.switch_page("pages/2_backward_analysis.py")
        st.stop()

    analysis = st.session_state.backward_analysis_result
    st.info(
        "Backward analysis is confirmed. Running the optimizer now — "
        "this usually takes 15–20 seconds."
    )
    try:
        with st.status("Running MMM fit and optimizer…", expanded=True) as status:
            optim, channel_params, budget, model_b, model_c = run_optimization_pipeline(
                confirmed_budget=st.session_state.get("confirmed_budget"),
                detected_budget=getattr(analysis, "detected_budget", None),
                channel_params=st.session_state.get("channel_params"),
                train_df=st.session_state.get("train_df"),
            )
            apply_optimization_to_session(
                st.session_state, optim, channel_params, model_b=model_b, model_c=model_c
            )
            status.update(label="Optimization complete", state="complete", expanded=False)
        st.rerun()
    except Exception as exc:
        st.error(
            "Could not run the optimizer. Return to Step 2 and retry, or re-run "
            f"Step 1 if train/test files are missing.\n\n`{exc}`"
        )
        if st.button("Back to Step 2"):
            st.switch_page("pages/2_backward_analysis.py")
        st.stop()


def _get(obj, attr, default=None):
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return default


# Backfill baseline for sessions optimized before baseline.py was wired
if optim_result is not None and not _get(optim_result, "baseline_allocation"):
    train_df = st.session_state.get("train_df")
    channel_params = st.session_state.get("channel_params")
    if train_df is not None and channel_params:
        from src.baseline import apply_baseline_to_result
        from src.data_prep import load_config

        config = load_config()
        channels = list(config["channels"]["modeled"])
        budget = float(_get(optim_result, "total_spent", 0) or sum(_get(optim_result, "allocation", {}).values()))
        optim_result = apply_baseline_to_result(
            optim_result, channel_params, channels, budget, train_df, config=config
        )
        st.session_state.optim_result = optim_result


allocation = _get(optim_result, "allocation", {})
baseline = _get(optim_result, "baseline_allocation", {})
predicted = _get(optim_result, "predicted_conversions", 0.0)
baseline_conv = _get(optim_result, "baseline_conversions", 0.0)
lift = _get(optim_result, "lift_pct", 0.0)
total_spent = _get(optim_result, "total_spent", sum(allocation.values()))
status = _get(optim_result, "status", "unknown")

# -----------------------------------------------------------------------------
# Headline metrics
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Total Budget Allocated",
    f"${total_spent:,.0f}",
    help="How much of your total budget the optimizer actually spends.",
)
col2.metric(
    "Predicted Conversions",
    f"{predicted:,.1f}",
    help="Expected conversions from the recommended allocation.",
)
col3.metric(
    "Baseline Conversions",
    f"{baseline_conv:,.1f}",
    help="Expected conversions if you spent in the same proportions as your history.",
)
col4.metric(
    "Lift vs Baseline",
    f"{lift:+.1f}%",
    help="How much better the recommendation is than the baseline, in percent.",
)

st.caption(f"Solver status: `{status}`")
if isinstance(status, str) and "KKT pass" in status:
    st.caption(
        "✅ *KKT pass* means the math checks confirm this is a genuinely optimal "
        "way to split the budget given your channels, curves, and constraints."
    )

# -----------------------------------------------------------------------------
# Diagnostics — surface multicollinearity, cap-binding, near-linear fits
# -----------------------------------------------------------------------------
channel_params = st.session_state.get("channel_params")
channel_caps = st.session_state.get("channel_caps")  # set by Meghna's optimizer

diagnostics = diagnose_allocation(
    allocation=allocation,
    channel_params=channel_params,
    channel_caps=channel_caps,
)
if diagnostics:
    with st.expander(f"⚠️ {len(diagnostics)} modeling caveat(s) — read before sharing", expanded=True):
        for d in diagnostics:
            level = d.get("level", "info")
            msg = d.get("message", "")
            if level == "warn":
                st.warning(msg)
            elif level == "error":
                st.error(msg)
            else:
                st.info(msg)

# -----------------------------------------------------------------------------
# Allocation bar chart
# -----------------------------------------------------------------------------
st.subheader("Allocation by Channel")
st.plotly_chart(
    plot_allocation_bar(allocation, baseline=baseline), use_container_width=True
)

# -----------------------------------------------------------------------------
# Detailed table
# -----------------------------------------------------------------------------
st.subheader("Allocation Detail")
detail = pd.DataFrame(
    {
        "Channel": list(allocation.keys()),
        "Recommended": list(allocation.values()),
        "Baseline": [baseline.get(c, 0.0) for c in allocation.keys()],
    }
)
detail["Δ vs Baseline"] = detail["Recommended"] - detail["Baseline"]
detail["% of Budget"] = (
    detail["Recommended"] / max(total_spent, 1.0) * 100.0
)

st.dataframe(
    detail.style.format(
        {
            "Recommended": "${:,.0f}",
            "Baseline": "${:,.0f}",
            "Δ vs Baseline": "${:+,.0f}",
            "% of Budget": "{:.1f}%",
        }
    ),
    use_container_width=True,
)

# -----------------------------------------------------------------------------
# Lift comparison
# -----------------------------------------------------------------------------
st.subheader("Where the Agent Moved Money")
st.plotly_chart(plot_baseline_lift(baseline, allocation), use_container_width=True)

# -----------------------------------------------------------------------------
# Explanation
# -----------------------------------------------------------------------------
st.subheader("💡 Why This Recommendation")
with st.spinner("Generating explanation..."):
    explanation = generate_explanation(
        optim_result,
        params={
            "channels": list(allocation.keys()),
            "budget": total_spent,
            "objective": _get(optim_result, "objective", "conversions"),
        },
    )
st.markdown(explanation)
