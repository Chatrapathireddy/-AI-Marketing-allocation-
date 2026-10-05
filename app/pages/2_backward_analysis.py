
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.backward_analysis import run_backward_analysis, strip_stage_charts
from src.optimization_pipeline import apply_optimization_to_session, run_optimization_pipeline

st.header("Step 2: Backward Analysis")
st.markdown(
    """
Before optimizing, we work **backward from your observed outcomes** to understand
what your data is telling us — identifying your objective function and
constraints from the data itself.
"""
)

with st.expander("🟢 In plain English — what is this page?"):
    st.markdown(
        """
Before recommending where to spend, we reverse-engineer your data to make sure
we're solving the **right** problem. Each stage answers one simple question:

- **What are we trying to grow?** (e.g. purchases)
- **Which channels actually matter?**
- **Does spending more really drive results?**
- **Where do returns start to flatten out?**
- **What's the total budget limit?**

At the very end you click **Confirm** — and only then does the optimizer run.
Nothing is decided behind your back.
"""
    )

if not st.session_state.get("schema_confirmed"):
    st.warning("Complete Step 1 first.")
    st.stop()

df = st.session_state.get("cleaned_df")
if df is None:
    st.error("No dataset in session. Please re-upload.")
    st.stop()

if st.session_state.get("backward_analysis_result") is None:
    with st.spinner("Running backward analysis..."):
        result = run_backward_analysis(
            df=df,
            user_budget=st.session_state.get("confirmed_budget"),
        )
    st.session_state.backward_analysis_result = result
else:
    result = st.session_state.backward_analysis_result

if st.session_state.get("backward_analysis_confirmed"):
    strip_stage_charts(result)
    st.session_state.backward_analysis_result = result

confirmed = st.session_state.get("backward_analysis_confirmed", False)
show_charts = not confirmed

for i, stage in enumerate(result.stages):
    with st.expander(
        f"**{stage.title}**",
        expanded=(not confirmed and i == len(result.stages) - 1),
    ):
        st.markdown(stage.finding)
        st.caption(stage.technical_detail)
        if show_charts and stage.chart is not None:
            st.plotly_chart(stage.chart, use_container_width=True)

if confirmed and not show_charts:
    st.caption("Charts hidden after confirmation to keep the page responsive.")

st.divider()
st.subheader("Confirm your optimization setup")
st.markdown(result.objective_function_text)
st.code(result.objective_function_math, language="text")
for c in result.constraint_text:
    st.markdown(f"- {c}")

st.caption(
    "Reviewing all 7 stages above is not enough — click the button below to "
    "confirm and run the optimizer (~15–20 seconds)."
)


def _execute_optimization():
    optim, channel_params, budget, model_b, model_c = run_optimization_pipeline(
        confirmed_budget=st.session_state.get("confirmed_budget"),
        detected_budget=result.detected_budget,
        channel_params=st.session_state.get("channel_params"),
        train_df=st.session_state.get("train_df"),
    )
    apply_optimization_to_session(
        st.session_state, optim, channel_params, model_b=model_b, model_c=model_c
    )
    return optim, budget


if not st.session_state.get("backward_analysis_confirmed", False):
    if st.button("Confirm and run optimization", type="primary"):
        result.confirmed_by_user = True
        st.session_state.backward_analysis_result = result
        st.session_state.backward_analysis_confirmed = True
        st.session_state.phase = "optimize"

        strip_stage_charts(result)
        st.session_state.backward_analysis_result = result

        try:
            with st.status("Running MMM fit and optimizer…", expanded=True) as status:
                optim, budget = _execute_optimization()
                status.update(
                    label="Optimization complete",
                    state="complete",
                    expanded=False,
                )
            st.success(
                f"Optimization complete — **{optim.predicted_conversions:,.0f}** predicted "
                f"conversions on a **${budget:,.0f}** budget (KKT: {optim.kkt_status})."
            )
            if st.button("View Step 3: Allocation", key="goto_allocation_fresh"):
                st.switch_page("pages/3_allocation.py")
        except Exception as exc:
            st.error(
                "Could not complete optimization. Ensure Step 1 finished cleaning "
                f"and saved train/test splits, then try again.\n\n`{exc}`"
            )
else:
    st.success("Backward analysis confirmed.")
    optim = st.session_state.get("optim_result")
    if st.session_state.get("optimization_complete") and optim is not None:
        st.success(
            f"Optimizer result ready — **{optim.predicted_conversions:,.0f}** predicted "
            "conversions. Open Step 3 to view the allocation."
        )
        if st.button("View Step 3: Allocation"):
            st.switch_page("pages/3_allocation.py")
    elif not st.session_state.get("optimization_complete"):
        st.warning("Optimization has not run yet (or failed on the last attempt).")
        if st.button("Run optimization now", type="primary"):
            try:
                with st.status("Running MMM fit and optimizer…", expanded=True) as status:
                    optim, budget = _execute_optimization()
                    status.update(label="Optimization complete", state="complete", expanded=False)
                st.success(
                    f"Optimization complete — **{optim.predicted_conversions:,.0f}** predicted "
                    f"conversions on a **${budget:,.0f}** budget (KKT: {optim.kkt_status})."
                )
                st.rerun()
            except Exception as exc:
                st.error(
                    "Could not complete optimization. Ensure Step 1 finished cleaning "
                    f"and saved train/test splits, then try again.\n\n`{exc}`"
                )
