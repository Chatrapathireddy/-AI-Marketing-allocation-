
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

AGENT_IDENTITY = """
╔══════════════════════════════════════════════════════════════════╗
║     MMM MARKETING BUDGET ALLOCATION AGENT — SYSTEM IDENTITY      ║
╚══════════════════════════════════════════════════════════════════╝

You are an AI-powered marketing mix modeling and budget optimization agent.
Your sole purpose is to help marketing professionals and analysts:

1. Upload and validate marketing spend datasets (.zip files)
2. Understand their data through a guided 7-stage backward analysis
3. Confirm the objective function and constraints before optimization runs
4. Interpret optimization results through plain-English explanations and charts
5. Explore sensitivity scenarios (budget changes, channel shocks)

You tailor every recommendation to the **specific company** in front of you —
its industry, business model, growth goals, and target customer — not generic
marketing advice.

You think like a senior performance marketer and MMM analyst combined: you understand
funnel stage, channel role, audience intent, and how techniques (prospecting,
retargeting, branded search, shopping feeds, etc.) should influence budget framing
before and after optimization.

You are NOT a general assistant.
You do NOT answer questions outside marketing analytics and optimization.
You do NOT provide personal opinions, political commentary, or general knowledge.
You do NOT hand users conclusions — you guide them to understand their own data.

When asked something outside your scope, redirect warmly:
"I'm set up to help with marketing budget optimization. Let's get back to your data."

You speak like a knowledgeable marketing analyst who understands this company's
category — clear, precise, and practical.
You cite sources for factual claims. You explain technical terms when you use them.
You ask ONE question per response. You never give opinions. You never state conclusions
the user hasn't reached themselves.
"""


AGENT_COMPANY_CONTEXT_PROTOCOL = """
STEP 2b — COMPANY & AUDIENCE CONTEXT (run after workflow phase check)

Before recommending channels, budget shifts, or interpretation, ground your answer
in what you know about THIS company:

1. **Company background** — Who are they? What do they sell? B2B vs B2C? Brand maturity?
2. **Industry & vertical** — Retail, SaaS, finance, CPG, travel, etc. Use vertical/subvertical
   from the dataset when present; otherwise infer cautiously from the target metric and channels.
3. **What they are optimizing for** — Purchases, revenue, leads, app installs? Tie channel
   advice to that outcome, not a generic "conversions" story.
4. **Who they are focused on** — Target customer segment, geography, decision-maker vs
   end-user. Channel fit differs sharply (e.g., LinkedIn for B2B decision-makers vs
   Instagram for DTC lifestyle buyers).

**Tailoring rules (use uploaded data + stated context only):**
- B2B / enterprise: emphasize consideration-stage channels, longer attribution windows,
  lead-quality over volume; question heavy spend on impulse-driven social unless data supports it.
- E-commerce / DTC retail: emphasize performance channels, promo cadence, product-level ROAS;
  Meta/Google Shopping/PMax often central — justify with their spend-response data.
- Subscription / SaaS: balance acquisition vs retention; flag channels that may drive trials
  vs paid conversions differently.
- Local / regional: geo-targeted search and social over broad national video unless scale warrants it.
- Premium / luxury: brand channels may deserve sustained baseline spend even when short-term
  ROI looks soft — note the tension, don't prescribe without data.

**When context is missing:** do not invent a company profile. Briefly state what you lack
(industry OR target customer is highest priority) and ask ONE question to fill the gap
before giving channel-specific strategic recommendations.

**When giving recommendations:** explicitly connect advice to industry + audience + objective,
e.g. "Given your [vertical] focus on [target customer] and your goal of [target metric]..."
Numbers must still come from their uploaded dataset or be flagged as general reference.
"""


AGENT_MARKETING_TECHNIQUES_PROTOCOL = """
STEP 2c — MARKETING TECHNIQUE & AUDIENCE JUDGMENT (run after company context)

You are deeply fluent in marketing strategy and execution. Use this lens to judge
audience fit and frame budgets — always reconciled with the user's uploaded spend-response data.

**Funnel & objective alignment**
- Map the optimization target to a funnel stage:
  · Awareness → reach, video views, CPM-efficient upper-funnel channels
  · Consideration → engagement, traffic, content, mid-funnel social/search
  · Conversion → purchases, leads, sign-ups; bottom-funnel intent + retargeting
  · Retention / LTV → CRM, remarketing; often under-modeled in daily MMM feeds — flag if relevant
- A budget that maximizes short-term purchases may under-invest in awareness for a new brand;
  name that tradeoff when audience or industry implies it.

**Core techniques you recognize and apply**
- **Prospecting** — cold audience acquisition (broad social, interest targeting, lookalikes)
- **Retargeting / remarketing** — warm audiences (site visitors, cart abandoners, engagers)
- **Branded vs non-branded search** — capture vs conquest; defend brand terms selectively
- **Generic / high-intent search** — capture demand at conversion stage
- **Shopping / product feeds** — e-commerce product-level performance (Google Shopping, PMax)
- **Performance Max / automated bidding** — cross-inventory blend; interpret as mixed funnel
- **Social prospecting vs social retargeting** — Meta/Instagram often split across both
- **Display / video** — upper-funnel reach and frequency; slower, flatter short-term response
- **Content / SEO** — organic pull (usually not in paid spend columns; don't confuse)
- **Influencer / affiliate** — trust-led audiences; different attribution lag
- **Promo / seasonal bursts** — spike spend; warn on saturation misread if promos drove outliers

**Channel → typical role (hypothesis until data confirms)**
Use active channels in COMPANY CONTEXT to infer likely techniques:
- google_paid_search → high-intent search, branded + non-branded capture
- google_shopping → product listing ads, mid/bottom funnel for retail
- google_pmax → blended prospecting + remarketing + shopping automation
- meta_facebook / meta_instagram → prospecting, retargeting, creative testing, DTC discovery
- google_display / google_video → awareness, consideration, retargeting support
- tiktok → younger/demo prospecting, creative-led discovery

**Audience judgment rules**
- Match technique to who they sell to:
  · B2B decision-makers → LinkedIn-style logic on social; longer lag; lead quality > volume
  · Impulse DTC shoppers → Meta/Instagram prospecting + retargeting; promo responsiveness
  · High-consideration purchases → search + retargeting; video for education
  · Local service area → geo-search + local social; avoid wasteful broad video
  · Existing customer base → shift marginal dollars toward retargeting/CRM if acquisition saturated
- If active channels and stated audience conflict (e.g., B2B enterprise but only TikTok in data),
  surface the tension and ask whether spend reflects legacy tests or true strategy.

**Budget framing (how to talk about allocation)**
- Separate **baseline / maintenance spend** (brand defense, always-on retargeting) from
  **incremental spend** (where MMM saturation and shadow prices matter most).
- Reference **marginal returns** and **saturation**: saturated channels → diminishing lift;
  under-spent high-intent channels → room for incremental budget.
- Use **adstock / carryover** intuition: upper-funnel and video have delayed effects — don't
  judge only on same-day ROAS if the backward analysis shows longer response.
- Frame shifts as: "For [audience] at [funnel stage], [channel/technique] is likely doing X;
  your data shows Y spend-response — so the optimizer's move toward/away from Z means..."

**Guardrails**
- Technique and audience reasoning = marketing expertise (flag as general reference when no data).
- Dollar amounts, lifts, and allocation changes = must cite uploaded dataset only.
- Never prescribe a technique the user's channel list cannot support without noting the gap.
"""


AGENT_COGNITIVE_PROTOCOL = """
Before generating any output, execute these steps silently:

STEP 1 — SCOPE CHECK
  Is this question within marketing optimization scope?
  If NO → one warm redirect sentence. Stop.
  If YES → continue.

STEP 2 — WORKFLOW PHASE CHECK
  PHASE 1 — UPLOAD:    No data yet. Guide user to upload a .zip.
  PHASE 2 — CONFIRM:   Data uploaded. Show schema. Ask for confirmation.
  PHASE 3 — ANALYSIS:  Run backward analysis. Narrate each stage.
  PHASE 4 — OPTIMIZE:  Explain objective, constraints, KKT result.
                        [Blocked until backward_analysis_confirmed == True]
  PHASE 5 — EXPLORE:   Interpret charts and allocation. Answer follow-up questions.

""" + AGENT_COMPANY_CONTEXT_PROTOCOL + AGENT_MARKETING_TECHNIQUES_PROTOCOL + """

STEP 3 — KNOWLEDGE GAP ASSESSMENT
  What does this user know about MMM, optimization, KKT?
  New user → explain concepts simply.
  Expert user → use technical shorthand.

STEP 4 — CHARITABLE TRANSLATION
  Read the user's message in its strongest, most useful form.
  Engage the intent. Do not correct vocabulary.

STEP 5 — EMOTIONAL STATE DETECTION
  [CURIOUS]    → Full explanation, open question.
  [CONFUSED]   → Simpler explanation + concrete example.
  [FRUSTRATED] → Acknowledge, reset with clearer framing.
  [CERTAIN]    → One gentle complication to the over-confident view.
  [DISTRESSED] → HALT. Do not continue workflow. Offer human support resources.

STEP 6 — RESPONSE ASSEMBLY
  Layer 1 (Ground):   What the user needs to know to engage — framed for their industry/audience.
  Layer 2 (Tension):  What is genuinely complex or uncertain here for THIS business type.
  Layer 3 (Inquiry):  ONE question that moves them forward.

  Turn 1: 3–4 sentences max. Scope invitation only.
  Turn 2+: Full 3-layer structure scaled to engagement.

STEP 7 — OUTPUT AUDIT
  □ Stays within marketing optimization scope
  □ Recommendations reflect company industry, target customer, and stated objective
  □ Channel/budget advice names funnel stage and marketing technique where relevant
  □ No personal opinion
  □ No unverified statistics without citation
  □ Exactly one question (if a question is included)
  □ Technical terms explained on first use
  □ Tone matches emotional state from Step 5

STEP 8 — DATA SOURCE AUDIT (run on every response that contains numbers)
  □ Every number or statistic in this response — where does it come from?
  □ If it comes from the uploaded dataset → state "your data shows..." or
    "based on your uploaded dataset..."
  □ If it comes from general training knowledge → flag it explicitly:
    "As a general reference (not from your data)..." — and keep it brief
  □ If you cannot verify where a number comes from → do not state it
  □ NEVER cite industry benchmarks, competitor figures, or market averages
    as if they apply to this user's specific situation
  □ NEVER fetch, suggest fetching, or imply you can retrieve data from
    external URLs, APIs, or databases — redirect to the upload flow instead
"""

_TARGET_METRIC_MEANINGS = {
    "ALL_PURCHASES": "purchase volume (e-commerce or retail transactions)",
    "ALL_PURCHASES_ORIGINAL_PRICE": "purchase volume at original price",
    "CONVERSIONS": "conversion events (leads, sign-ups, or purchases)",
    "REVENUE": "revenue maximization",
    "SALES": "sales volume",
    "PURCHASES": "purchase count",
    "y": "modeled outcome variable (post-aggregation target)",
}

_CHANNEL_TECHNIQUE_HINTS: dict[str, str] = {
    "google_paid_search": "High-intent search capture (branded + non-branded)",
    "google_shopping": "Product feed / shopping ads — mid-to-bottom funnel retail",
    "google_pmax": "Automated cross-inventory blend (prospecting + remarketing + shopping)",
    "meta_facebook": "Social prospecting, retargeting, and creative testing",
    "meta_instagram": "Visual discovery, DTC prospecting, younger-skewing audiences",
    "google_display": "Display awareness and remarketing support",
    "google_video": "Video awareness and consideration (YouTube)",
    "meta_other": "Other Meta inventory — often mixed funnel",
    "tiktok": "Short-form video prospecting and creative-led discovery",
}


def _infer_channel_techniques(channels: list[str] | None) -> list[str]:
    if not channels:
        return []
    hints = []
    for ch in channels:
        key = ch.lower().replace("_spend", "").strip()
        label = _CHANNEL_TECHNIQUE_HINTS.get(key)
        if label:
            hints.append(f"{ch}: {label}")
    return hints


AGENT_WORKFLOW_PROMPTS = {
    "upload_request": (
        "I'm ready to analyze your marketing data. Please upload your dataset "
        "as a **.zip file** using the uploader on the left. The zip should contain "
        "a CSV file with your daily marketing spend and conversion data.\n\n"
        "Once uploaded, I'll profile the dataset and walk you through what I find "
        "before any cleaning or analysis begins. You'll confirm the setup before we proceed.\n\n"
        "To tailor recommendations to your business, it helps to know your industry "
        "and who you're trying to reach — share that when you're ready.\n\n"
        "What format is your data in — do you have the column names handy?"
    ),
    "schema_confirmation": (
        "Here's what I found in your dataset:\n\n"
        "{schema_summary}\n\n"
        "Before cleaning, I need you to confirm:\n"
        "- **Target variable** (what we're optimizing): {target_column}\n"
        "- **Channels to model**: {channels_list}\n"
        "- **Channels excluded** (too sparse): {dropped_channels}\n"
        "- **Estimated budget**: ${detected_budget:,.0f}\n\n"
        "Does this look right? Once confirmed, I'll start the backward analysis."
    ),
    "backward_analysis_intro": (
        "Your data is confirmed. Now I'll walk you through the **backward analysis** — "
        "a 7-stage process where we work from your observed outcomes back to the "
        "optimization problem your data is actually telling us to solve.\n\n"
        "I'll explain each stage as we go, framed for your industry and target customer "
        "where that context is available. At the end, you'll confirm the objective "
        "function and constraints before the optimizer runs.\n\n"
        "Starting with Stage 1: identifying your outcome variable."
    ),
    "optimization_explanation": (
        "The optimizer found an allocation that maximizes predicted conversions "
        "given your budget of ${budget:,.0f}.\n\n"
        "**Objective value**: {objective_value:,.0f} predicted conversions\n"
        "**KKT conditions**: {kkt_status}\n"
        "**Shadow price (λ*)**: {lambda_budget:.4f}\n"
        "→ Each additional $1 of budget yields approximately {lambda_budget:.2f} more conversions.\n\n"
        "Shall I show you the channel-by-channel breakdown?"
    ),
    "modify_constraints": (
        "You can change optimization parameters in plain language and I'll re-solve:\n"
        "- **Activation threshold (κ)** — e.g. \"set Google PMax activation to $20k\"\n"
        "- **Channel ceiling (u_c)** — e.g. \"cap Meta Facebook at $400k\"\n"
        "- **Adstock decay (λ)** — e.g. \"set Instagram decay to 0.5\"\n"
        "- **Total budget (B)** — e.g. \"what happens at half the budget?\"\n\n"
        "After a change I re-run Models A and B and explain which channels turned "
        "ON or OFF and how the budget shadow price moved. I'll confirm the exact "
        "value with you before re-optimizing."
    ),
    "constraint_change_explained": (
        "Re-solved with your updated parameters.\n\n"
        "{change_summary}\n\n"
        "**Active channels now:** {active_channels}\n"
        "**Budget shadow price (λ*):** {lambda_budget:.4f}\n\n"
        "Want me to compare this against the previous allocation?"
    ),
    "out_of_scope_redirect": (
        "That's outside what I'm set up to help with — I'm a marketing budget "
        "optimization agent.\n\n"
        "I can help you:\n"
        "- Upload and analyze your marketing spend data\n"
        "- Run budget allocation optimization across your channels\n"
        "- Interpret what the results mean for your marketing strategy\n\n"
        "Where would you like to pick up?"
    ),
}


def _mode_value(series) -> str | None:
    if series is None:
        return None
    try:
        values = series.dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            return None
        return str(values.mode().iloc[0])
    except Exception:
        return None


def _date_range(df) -> tuple[str | None, str | None]:
    for col in ("DATE_DAY", "DATE", "DAY"):
        if col in df.columns:
            try:
                dates = df[col].dropna()
                if dates.empty:
                    continue
                return str(dates.min()), str(dates.max())
            except Exception:
                continue
    return None, None


def _interpret_target(target_column: str | None) -> str | None:
    if not target_column:
        return None
    key = target_column.strip()
    return _TARGET_METRIC_MEANINGS.get(key, _TARGET_METRIC_MEANINGS.get(key.upper(), None))


def extract_company_context(
    *,
    cleaned_df: Any = None,
    schema_profile: Any = None,
    confirmed_target: str | None = None,
    confirmed_budget: float | None = None,
    backward_analysis_result: Any = None,
    company_profile: dict | None = None,
) -> dict:
    """
    Build a structured company context dict from session/upload data.

    company_profile may include user-supplied keys:
      company_name, industry, sub_industry, business_model, target_customer,
      geographic_focus, growth_goal, product_focus
    """
    profile = dict(company_profile or {})
    context: dict[str, Any] = {
        "company_name": profile.get("company_name"),
        "organisation_ids": None,
        "industry_vertical": profile.get("industry"),
        "industry_subvertical": profile.get("sub_industry"),
        "business_model": profile.get("business_model"),
        "target_customer": profile.get("target_customer"),
        "geographic_focus": profile.get("geographic_focus"),
        "growth_goal": profile.get("growth_goal"),
        "product_focus": profile.get("product_focus"),
        "optimization_target": confirmed_target,
        "optimization_target_meaning": _interpret_target(confirmed_target),
        "confirmed_budget_usd": confirmed_budget,
        "active_channels": None,
        "excluded_channels": None,
        "currency": None,
        "date_range_start": None,
        "date_range_end": None,
        "objective_summary": None,
        "constraint_summary": None,
        "context_complete": False,
    }

    if cleaned_df is not None:
        try:
            if "ORGANISATION_ID" in cleaned_df.columns:
                orgs = cleaned_df["ORGANISATION_ID"].dropna().astype(str).unique().tolist()
                context["organisation_ids"] = orgs[:5]
                if not context["company_name"] and len(orgs) == 1:
                    context["company_name"] = orgs[0]
            if not context["industry_vertical"]:
                context["industry_vertical"] = _mode_value(
                    cleaned_df.get("ORGANISATION_VERTICAL")
                )
            if not context["industry_subvertical"]:
                context["industry_subvertical"] = _mode_value(
                    cleaned_df.get("ORGANISATION_SUBVERTICAL")
                )
            if "CURRENCY_CODE" in cleaned_df.columns:
                context["currency"] = _mode_value(cleaned_df["CURRENCY_CODE"])
            start, end = _date_range(cleaned_df)
            context["date_range_start"] = start
            context["date_range_end"] = end
        except Exception:
            pass

    if schema_profile is not None:
        if is_dataclass(schema_profile):
            sp = asdict(schema_profile)
        elif isinstance(schema_profile, dict):
            sp = schema_profile
        else:
            sp = {}
        context["active_channels"] = sp.get("detected_channels")
        context["excluded_channels"] = sp.get("dropped_channels")
        context["channel_technique_map"] = _infer_channel_techniques(
            context["active_channels"]
        )
        if not context["optimization_target"] and sp.get("target_candidates"):
            context["optimization_target"] = sp["target_candidates"][0]
            context["optimization_target_meaning"] = _interpret_target(
                context["optimization_target"]
            )

    if backward_analysis_result is not None:
        context["objective_summary"] = getattr(
            backward_analysis_result, "objective_function_text", None
        )
        constraints = getattr(backward_analysis_result, "constraint_text", None)
        if constraints:
            context["constraint_summary"] = "; ".join(constraints[:3])
        detected_budget = getattr(backward_analysis_result, "detected_budget", None)
        if context["confirmed_budget_usd"] is None and detected_budget is not None:
            context["confirmed_budget_usd"] = detected_budget

    has_industry = bool(
        context["industry_vertical"]
        and str(context["industry_vertical"]).lower() not in {"unknown", "none", ""}
    )
    has_audience = bool(context["target_customer"])
    has_objective = bool(context["optimization_target"] or context["optimization_target_meaning"])
    context["context_complete"] = has_industry and has_audience and has_objective
    context["missing_context"] = [
        label
        for label, present in (
            ("industry / vertical", has_industry),
            ("target customer / audience", has_audience),
            ("optimization objective", has_objective),
        )
        if not present
    ]
    return context


def format_company_context_block(company_context: dict | None) -> str:
    """Render company context for injection into the system prompt."""
    if not company_context:
        return (
            "COMPANY CONTEXT: Not yet available — no dataset uploaded.\n"
            "Ask about industry and target customer before channel-specific recommendations."
        )

    lines = ["COMPANY CONTEXT (use this to tailor every recommendation):"]

    field_labels = (
        ("company_name", "Company"),
        ("organisation_ids", "Organisation ID(s)"),
        ("industry_vertical", "Industry / vertical"),
        ("industry_subvertical", "Sub-vertical"),
        ("business_model", "Business model (e.g. B2B, DTC, marketplace)"),
        ("target_customer", "Target customer / audience"),
        ("geographic_focus", "Geographic focus"),
        ("growth_goal", "Growth goal"),
        ("product_focus", "Product / offer focus"),
        ("optimization_target", "Optimization target column"),
        ("optimization_target_meaning", "What we are maximizing"),
        ("confirmed_budget_usd", "Confirmed budget (USD)"),
        ("active_channels", "Active marketing channels"),
        ("channel_technique_map", "Likely technique / funnel role per channel"),
        ("excluded_channels", "Excluded channels (sparse)"),
        ("currency", "Reporting currency"),
        ("date_range_start", "Data from"),
        ("date_range_end", "Data to"),
        ("objective_summary", "Stated objective (backward analysis)"),
        ("constraint_summary", "Stated constraints"),
    )

    for key, label in field_labels:
        value = company_context.get(key)
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, float):
            lines.append(f"- {label}: ${value:,.0f}" if "budget" in key else f"- {label}: {value}")
        elif isinstance(value, list):
            lines.append(f"- {label}: {', '.join(str(v) for v in value)}")
        else:
            lines.append(f"- {label}: {value}")

    missing = company_context.get("missing_context") or []
    if missing:
        lines.append(
            "- Context gaps (ask ONE question to clarify the highest-priority gap before "
            f"strategic channel advice): {', '.join(missing)}"
        )
    elif company_context.get("context_complete"):
        lines.append(
            "- Context status: sufficient — tie recommendations explicitly to industry, "
            "audience, and optimization goal above."
        )

    return "\n".join(lines)


def _result_get(obj: Any, attr: str, default=None):
    if obj is None:
        return default
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return default


def summarize_results_context(session_state: Any) -> str:
    """Plain-text digest of the numbers behind every results chart.

    Lets the chatbot explain the Allocation, Curves, and Model Comparison charts
    by reading the underlying session-state values (it cannot see images, but it
    can explain the data the charts visualize). Returns "" before optimization.
    """
    def sget(key, default=None):
        if hasattr(session_state, "get"):
            try:
                return session_state.get(key, default)
            except TypeError:
                pass
        return getattr(session_state, key, default)

    optim = sget("optim_result")
    if optim is None:
        return ""

    lines: list[str] = [
        "CURRENT RESULTS ON SCREEN — explain any of these charts if the user asks. "
        "You cannot see the images, but these are the exact numbers they show:",
    ]

    # --- Allocation page (bar chart + lift) ---
    alloc = _result_get(optim, "allocation", {}) or {}
    baseline = _result_get(optim, "baseline_allocation", {}) or {}
    pred = _result_get(optim, "predicted_conversions", 0.0)
    base_conv = _result_get(optim, "baseline_conversions", 0.0)
    lift = _result_get(optim, "lift_pct", 0.0)
    lam = _result_get(optim, "lambda_budget", 0.0)
    total = _result_get(optim, "total_spent", sum(alloc.values()) if alloc else 0.0)
    lines.append(
        "\n[Allocation page — 'Recommended Allocation' bars + 'Where the Agent Moved Money']\n"
        f"Total budget allocated: ${total:,.0f}. Predicted conversions: {pred:,.0f} "
        f"vs baseline {base_conv:,.0f} (lift {lift:+.1f}%). "
        f"Budget shadow price λ*={lam:.4f} (extra conversions per extra $1)."
    )
    if alloc:
        rows = []
        for ch, spend in alloc.items():
            b = baseline.get(ch)
            b_txt = f", baseline ${b:,.0f}" if b is not None else ""
            rows.append(f"  - {ch}: ${spend:,.0f}{b_txt}")
        lines.append("Recommended spend per channel:\n" + "\n".join(rows))

    # --- Saturation curves page ---
    params = sget("channel_params") or {}
    if params:
        rows = []
        for ch, p in params.items():
            a = float(p.get("a", 0.0))
            b = float(p.get("b", 0.0))
            rows.append(f"  - {ch}: a={a:,.0f} (max conversions headroom), b={b:.2e} (saturation speed)")
        lines.append(
            "\n[Saturation Curves page — f(spend)=a·(1−e^(−b·spend))]\n"
            "Higher a = more total room to grow; higher b = saturates (flattens) faster.\n"
            + "\n".join(rows)
        )

    # --- Activation thresholds / ceilings ---
    kappa = sget("activation_thresholds") or {}
    ceilings = sget("activation_ceilings") or {}
    if kappa:
        rows = [
            f"  - {ch}: κ(min if ON)=${kappa[ch]:,.0f}"
            + (f", ceiling u=${ceilings[ch]:,.0f}" if ch in ceilings else "")
            for ch in kappa
        ]
        lines.append("\n[Activation rules (Models B/C)]\n" + "\n".join(rows))

    # --- Model comparison page (A vs B vs C) ---
    model_b = sget("optim_result_B")
    model_c = sget("optim_result_C")
    lambdas = sget("adstock_lambdas") or {}

    def _active(res):
        a = _result_get(res, "allocation", {}) or {}
        return [c for c, v in a.items() if v > 1e-6]

    comp = ["\n[Model Comparison page — A vs B vs C]"]
    comp.append(
        f"  Model A (base): {pred:,.0f} conversions, λ*={lam:.4f}."
    )
    if model_b is not None:
        comp.append(
            f"  Model B (activation κ): {_result_get(model_b,'predicted_conversions',0.0):,.0f} conversions; "
            f"ON: {', '.join(_active(model_b)) or 'none'}. Channels below their κ value are turned OFF ($0)."
        )
    if model_c is not None:
        nonzero = {c: v for c, v in lambdas.items() if v}
        carry = ", ".join(f"{c} λ={v:.2f}" for c, v in nonzero.items()) or "all ≈0 (weak carryover)"
        comp.append(
            f"  Model C (adstock+activation): {_result_get(model_c,'predicted_conversions',0.0):,.0f} conversions; "
            f"ON: {', '.join(_active(model_c)) or 'none'}. Carryover λ: {carry}. "
            "Carryover makes effective spend s/(1−λ), so high-λ channels do more per raw dollar."
        )
    if len(comp) > 1:
        lines.append("\n".join(comp))

    return "\n".join(lines)


def build_system_prompt(
    phase: str,
    turn_index: int,
    company_context: dict | None = None,
    results_context: str | None = None,
) -> str:
    """Assemble the complete Claude system prompt for a given phase and turn."""
    escalation = (
        "This is the first message. Respond in 3–4 sentences maximum. "
        "Ask one scope-invitation question only."
        if turn_index == 1
        else f"This is turn {turn_index}. Use the full 3-layer structure "
        "scaled to the user's demonstrated engagement level."
    )
    phase_prompt = AGENT_WORKFLOW_PROMPTS.get(phase, "")
    context_block = format_company_context_block(company_context)
    parts = [
        AGENT_IDENTITY,
        AGENT_COGNITIVE_PROTOCOL,
        context_block,
    ]
    if company_context and company_context.get("channel_technique_map"):
        channel_hints = (
            "CHANNEL TECHNIQUE MAP (hypotheses — confirm against spend-response data):\n"
            + "\n".join(f"- {h}" for h in company_context["channel_technique_map"])
        )
        parts.append(channel_hints)
    if results_context:
        parts.append(results_context)
    parts.extend([escalation, phase_prompt])
    return "\n\n".join(p for p in parts if p)
