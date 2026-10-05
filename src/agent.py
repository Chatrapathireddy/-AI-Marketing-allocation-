"""LLM agent orchestration for the marketing budget workflow."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "upload": ["upload", "zip", "csv", "file", "dataset"],
    "confirm": ["confirm", "schema", "column", "looks right", "target variable"],
    "analyze": ["backward", "analysis", "stage", "outcome", "saturation", "adstock"],
    "optimize": ["optimize", "optimization", "allocation", "allocate", "budget", "kkt", "shadow price", "solver"],
    "explore": ["chart", "sensitivity", "scenario", "breakdown", "curve", "tornado"],
    "modify_constraints": [
        "set ", "change", "raise", "lower", "increase", "decrease", "update",
        "threshold", "activation", "minimum", "min spend", "kappa", "κ",
        "decay", "lambda", "λ", "carryover", "ceiling", "cap", "re-optimize",
        "re-solve", "rerun", "re-run",
    ],
}

_FALLBACK_NO_KEY = (
    "No LLM provider is configured yet. Add one of the following to your project `.env` file:\n\n"
    "- `OLLAMA_MODEL=llama3.1:8b` and optionally `OLLAMA_BASE_URL=http://localhost:11434`\n"
    "- `ANTHROPIC_API_KEY=sk-ant-...`\n"
    "- `HF_MODEL=...` and `HF_API_TOKEN=...`\n\n"
    "Then restart Streamlit (`streamlit run app/app.py`).\n\n"
    "I'm your marketing budget optimization guide. Upload a `.zip` or `.csv` "
    "with daily channel spend and conversions to begin."
)

_OPTIMIZATION_BLOCKED = (
    "Optimization runs after you complete the 7-stage backward analysis and "
    "confirm the objective and constraints.\n\n"
    "Would you like to continue with the backward analysis first?"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_config(config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path else _repo_root() / "config.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_ENV_KEY_NAMES = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "ANTHROPIC_KEY",
    "API_Key",
    "OLLAMA_MODEL",
    "OLLAMA_BASE_URL",
    "HF_MODEL",
    "HF_API_TOKEN",
    "HF_BASE_URL",
    "LLM_PROVIDER",
)


def _load_env_files() -> list[Path]:
    """Load .env from common locations; return paths that were found."""
    root = _repo_root()
    candidates = [
        root / ".env",
        root / "app" / ".env",
        Path.cwd() / ".env",
    ]
    loaded: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        load_dotenv(path, override=False)
        loaded.append(path)
    return loaded


def _api_key() -> str | None:
    _load_env_files()
    for name in _ENV_KEY_NAMES:
        value = (os.getenv(name) or "").strip()
        if value and value not in {"your_key_here", "changeme", "xxx"}:
            return value
    return None


def _resolved_provider() -> str:
    config = _load_config()
    llm = config.get("llm", {})
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit:
        return explicit

    if any(os.getenv(name) for name in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_KEY", "API_Key")):
        return "anthropic"
    if any(os.getenv(name) for name in ("OLLAMA_MODEL", "OLLAMA_BASE_URL")):
        return "ollama"
    if any(os.getenv(name) for name in ("HF_MODEL", "HF_API_TOKEN", "HUGGINGFACE_API_TOKEN", "HF_BASE_URL")):
        return "huggingface"

    provider = str(llm.get("provider") or "ollama").strip().lower()
    return provider


def _ollama_available() -> bool:
    try:
        base_url = _resolved_base_url("ollama").rstrip("/")
        response = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _resolved_model(model_name: str | None = None) -> str:
    config = _load_config()
    llm = config.get("llm", {})
    provider = _resolved_provider()
    env_name = {
        "ollama": "OLLAMA_MODEL",
        "huggingface": "HF_MODEL",
        "anthropic": "ANTHROPIC_MODEL",
        "claude": "ANTHROPIC_MODEL",
    }.get(provider, "ANTHROPIC_MODEL")
    default_model = {
        "ollama": llm.get("model", "llama3.1:8b"),
        "huggingface": llm.get("huggingface_model") or llm.get("model", "microsoft/Phi-3-mini-4k-instruct"),
        "anthropic": llm.get("model", "claude-sonnet-4-20250514"),
        "claude": llm.get("model", "claude-sonnet-4-20250514"),
    }.get(provider, llm.get("model", "llama3.1:8b"))
    return model_name or os.getenv(env_name) or default_model


def _resolved_base_url(provider: str | None = None) -> str:
    provider = (provider or _resolved_provider()).lower()
    if provider == "ollama":
        return os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    if provider == "huggingface":
        return os.getenv("HF_BASE_URL") or "https://api-inference.huggingface.co/models"
    return ""


def get_claude_client():
    """Return configured Anthropic client, or None if no API key."""
    api_key = _api_key()
    if not api_key:
        return None

    import anthropic

    return anthropic.Anthropic(api_key=api_key)


def _extract_response_text(response) -> str:
    parts = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts).strip()


def call_claude(
    messages: list[dict],
    system_prompt: str | None = None,
    model_name: str | None = None,
) -> str | None:
    """Send a Claude messages request; return text or None when unavailable."""
    client = get_claude_client()
    if client is None:
        return None

    config = _load_config().get("llm", {})
    response = client.messages.create(
        model=_resolved_model(model_name),
        max_tokens=config.get("max_tokens", 1024),
        temperature=config.get("temperature", 0.2),
        system=system_prompt or "",
        messages=messages,
    )
    text = _extract_response_text(response)
    return text or None


def _parse_ollama_response(payload: dict) -> str | None:
    message = (payload or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip() or None
    return None


def _call_ollama(messages: list[dict], system_prompt: str | None = None, model_name: str | None = None) -> str | None:
    model = _resolved_model(model_name)
    url = f"{_resolved_base_url('ollama')}/api/chat"
    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            *messages,
        ],
        "stream": False,
    }
    response = httpx.post(url, json=request_payload, timeout=120.0)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama request failed with status {response.status_code}: {response.text}")
    payload = response.json()
    return _parse_ollama_response(payload)


def _call_huggingface(messages: list[dict], system_prompt: str | None = None, model_name: str | None = None) -> str | None:
    model = _resolved_model(model_name)
    token = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError("HF_API_TOKEN is not configured for the Hugging Face provider.")

    prompt = "\n\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages if msg.get("content")
    )
    if system_prompt:
        prompt = f"system: {system_prompt}\n\n{prompt}"

    url = f"{_resolved_base_url('huggingface')}/{model}"
    response = httpx.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.2}},
        timeout=120.0,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face request failed with status {response.status_code}: {response.text}")

    payload = response.json()
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        generated = payload[0].get("generated_text") or payload[0].get("answer")
        if isinstance(generated, str):
            return generated.strip() or None
    if isinstance(payload, dict):
        generated = payload.get("generated_text") or payload.get("answer")
        if isinstance(generated, str):
            return generated.strip() or None
    return None


def call_llm(
    messages: list[dict],
    system_prompt: str | None = None,
    model_name: str | None = None,
) -> str | None:
    """Call the configured LLM provider, including local Ollama and Hugging Face endpoints."""
    provider = _resolved_provider()
    if provider in {"anthropic", "claude"}:
        return call_claude(messages, system_prompt=system_prompt, model_name=model_name)
    if provider == "ollama":
        return _call_ollama(messages, system_prompt=system_prompt, model_name=model_name)
    if provider == "huggingface":
        return _call_huggingface(messages, system_prompt=system_prompt, model_name=model_name)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def parse_problem(user_message: str, context: dict | None = None) -> dict:
    """Parse user intent into a structured problem description."""
    context = context or {}
    lower = (user_message or "").strip().lower()

    intents = [
        intent
        for intent, keywords in _INTENT_KEYWORDS.items()
        if any(keyword in lower for keyword in keywords)
    ]
    primary_intent = intents[0] if intents else "general"

    optimization_blocked = not context.get("backward_analysis_confirmed", False)
    blocked_reason = None
    if primary_intent == "optimize" and optimization_blocked:
        blocked_reason = "backward_analysis_not_confirmed"

    return {
        "primary_intent": primary_intent,
        "intents": intents,
        "phase": context.get("phase", "upload_request"),
        "user_message": user_message,
        "requires_data": primary_intent in {"confirm", "analyze", "optimize", "explore"},
        "optimization_blocked": optimization_blocked,
        "blocked_reason": blocked_reason,
        "upload_complete": context.get("upload_complete", False),
        "schema_confirmed": context.get("schema_confirmed", False),
    }


# --------------------------------------------------------------------------- #
# Runtime parameter changes (stakeholder mod v3 — agent owns κ/λ/B/u_c override)
#
# Flow: detect → validate → confirm → apply (session state + params_dirty).
# Extraction is HYBRID: deterministic regex first, Claude JSON fallback only
# when regex finds nothing. The LLM never bypasses validation.
# Re-solve itself runs in the app layer (optimization_pipeline.maybe_resolve).
# --------------------------------------------------------------------------- #

_KIND_KEYWORDS: dict[str, tuple[str, ...]] = {
    "lambda": ("decay", "lambda", "λ", "carryover", "adstock decay"),
    "ceiling": ("ceiling", "cap", "max spend", "maximum spend", "upper bound"),
    "budget": ("total budget", "overall budget", "budget to", "budget of", "budget at", "budget="),
    "kappa": ("activation", "threshold", "minimum", "min spend", "kappa", "κ", "floor"),
}

_CONFIRM_WORDS = ("yes", "yep", "yeah", "proceed", "confirm", "confirmed", "go ahead", "do it", "ok", "okay", "sure")
_CANCEL_WORDS = ("no", "cancel", "stop", "nevermind", "never mind", "don't", "do not")

# A parameter change must be an explicit instruction, never a question.
_ACTION_VERBS = (
    "set", "change", "update", "raise", "lower", "increase", "decrease",
    "reduce", "cap", "make", "bump", "adjust", "boost",
)
_QUESTION_STARTERS = (
    "how", "why", "what", "when", "where", "which", "who", "does", "do",
    "did", "can", "could", "is", "are", "was", "were", "if", "should",
    "would", "will", "explain", "tell",
)

# Polite imperatives ("can you set...", "could you lower...", "would you raise...")
# read like questions but are really commands. Detect them so they aren't blocked.
_POLITE_REQUEST = re.compile(r"^\s*(please\s+)?(can|could|would|will)\s+(you|we)\b", re.IGNORECASE)
# ...unless the polite request is actually asking for an explanation.
_EXPLAIN_CUES = (
    "explain", "how do", "how does", "how did", "what is", "what are",
    "why", "tell me", "walk me", "describe",
)

# Sentinel channel meaning "apply to every modeled channel".
ALL_CHANNELS = "__all__"
# Cues that a κ/u_c/λ change targets all channels at once.
_BULK_CUES = (
    "all channel", "every channel", "each channel", "all of the channel",
    "all of them", "across the board", "channel threshold", "channel ceiling",
    "channel cap", "channel decay", "channel lambda", "channels",
)


@dataclass
class ParamChange:
    """A single validated-or-pending runtime parameter change."""

    kind: str  # "kappa" | "lambda" | "budget" | "ceiling"
    value: float
    channel: str | None = None
    raw: str = ""


def _modeled_channels(config: dict | None = None) -> list[str]:
    cfg = config or _load_config()
    return list((cfg.get("channels") or {}).get("modeled") or [])


def _channel_aliases(channels: list[str]) -> dict[str, str]:
    """Map spoken forms ('google pmax', 'facebook') → canonical channel key."""
    aliases: dict[str, str] = {}
    for ch in channels:
        aliases[ch.lower()] = ch
        aliases[ch.lower().replace("_", " ")] = ch
        parts = ch.split("_")
        # last token (e.g. 'pmax', 'facebook', 'instagram', 'shopping')
        aliases.setdefault(parts[-1].lower(), ch)
        # platform + last ('google pmax', 'meta facebook')
        if len(parts) >= 2:
            aliases.setdefault(" ".join(parts[-2:]).lower(), ch)
    # common shorthands
    extra = {
        "paid search": "google_paid_search",
        "search": "google_paid_search",
        "fb": "meta_facebook",
        "ig": "meta_instagram",
        "insta": "meta_instagram",
    }
    for k, v in extra.items():
        if v in channels:
            aliases.setdefault(k, v)
    return aliases


def _match_channel(message: str, channels: list[str]) -> str | None:
    lower = message.lower()
    aliases = _channel_aliases(channels)
    # prefer longest alias to avoid 'search' matching before 'google paid search'
    for alias in sorted(aliases, key=len, reverse=True):
        if alias and alias in lower:
            return aliases[alias]
    return None


def _parse_amount(message: str) -> float | None:
    """Extract a dollar/number amount; supports $, commas, k/m suffixes."""
    m = re.search(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([kKmM])?", message)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    suffix = (m.group(2) or "").lower()
    if suffix == "k":
        num *= 1_000
    elif suffix == "m":
        num *= 1_000_000
    return num


def _detect_kind(lower: str, has_channel: bool) -> str | None:
    for kind in ("lambda", "ceiling", "budget", "kappa"):
        if any(kw in lower for kw in _KIND_KEYWORDS[kind]):
            if kind in {"kappa", "ceiling", "lambda"} and not has_channel:
                continue
            return kind
    return None


def parse_parameter_change(message: str, config: dict | None = None) -> ParamChange | None:
    """Deterministic regex extraction of a runtime parameter change.

    Returns None when no clear single change is found (caller may fall back to LLM).
    """
    text = (message or "").strip()
    if not text:
        return None
    lower = text.lower()
    channels = _modeled_channels(config)
    channel = _match_channel(text, channels)
    bulk = channel is None and any(cue in lower for cue in _BULK_CUES)
    kind = _detect_kind(lower, has_channel=channel is not None or bulk)
    if kind is None:
        return None
    target = channel if channel is not None else (ALL_CHANNELS if bulk else None)

    if kind == "lambda":
        m = re.search(r"(0?\.[0-9]+|[01](?:\.0+)?)", lower)
        if not m:
            return None
        return ParamChange(kind="lambda", value=float(m.group(1)), channel=target, raw=text)

    amount = _parse_amount(text)
    if amount is None:
        return None
    return ParamChange(kind=kind, value=amount, channel=target if kind != "budget" else None, raw=text)


def _llm_parse_parameter_change(message: str, config: dict | None = None) -> ParamChange | None:
    """Claude JSON fallback when regex finds nothing. Output still validated by caller."""
    if get_claude_client() is None:
        return None
    channels = _modeled_channels(config)
    system = (
        "Extract a SINGLE marketing optimization parameter change from the user message. "
        "Respond with ONLY compact JSON: {\"kind\": one of [kappa, lambda, budget, ceiling], "
        "\"channel\": one of " + json.dumps(channels) + " or null, \"value\": number}. "
        "kappa/ceiling are USD amounts; lambda is a decay in [0,1]; budget is total USD with channel null. "
        "If there is no clear parameter change, respond with {\"kind\": null}."
    )
    try:
        raw = call_claude([{"role": "user", "content": message}], system_prompt=system)
        if not raw:
            return None
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM parameter parse failed: %s", exc)
        return None
    kind = data.get("kind")
    if kind not in {"kappa", "lambda", "budget", "ceiling"}:
        return None
    try:
        value = float(data.get("value"))
    except (TypeError, ValueError):
        return None
    channel = data.get("channel")
    if channel is not None and channel not in channels:
        channel = None
    return ParamChange(kind=kind, value=value, channel=channel, raw=message)


def _looks_like_command(message: str) -> bool:
    """True only for explicit instructions, not questions.

    Guards against misreading questions like "if the ceiling is 375K, how did
    the optimizer allocate 1.4M?" as a ceiling-change command.
    """
    text = (message or "").strip().lower()
    if not text:
        return False
    has_action = re.search(r"\b(" + "|".join(_ACTION_VERBS) + r")\b", text) is not None
    if not has_action:
        return False
    # "can you change κ to 20k" is a command; "can you explain κ" is not.
    if _POLITE_REQUEST.match(text):
        return not any(cue in text for cue in _EXPLAIN_CUES)
    first = text.split()[0].strip(".,!").lower()
    if first in _QUESTION_STARTERS:
        return False
    return True


def detect_parameter_change(
    message: str,
    config: dict | None = None,
    *,
    use_llm_fallback: bool = True,
) -> ParamChange | None:
    """Hybrid extraction: regex first, then Claude fallback.

    Only runs when the message is an explicit instruction (has an action verb
    and is not phrased as a question) — questions fall through to normal chat.
    """
    if not _looks_like_command(message):
        return None
    change = parse_parameter_change(message, config)
    if change is not None:
        return change
    if use_llm_fallback:
        return _llm_parse_parameter_change(message, config)
    return None


def _session_get(session_state, key: str, default=None):
    if session_state is None:
        return default
    if hasattr(session_state, "get"):
        try:
            return session_state.get(key, default)
        except TypeError:
            pass
    return getattr(session_state, key, default)


def _session_set(session_state, key: str, value) -> None:
    try:
        session_state[key] = value  # mapping-style (st.session_state, dict)
    except TypeError:
        setattr(session_state, key, value)


def _ceiling_for(channel: str, session_state, config: dict) -> float | None:
    ceilings = _session_get(session_state, "activation_ceilings", None)
    if not ceilings:
        ceilings = (config.get("activation") or {}).get("ceilings") or {}
    val = ceilings.get(channel) if isinstance(ceilings, dict) else None
    return float(val) if val is not None else None


def _kappa_for(channel: str, session_state, config: dict) -> float | None:
    thresholds = _session_get(session_state, "activation_thresholds", None)
    if not thresholds:
        thresholds = (config.get("activation") or {}).get("thresholds") or {}
    val = thresholds.get(channel) if isinstance(thresholds, dict) else None
    return float(val) if val is not None else None


def validate_parameter_change(
    change: ParamChange,
    session_state=None,
    config: dict | None = None,
) -> tuple[bool, str]:
    """Check feasibility. Returns (ok, message). Message is a clarification on failure."""
    cfg = config or _load_config()
    channels = _modeled_channels(cfg)

    if change.kind in {"kappa", "lambda", "ceiling"} and not change.channel:
        return False, (
            "Which channel? I can adjust: " + ", ".join(f"`{c}`" for c in channels)
            + ". Or say \"all channels\" to apply it to every channel."
        )
    if change.channel and change.channel != ALL_CHANNELS and change.channel not in channels:
        return False, (
            f"I don't recognize `{change.channel}`. Valid channels are: "
            + ", ".join(f"`{c}`" for c in channels) + "."
        )

    if change.channel == ALL_CHANNELS:
        if change.kind == "lambda":
            if not (0.0 <= change.value <= 1.0):
                return False, f"Decay λ must be between 0 and 1; you gave {change.value:g}."
            return True, "ok"
        if change.value <= 0:
            return False, f"That value must be positive; you gave ${change.value:,.0f}."
        if change.kind == "kappa":
            bad = [c for c in channels
                   if (uc := _ceiling_for(c, session_state, cfg)) is not None and change.value >= uc]
            if bad:
                return False, (
                    f"${change.value:,.0f} is at or above the ceiling for "
                    + ", ".join(f"`{c}`" for c in bad)
                    + ", which would force those channels permanently OFF. "
                    "Pick a threshold below their ceilings, or adjust one channel at a time."
                )
            return True, "ok"
        if change.kind == "ceiling":
            bad = [c for c in channels
                   if (k := _kappa_for(c, session_state, cfg)) is not None and change.value <= k]
            if bad:
                return False, (
                    f"${change.value:,.0f} is at or below the activation threshold for "
                    + ", ".join(f"`{c}`" for c in bad)
                    + ". A ceiling must exceed each channel's κ."
                )
            return True, "ok"

    if change.kind == "lambda":
        if not (0.0 <= change.value <= 1.0):
            return False, f"Decay λ must be between 0 and 1; you gave {change.value:g}."
        return True, "ok"

    if change.kind == "budget":
        if change.value <= 0:
            return False, f"Budget must be positive; you gave {change.value:g}."
        return True, "ok"

    if change.kind == "kappa":
        if change.value <= 0:
            return False, f"Activation threshold must be positive; you gave ${change.value:,.0f}."
        uc = _ceiling_for(change.channel, session_state, cfg)
        if uc is not None and change.value >= uc:
            return False, (
                f"Activation threshold for `{change.channel}` must be below its ceiling "
                f"u_c = ${uc:,.0f}; you gave ${change.value:,.0f}."
            )
        return True, "ok"

    if change.kind == "ceiling":
        if change.value <= 0:
            return False, f"Ceiling must be positive; you gave ${change.value:,.0f}."
        kappa = _kappa_for(change.channel, session_state, cfg)
        if kappa is not None and change.value <= kappa:
            return False, (
                f"Ceiling for `{change.channel}` must exceed its activation threshold "
                f"κ = ${kappa:,.0f}; you gave ${change.value:,.0f}."
            )
        return True, "ok"

    return False, "I couldn't interpret that parameter change."


def format_change_confirmation(change: ParamChange) -> str:
    """Confirmation echo shown before mutating (demo-safety)."""
    if change.kind == "lambda":
        where = "all channels" if change.channel == ALL_CHANNELS else f"`{change.channel}`"
        return (
            f"Set adstock decay λ for {where} to **{change.value:g}** and re-optimize? "
            "Reply **yes** to proceed."
        )
    if change.kind == "budget":
        return (
            f"Set the total weekly budget to **${change.value:,.0f}** and re-optimize? "
            "Reply **yes** to proceed."
        )
    label = "activation threshold κ" if change.kind == "kappa" else "ceiling u_c"
    where = "all channels" if change.channel == ALL_CHANNELS else f"`{change.channel}`"
    return (
        f"Set the {label} for {where} to **${change.value:,.0f}** and re-optimize? "
        "Reply **yes** to proceed."
    )


def _apply_targets(change: ParamChange, existing: dict) -> list[str]:
    """Channels this change writes to (all modeled channels for the bulk sentinel)."""
    if change.channel == ALL_CHANNELS:
        chans = _modeled_channels()
        return chans or list(existing.keys())
    return [change.channel]


def apply_parameter_change(session_state, change: ParamChange) -> None:
    """Mutate session state with the change and flag a pending re-solve."""
    if change.kind == "kappa":
        thresholds = dict(_session_get(session_state, "activation_thresholds", {}) or {})
        for ch in _apply_targets(change, thresholds):
            thresholds[ch] = float(change.value)
        _session_set(session_state, "activation_thresholds", thresholds)
    elif change.kind == "ceiling":
        ceilings = dict(_session_get(session_state, "activation_ceilings", {}) or {})
        for ch in _apply_targets(change, ceilings):
            ceilings[ch] = float(change.value)
        _session_set(session_state, "activation_ceilings", ceilings)
    elif change.kind == "lambda":
        lambdas = dict(_session_get(session_state, "adstock_lambdas", {}) or {})
        for ch in _apply_targets(change, lambdas):
            lambdas[ch] = float(change.value)
        _session_set(session_state, "adstock_lambdas", lambdas)
    elif change.kind == "budget":
        _session_set(session_state, "confirmed_budget", float(change.value))
    _session_set(session_state, "params_dirty", True)


def is_confirmation(message: str) -> bool:
    lower = (message or "").strip().lower()
    return any(lower == w or lower.startswith(w + " ") or lower == w + "." for w in _CONFIRM_WORDS)


def is_cancellation(message: str) -> bool:
    lower = (message or "").strip().lower()
    return any(lower == w or lower.startswith(w + " ") for w in _CANCEL_WORDS)


def process_parameter_message(
    session_state,
    message: str,
    config: dict | None = None,
    *,
    use_llm_fallback: bool = True,
) -> dict:
    """Coordinator the app calls before run_agent.

    Returns {"handled": bool, "response": str|None, "needs_resolve": bool}.
    Re-solve itself is performed by the app via optimization_pipeline.maybe_resolve.
    """
    cfg = config or _load_config()
    pending = _session_get(session_state, "pending_param_change", None)

    # 1) Resolve a previously-confirmed pending change.
    if pending is not None:
        if is_confirmation(message):
            change = pending if isinstance(pending, ParamChange) else ParamChange(**pending)
            apply_parameter_change(session_state, change)
            _session_set(session_state, "pending_param_change", None)
            return {
                "handled": True,
                "response": None,  # app will summarize after re-solve
                "needs_resolve": True,
                "change": change,
            }
        if is_cancellation(message):
            _session_set(session_state, "pending_param_change", None)
            return {
                "handled": True,
                "response": "Okay, I left the parameters unchanged.",
                "needs_resolve": False,
            }
        # Neither confirm nor cancel — fall through to normal handling.

    # 2) Detect a new parameter change.
    change = detect_parameter_change(message, cfg, use_llm_fallback=use_llm_fallback)
    if change is None:
        return {"handled": False, "response": None, "needs_resolve": False}

    ok, msg = validate_parameter_change(change, session_state, cfg)
    if not ok:
        return {"handled": True, "response": msg, "needs_resolve": False}

    _session_set(session_state, "pending_param_change", change)
    return {
        "handled": True,
        "response": format_change_confirmation(change),
        "needs_resolve": False,
    }


def _normalize_history(conversation_history: list[dict], user_message: str) -> list[dict]:
    """Drop a trailing duplicate user turn if the app already appended it."""
    history = list(conversation_history or [])
    if history and history[-1].get("role") == "user":
        last_content = (history[-1].get("content") or "").strip()
        if last_content == (user_message or "").strip():
            history = history[:-1]
    return history


def _format_history_for_claude(history: list[dict]) -> list[dict]:
    formatted = []
    for msg in history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content or role not in {"user", "assistant"}:
            continue
        formatted.append({"role": role, "content": content})
    return formatted


def _fallback_response(user_message: str, context: dict | None = None) -> str:
    parsed = parse_problem(user_message, context)
    if parsed.get("blocked_reason") == "backward_analysis_not_confirmed":
        return _OPTIMIZATION_BLOCKED
    if parsed["primary_intent"] == "upload":
        return (
            "Please upload your marketing dataset as a `.zip` or `.csv` using the "
            "uploader on the Upload page.\n\n"
            "What format is your data in today?"
        )
    return _FALLBACK_NO_KEY


def run_agent(
    user_message: str,
    system_prompt: str,
    conversation_history: list[dict],
    context: dict | None = None,
) -> str:
    """Run one agent turn through the configured LLM provider; fall back gracefully when unavailable."""
    msg = (user_message or "").strip()
    if not msg:
        return "Please type a message so I can help you."

    parsed = parse_problem(msg, context)
    if parsed.get("blocked_reason") == "backward_analysis_not_confirmed":
        return _OPTIMIZATION_BLOCKED

    provider = _resolved_provider()
    if provider == "anthropic" and get_claude_client() is None:
        return _fallback_response(msg, context)
    if provider == "ollama" and not _ollama_available():
        return _fallback_response(msg, context)
    if provider == "huggingface" and not (os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")):
        return _fallback_response(msg, context)

    history = _normalize_history(conversation_history, msg)
    llm_messages = _format_history_for_claude(history)
    llm_messages.append({"role": "user", "content": msg})

    try:
        text = call_llm(llm_messages, system_prompt=system_prompt)
        if text:
            return text
        return (
            "The configured model returned an empty response. Check the provider URL, model name, "
            "and authentication, then try again."
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully for UI stability
        logger.warning("LLM call failed for provider %s: %s", provider, exc)
        if provider == "anthropic":
            return (
                "I couldn't reach Claude right now. "
                f"Error: {exc}\n\n"
                "Check that `ANTHROPIC_API_KEY` is valid in `.env` and restart the app."
            )
        if provider == "ollama":
            return (
                "I couldn't reach the local Ollama server. "
                f"Error: {exc}\n\n"
                "Start Ollama and confirm `OLLAMA_MODEL` is installed, or update `OLLAMA_BASE_URL`."
            )
        return (
            "I couldn't reach the configured Hugging Face model. "
            f"Error: {exc}\n\n"
            "Check `HF_MODEL`, `HF_API_TOKEN`, and the model endpoint."
        )


def run_benchmark(
    messages: list[str],
    system_prompt: str | None = None,
    context: dict | None = None,
) -> dict:
    """Benchmark agent responses for a list of in-scope messages."""
    from src.agent_prompts import build_system_prompt

    prompt = system_prompt or build_system_prompt("upload_request", 1)
    results: list[dict] = []
    latencies: list[float] = []

    for message in messages:
        started = time.perf_counter()
        try:
            response = run_agent(message, prompt, [], context=context)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            results.append(
                {
                    "message": message,
                    "response": response,
                    "latency_s": round(elapsed, 4),
                    "ok": True,
                    "response_chars": len(response),
                }
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            results.append(
                {
                    "message": message,
                    "error": str(exc),
                    "latency_s": round(elapsed, 4),
                    "ok": False,
                }
            )

    success_count = sum(1 for row in results if row["ok"])
    avg_latency = round(sum(latencies) / len(latencies), 4) if latencies else 0.0

    return {
        "n_messages": len(messages),
        "n_success": success_count,
        "n_failed": len(messages) - success_count,
        "avg_latency_s": avg_latency,
        "results": results,
    }


def diagnose_claude_setup() -> dict:
    """Return non-secret diagnostics for troubleshooting the configured model provider."""
    loaded_paths = [str(p) for p in _load_env_files()]
    api_key = _api_key()
    provider = _resolved_provider()
    root = _repo_root()
    return {
        "repo_root": str(root),
        "env_files_loaded": loaded_paths,
        "env_file_at_repo_root": (root / ".env").is_file(),
        "api_key_configured": bool(api_key),
        "provider": provider,
        "provider_ready": provider == "ollama" or bool(api_key) or bool(os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")),
        "model": _resolved_model(),
        "base_url": _resolved_base_url(provider),
        "accepted_env_vars": list(_ENV_KEY_NAMES),
    }
