
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_HARM_TRIGGERS = [
    "how to kill",
    "how to hurt",
    "i want to hurt",
    "suicide",
    "how to attack",
    "how to make a weapon",
    "how to make a bomb",
]

# ── External data fetch requests — BLOCK ──────────────────────────────────────
_EXTERNAL_DATA_TRIGGERS = [
    "go to",
    "visit",
    "open the link",
    "fetch from",
    "pull from",
    "scrape",
    "crawl",
    "browse to",
    "http://",
    "https://",
    "www.",
    "call the api",
    "query the database",
    "connect to",
    "use the api",
    "pull from google analytics",
    "pull from facebook ads",
    "pull from hubspot",
    "get data from",
    "download data from",
    "import from",
    "search the web",
    "search online",
    "look it up",
    "find online",
    "look up online",
    "google it",
    "check online",
]

# ── Competitor / industry benchmark requests — REDIRECT ───────────────────────
_EXTERNAL_BENCHMARK_TRIGGERS = [
    "industry average",
    "industry benchmark",
    "competitors spend",
    "competitor roas",
    "market average",
    "sector average",
    "what does [company] spend",
    "how much does [brand] spend",
    "average cpm",
    "average cpc",
    "average roas",
    "typical roas",
    "best in class",
    "benchmark data",
    "compare to industry",
    "how do we compare to",
    "what is the norm",
    "what is standard",
]

_RAW_DATA_PATTERN = re.compile(r"(?m)^(?:[^,\n]+,){3,}[^,\n]+$")
_RAW_DATA_MIN_LINES = 3

_EXTERNAL_KNOWLEDGE_SIGNALS = [
    "industry average is",
    "typically companies spend",
    "on average brands",
    "research shows that",
    "studies suggest",
    "according to gartner",
    "according to forrester",
    "according to nielsen",
    "benchmark report",
    "market research indicates",
    "globally, companies",
]

_OUT_OF_SCOPE_TRIGGERS = [
    "what is the weather",
    "who is the president",
    "write me a poem",
    "help me with my homework",
    "what do you think about politics",
    "tell me a joke",
    "what is the meaning of life",
]

_MARKETING_SCOPE_KEYWORDS = [
    "budget",
    "spend",
    "channel",
    "marketing",
    "campaign",
    "allocation",
    "google",
    "meta",
    "facebook",
    "instagram",
    "conversion",
    "roas",
    "optimize",
    "mmm",
    "roi",
    "cpm",
    "cpc",
    "impression",
    "click",
    "revenue",
    "purchase",
    "saturation",
    "adstock",
    "upload",
    "dataset",
    "analysis",
    "backward",
    "objective",
    "constraint",
]

_PROFANITY_WORDS = ["damn", "hell", "shit", "fuck", "asshole", "bastard"]

_POLITICAL_LEAN_SIGNALS = [
    "you should vote",
    "the right answer is",
    "clearly the correct position",
    "obviously true that",
]

_PII_PATTERNS = [
    (r"\b[\w.-]+@[\w.-]+\.\w+\b", "email"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone"),
    (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "credit_card"),
]

_BLOCK_MESSAGES = {
    "harm_intent": (
        "I cannot help with that. If you are in crisis, please contact "
        "emergency services or a crisis helpline."
    ),
    "profanity": (
        "Please keep the conversation professional. "
        "I'm here to help with marketing budget optimization."
    ),
    "empty_input": "Please type a message so I can help you.",
    "too_long": "Your message is too long. Please shorten it and try again.",
    "external_data_fetch": (
        "I can only work with the dataset you uploaded — I'm not able to fetch "
        "data from external URLs, APIs, or websites. "
        "If you have additional data, please include it in your .zip upload."
    ),
}

_REDIRECT_MESSAGES = {
    "out_of_scope": (
        "I'm a marketing budget optimization agent — that topic is outside my scope. "
        "I can help you analyze your marketing spend, run budget allocation optimization, "
        "and interpret the results. What would you like to explore?"
    ),
    "external_benchmark": (
        "I don't have access to external industry benchmarks or competitor data — "
        "I can only analyze the dataset you uploaded. "
        "What I can do is show you how your channels perform relative to each other "
        "within your own data. Would that be helpful?"
    ),
    "raw_data_in_chat": (
        "It looks like you may have pasted data directly into the chat. "
        "To keep your data secure and ensure it's processed correctly, "
        "please upload it as a .zip or .csv file using the uploader instead. "
        "Pasted data won't be stored or used for analysis."
    ),
}

_OUTPUT_DATA_SOURCE_DISCLAIMER = (
    "\n\n*Note: all figures above are derived from your uploaded dataset only. "
    "This agent does not access external data sources.*"
)

_DATA_ATTRIBUTION_PHRASES = (
    "according to",
    "per ",
    "source:",
    "from your data",
    "your data shows",
    "your data",
    "based on your uploaded dataset",
    "uploaded dataset",
    "based on your data",
)


@dataclass
class GuardrailResult:
    action: str  # "pass" | "sanitize" | "block" | "redirect" | "warn"
    original: str
    sanitized: str
    flags: list[str] = field(default_factory=list)
    block_reason: str | None = None
    redirect_message: str | None = None
    warning_message: str | None = None


class GuardrailsService:
    """Stage 1 (input) + Stage 3 (output) guardrails for the MMM agent."""

    def __init__(self, config_path: str = "config.yaml"):
        self._config = self._load_config(config_path)

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        if not path.exists():
            root = Path(__file__).resolve().parent.parent / "config.yaml"
            path = root if root.exists() else path
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def apply_input_guardrails(self, user_message: str) -> GuardrailResult:
        """
        Rule-based input checks. Order of evaluation:

        1. Length — block if < 3 chars
        2. Harm triggers — block immediately
        3. External data fetch — block (user asking agent to call URLs/APIs)
        4. Raw data in chat — redirect to uploader with warning
        5. PII detection — sanitize ([REDACTED])
        6. Competitor / benchmark requests — redirect with explanation
        7. Out-of-scope detection — redirect
        8. Scope keyword check — if no marketing keyword AND question, redirect

        Return GuardrailResult. The Streamlit app inspects .action to decide
        whether to show the message, the block copy, or the redirect copy.
        """
        msg = (user_message or "").strip()
        flags: list[str] = []
        max_len = self._config.get("guardrails", {}).get("input", {}).get("max_message_length", 2000)

        if len(msg) < 3:
            return GuardrailResult(
                action="block",
                original=user_message,
                sanitized=msg,
                flags=["empty_input"],
                block_reason="empty_input",
            )

        if len(msg) > max_len:
            return GuardrailResult(
                action="block",
                original=user_message,
                sanitized=msg[:max_len],
                flags=["too_long"],
                block_reason="too_long",
            )

        lower = msg.lower()
        for trigger in _HARM_TRIGGERS:
            if trigger in lower:
                return GuardrailResult(
                    action="block",
                    original=user_message,
                    sanitized=msg,
                    flags=["harm_intent"],
                    block_reason="harm_intent",
                )

        if self._config.get("guardrails", {}).get("input", {}).get("block_profanity", True):
            for word in _PROFANITY_WORDS:
                if re.search(rf"\b{re.escape(word)}\b", lower):
                    return GuardrailResult(
                        action="block",
                        original=user_message,
                        sanitized=msg,
                        flags=["profanity"],
                        block_reason="profanity",
                    )

        if self._is_external_fetch_request(msg):
            return GuardrailResult(
                action="block",
                original=user_message,
                sanitized=msg,
                flags=["external_data_fetch"],
                block_reason="external_data_fetch",
            )

        if self._contains_raw_data(msg):
            return GuardrailResult(
                action="redirect",
                original=user_message,
                sanitized=msg,
                flags=["raw_data_in_chat"],
                redirect_message=_REDIRECT_MESSAGES["raw_data_in_chat"],
                warning_message=_REDIRECT_MESSAGES["raw_data_in_chat"],
            )

        sanitized, pii_found = self._redact_pii(msg)
        if pii_found:
            flags.append("pii_redacted")

        if self._is_benchmark_request(sanitized):
            return GuardrailResult(
                action="redirect",
                original=user_message,
                sanitized=sanitized,
                flags=["external_benchmark"] + flags,
                redirect_message=_REDIRECT_MESSAGES["external_benchmark"],
            )

        if self._has_explicit_out_of_scope_trigger(sanitized):
            redirect = (
                self._config.get("guardrails", {})
                .get("input", {})
                .get("out_of_scope_redirect", _REDIRECT_MESSAGES["out_of_scope"])
            )
            if not isinstance(redirect, str):
                redirect = _REDIRECT_MESSAGES["out_of_scope"]
            else:
                redirect = redirect.strip()
            return GuardrailResult(
                action="redirect",
                original=user_message,
                sanitized=sanitized,
                flags=["out_of_scope"] + flags,
                redirect_message=redirect,
            )

        if self._lacks_marketing_scope(sanitized):
            return GuardrailResult(
                action="redirect",
                original=user_message,
                sanitized=sanitized,
                flags=["out_of_scope_keyword"] + flags,
                redirect_message=_REDIRECT_MESSAGES["out_of_scope"],
            )

        return GuardrailResult(
            action="pass" if not pii_found else "sanitize",
            original=user_message,
            sanitized=sanitized,
            flags=flags,
        )

    def _has_explicit_out_of_scope_trigger(self, message: str) -> bool:
        lower = message.lower()
        return any(trigger in lower for trigger in _OUT_OF_SCOPE_TRIGGERS)

    def _lacks_marketing_scope(self, message: str) -> bool:
        lower = message.lower()
        has_marketing = any(kw in lower for kw in _MARKETING_SCOPE_KEYWORDS)
        return not has_marketing and message.strip().endswith("?")

    def _is_out_of_scope(self, message: str) -> bool:
        """Legacy helper — explicit triggers or non-marketing question."""
        return self._has_explicit_out_of_scope_trigger(message) or self._lacks_marketing_scope(
            message
        )

    def _redact_pii(self, text: str) -> tuple[str, bool]:
        found = False
        cleaned = text
        for pattern, _ in _PII_PATTERNS:
            if re.search(pattern, cleaned):
                found = True
                cleaned = re.sub(pattern, "[REDACTED]", cleaned)
        return cleaned, found

    def _is_external_fetch_request(self, message: str) -> bool:
        """True if user asks agent to retrieve data from an external source."""
        lower = message.lower()
        if re.search(r"https?://", lower) or re.search(r"\bwww\.", lower):
            return True
        return any(trigger in lower for trigger in _EXTERNAL_DATA_TRIGGERS)

    def _contains_raw_data(self, message: str) -> bool:
        """True if message appears to contain CSV-like pasted data."""
        matches = _RAW_DATA_PATTERN.findall(message)
        return len(matches) >= _RAW_DATA_MIN_LINES

    def _is_benchmark_request(self, message: str) -> bool:
        """True if user asks for industry benchmarks or competitor spend."""
        lower = message.lower()
        if any(trigger in lower for trigger in _EXTERNAL_BENCHMARK_TRIGGERS):
            return True
        if re.search(r"how much does .+ spend", lower):
            return True
        if re.search(r"what does .+ spend", lower):
            return True
        return False

    def blocked_response_message(self, block_reason: str | None) -> str:
        return _BLOCK_MESSAGES.get(block_reason or "harm_intent", _BLOCK_MESSAGES["harm_intent"])

    def apply_output_guardrails(
        self, response_text: str, data_source: str = "uploaded_dataset"
    ) -> str:
        """
        Rule-based output checks:

        1. Opinion leakage — log warning if political lean signal detected
        2. Question count — log warning if > 1 '?'
        3. Unsourced statistics — flag numbers without citation or data attribution
        4. External knowledge signals — detect phrases implying industry/market data
           the agent does not have; append _OUTPUT_DATA_SOURCE_DISCLAIMER if found
        5. Scope drift — log warning if no marketing keywords in response

        If external knowledge signals detected: append disclaimer automatically.
        Return cleaned response text.
        """
        text = response_text or ""
        _ = data_source

        for signal in _POLITICAL_LEAN_SIGNALS:
            if signal in text.lower():
                logger.warning("Opinion leakage signal detected: %s", signal)

        if self._count_questions(text) > 1:
            logger.warning("Output contains more than one question mark.")

        flagged = self._flag_unsourced_statistics(text)
        if flagged:
            logger.warning("Unsourced statistics flagged: %s", flagged[:3])

        if self._detect_external_knowledge(text):
            logger.warning("External knowledge signal detected in agent output.")
            if _OUTPUT_DATA_SOURCE_DISCLAIMER.strip() not in text:
                text = text + _OUTPUT_DATA_SOURCE_DISCLAIMER

        lower = text.lower()
        if not any(kw in lower for kw in _MARKETING_SCOPE_KEYWORDS):
            logger.warning("Scope drift: response lacks marketing keywords.")

        if self._config.get("guardrails", {}).get("output", {}).get("redact_usernames", True):
            text = re.sub(r"@\w+", "[USER]", text)

        return text

    def _flag_unsourced_statistics(self, text: str) -> list[str]:
        flagged = []
        for match in re.finditer(r"\b\d+(?:\.\d+)?%?\b", text):
            snippet_start = max(0, match.start() - 30)
            snippet_end = min(len(text), match.end() + 50)
            context = text[snippet_start:snippet_end].lower()
            has_citation = re.search(r"\[\d+\]", text[match.start() : match.end() + 30])
            has_attribution = any(phrase in context for phrase in _DATA_ATTRIBUTION_PHRASES)
            if not has_citation and not has_attribution:
                flagged.append(match.group())
        return flagged

    def _detect_external_knowledge(self, text: str) -> bool:
        """True if response cites knowledge outside the uploaded file."""
        lower = text.lower()
        return any(signal in lower for signal in _EXTERNAL_KNOWLEDGE_SIGNALS)

    def _count_questions(self, text: str) -> int:
        """Count question marks in response text."""
        return text.count("?")
