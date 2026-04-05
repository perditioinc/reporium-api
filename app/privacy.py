"""PII redaction helpers for persistent logging (issue #238).

Question text stored in ``query_log`` must not retain common PII such as email
addresses, phone numbers / SSNs, or leaked API keys. These helpers are applied
ONLY to the persistent copy — the original text is still passed to Claude so
answer quality is unaffected.
"""
from __future__ import annotations

import re

# Email addresses — conservative RFC-5322-subset matcher; good enough for
# redaction (false positives here are cheap, false negatives are not).
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
)

# API keys: ``sk-``, ``pk-``, ``ghp_``, ``xoxb-``, plus generic long
# ``[A-Za-z0-9_\-]{20,}`` runs that follow a ``key``/``token``-ish prefix.
_API_KEY_RE = re.compile(
    r"\b(?:sk|pk|rk|xoxb|xoxp|ghp|gho|ghu|ghs|ghr)[-_][A-Za-z0-9_\-]{20,}\b",
    re.IGNORECASE,
)

# Long digit runs — 10+ consecutive digits (phone numbers, SSNs with separators
# stripped, credit-card-ish). We intentionally only match pure digit runs so
# typical numeric values in questions ("top 50 repos") are preserved.
_LONG_DIGITS_RE = re.compile(r"\b\d{10,}\b")


def redact_pii(text: str) -> str:
    """Return ``text`` with emails, long digit runs, and API keys redacted.

    The function is idempotent and safe to call on any string. A ``None`` or
    empty input returns the input unchanged (callers should still feed real
    strings — this is a belt-and-suspenders guard).
    """
    if not text:
        return text
    # Order matters: redact API keys first so their trailing digits don't
    # accidentally trip the long-digit matcher.
    redacted = _API_KEY_RE.sub("[REDACTED_KEY]", text)
    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
    redacted = _LONG_DIGITS_RE.sub("[REDACTED_NUMBER]", redacted)
    return redacted
