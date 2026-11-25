"""Free-tier-aware Gemini calls: walk a chain of models instead of failing on one.

Free-tier quotas are per model (e.g. gemini-3.8-flash allows 20 requests/day), so when one
model's quota is spent the next model usually still has headroom. A daily-quota 429 cannot
be fixed by retrying, so we mark that model exhausted for the rest of the process and move
on; a per-minute 429 gets one short wait first. 404 (model not offered to this key) also
moves on. Transient 5xx are already retried by the SDK's HttpRetryOptions.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

# Models whose daily quota (or availability) is gone for this process; skipped on later calls.
_EXHAUSTED: set[str] = set()

# The longest per-minute retryDelay we are willing to sleep through before moving on.
_MAX_WAIT_S = 20.0


class AllModelsFailedError(RuntimeError):
    """Every model in the chain was exhausted, unavailable or errored."""


def model_chain(primary: str, fallbacks: Sequence[str] | None) -> list[str]:
    """Primary first, then fallbacks, de-duplicated, order preserved."""
    out: list[str] = []
    for m in [primary, *(fallbacks or [])]:
        m = str(m)
        if m and m not in out:
            out.append(m)
    return out


def _retry_delay_s(message: str) -> float | None:
    m = re.search(r"retry(?:Delay)?['\"]?:?\s*(?:in\s*)?['\"]?(\d+(?:\.\d+)?)s", message)
    return float(m.group(1)) if m else None


def generate_with_fallback(
    client: Any,
    models: Sequence[str],
    notes: list[str],
    **kwargs: Any,
) -> tuple[Any, str]:
    """Call client.models.generate_content on each model in turn.

    Returns (response, model_used). Raises AllModelsFailedError when none succeeds.
    """
    from google.genai import errors  # lazy: no import-time side effects

    failures: list[str] = []
    for model in models:
        if model in _EXHAUSTED:
            failures.append(f"{model}: skipped (quota exhausted earlier)")
            continue
        for attempt in range(2):
            try:
                response = client.models.generate_content(model=model, **kwargs)
                if failures:
                    notes.append(f"Gemini fell back to {model} (" + "; ".join(failures) + ")")
                return response, model
            except errors.APIError as exc:
                msg = str(exc)
                code = getattr(exc, "code", None)
                if code == 404:
                    _EXHAUSTED.add(model)
                    failures.append(f"{model}: 404 not available to this key")
                    break
                if code == 429:
                    daily = "per day" in msg.lower() or "perday" in msg.lower()
                    delay = _retry_delay_s(msg)
                    if not daily and attempt == 0 and delay is not None and delay <= _MAX_WAIT_S:
                        time.sleep(delay)  # per-minute limit: one short wait, then retry
                        continue
                    if daily:
                        _EXHAUSTED.add(model)
                    failures.append(f"{model}: 429 {'daily' if daily else 'rate'} quota exhausted")
                    break
                failures.append(f"{model}: {code} {msg[:120]}")
                break  # 5xx already retried by the SDK; try the next model
            except Exception as exc:  # timeouts / transport errors: try the next model
                failures.append(f"{model}: {type(exc).__name__} {str(exc)[:120]}")
                break
    notes.append("Gemini model chain failed: " + "; ".join(failures))
    raise AllModelsFailedError("; ".join(failures))
