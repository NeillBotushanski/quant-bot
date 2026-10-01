"""
utils.py — Quant Bot V4.0

Shared helpers for:
• Time handling
• Heartbeat logging
• Async retries for API calls
• Circuit breaker logic
• Simple math helpers (z-score)
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Iterable, Optional

import numpy as np
import pytz

from .config import (
    TIMEZONE,
    MAX_RETRIES,
    RETRY_SLEEP,
    HEARTBEAT_INTERVAL,
)

# ============================================================
# TIME HELPERS
# ============================================================

_TZ = pytz.timezone(TIMEZONE)


def now_eastern() -> datetime:
    """
    Return timezone-aware datetime in the configured market timezone.
    """
    return datetime.now(_TZ)


def format_ts(dt: Optional[datetime] = None) -> str:
    """
    Format a datetime for logging; defaults to 'now' if None.
    """
    if dt is None:
        dt = now_eastern()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat(logger: logging.Logger, label: str = "[HEARTBEAT]") -> None:
    """
    Simple heartbeat logger.

    Called periodically by the bot to confirm it's alive.
    Keeps the V3-style log format for backward compatibility.

    Example log:
        [HEARTBEAT] Bot alive at 2025-11-28 12:01:43
    """
    ts = format_ts()
    logger.info(f"{label} Bot alive at {ts}")


# ============================================================
# ASYNC RETRY WRAPPER
# ============================================================

async def retry_async(
        coro_func: Callable[..., Awaitable[Any]],
        *args,
        retries: int = MAX_RETRIES,
        delay: float = RETRY_SLEEP,
        logger: Optional[logging.Logger] = None,
        **kwargs,
) -> Any:
    """
    Run an async function with simple retry logic.

    Used heavily for Alpaca data calls (bars, snapshots, etc).

    Args:
        coro_func: async callable (e.g. fetch_bars_async)
        *args, **kwargs: passed to coro_func
        retries: max retries (default from config)
        delay: sleep between retries in seconds
        logger: optional logger; if None, uses module logger

    Returns:
        Result of coro_func on success, or None on hard failure.
    """
    log = logger or logging.getLogger(__name__)
    attempt = 0

    while True:
        attempt += 1
        try:
            return await coro_func(*args, **kwargs)
        except Exception as exc:
            if attempt >= retries:
                log.error(
                    f"[retry_async] Giving up after {attempt} attempts "
                    f"for {coro_func.__name__}: {exc}"
                )
                return None

            log.warning(
                f"[retry_async] Error in {coro_func.__name__} (attempt {attempt}/{retries}): "
                f"{exc} — retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)


# ============================================================
# CIRCUIT BREAKER
# ============================================================

def trigger_circuit_breaker(
        account: Any,
        logger: logging.Logger,
        max_dd_pct: float,
) -> bool:
    """
    Simple circuit breaker based on account drawdown.

    Uses account.equity vs account.last_equity as a DD proxy
    (this matches the V3 behavior so existing code still works).

    Args:
        account: Alpaca account object (must have equity, last_equity)
        logger: logger to emit messages
        max_dd_pct: e.g. 0.12 for -12% intraday DD

    Returns:
        True  -> circuit breaker tripped, stop entering new trades
        False -> drawdown is under control
    """
    try:
        equity = float(account.equity)
        last_equity = float(getattr(account, "last_equity", account.equity))
    except Exception as exc:
        logger.error(f"[CIRCUIT BREAKER] Failed to read equity from account: {exc}")
        return False

    if last_equity <= 0:
        # No valid baseline; don't trip breaker
        return False

    dd = (equity - last_equity) / last_equity

    if dd <= -max_dd_pct:
        logger.error(
            f"[CIRCUIT BREAKER] Drawdown {dd * 100:.2f}% "
            f"exceeds limit of {max_dd_pct * 100:.2f}%. "
            f"Stopping new entries."
        )
        return True

    # Optional debug log (commented to avoid noise)
    # logger.debug(
    #     f"[CIRCUIT BREAKER] Drawdown OK: {dd * 100:.2f}% "
    #     f"(limit {max_dd_pct * 100:.2f}%)"
    # )
    return False


# ============================================================
# MATH HELPERS
# ============================================================

def zscore(series: Iterable[float]) -> float:
    """
    Compute z-score for the latest value in a series.

    Used for:
        • Pairs spread z-score
        • Residual z-score on cointegrated pairs
        • Any mean-reversion style logic

    Returns:
        z-score of last element, or 0.0 if not enough data / std=0.
    """
    if series is None:
        return 0.0

    arr = np.asarray(series, dtype=float)
    if arr.size < 5:
        return 0.0

    mean = float(np.mean(arr))
    std = float(np.std(arr))

    if std == 0.0:
        return 0.0

    return (arr[-1] - mean) / std
