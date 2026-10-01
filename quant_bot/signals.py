"""
signals.py — Quant Bot V4.0

Signal generation layer for SINGLE-STOCK trades.

Responsibilities:
• Take raw 1m OHLCV data
• Compute indicators (EMA, ATR, RSI, VWAP, etc.)
• Apply filter modes: momentum / reversal / trend
• Decide side: long / short
• Propose entry, stop, target for execution

Pairs trading signal logic is handled mainly in pairs.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import pandas as pd

from .config import (
    EMA_FAST,
    EMA_MEDIUM,
    EMA_SLOW,
    EMA_TREND,
    RSI_PERIOD,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    ATR_PERIOD,
    ATR_MULT_STOP,
    ATR_MULT_TRAIL,
    RR_TARGET,
    SINGLE_HARD_STOP_PCT,
    SINGLE_TAKE_PROFIT_PCT,
    MAX_SINGLE_HOLD_MIN,
    LOG_SIGNALS_DEBUG,
)
from .indicators import compute_indicator_pack
from .filters import single_stock_filter
from .utils import now_eastern

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SingleSignal:
    """
    Represents a single-stock trade signal before position sizing.

    Execution layer (execution.py) will:
        • Determine quantity based on risk model
        • Submit orders (entry + SL + TP)
    """
    symbol: str
    side: str  # "long" or "short"
    mode: str  # "momentum", "reversal", "trend"
    entry: float
    stop: float
    target: float
    atr: float
    rsi: float
    created_at: str  # ISO timestamp
    max_hold_minutes: int = MAX_SINGLE_HOLD_MIN
    extra: Dict[str, float] = None  # any extra metadata (ema values, etc.)

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Ensure extra is always dict, not None
        if d.get("extra") is None:
            d["extra"] = {}
        return d


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _build_indicator_pack(df_1m: pd.DataFrame) -> Dict[str, object]:
    """
    Helper to compute the indicator pack once and reuse.
    """
    return compute_indicator_pack(
        df_1m,
        ema_fast=EMA_FAST,
        ema_med=EMA_MEDIUM,
        ema_slow=EMA_SLOW,
        ema_trend=EMA_TREND,
        rsi_period=RSI_PERIOD,
        atr_period=ATR_PERIOD,
    )


def _compute_side_and_levels_momentum(
        symbol: str,
        df_1m: pd.DataFrame,
        ind: Dict[str, object],
        log: logging.Logger,
) -> Optional[SingleSignal]:
    """
    Decide side + levels for momentum mode.
    """
    close = df_1m["close"].astype(float)

    ema_fast = ind["ema_fast"]
    ema_med = ind["ema_med"]
    ema_slow = ind["ema_slow"]
    ema_trend = ind["ema_trend"]
    rsi_val = ind["rsi"]
    atr_val = ind["atr"]

    if len(close) < 50:
        return None

    last_close = float(close.iloc[-1])
    last_ema_fast = float(ema_fast.iloc[-1])
    last_ema_med = float(ema_med.iloc[-1])
    last_ema_slow = float(ema_slow.iloc[-1])
    last_ema_trend = float(ema_trend.iloc[-1])
    last_rsi = float(rsi_val.iloc[-1])
    last_atr = float(atr_val.iloc[-1])

    # Trend
    up_trend = (
            last_ema_fast > last_ema_med
            and last_ema_med > last_ema_slow
            and last_ema_slow > last_ema_trend
            and last_close > last_ema_fast
    )
    down_trend = (
            last_ema_fast < last_ema_med
            and last_ema_med < last_ema_slow
            and last_ema_slow < last_ema_trend
            and last_close < last_ema_fast
    )

    side: Optional[str] = None

    # For momentum, we want to buy in uptrends that aren't too overbought,
    # and short in downtrends that aren't too oversold.
    if up_trend and last_rsi < RSI_OVERBOUGHT:
        side = "long"
    elif down_trend and last_rsi > RSI_OVERSOLD:
        side = "short"

    if side is None:
        if LOG_SIGNALS_DEBUG:
            log.debug(
                f"[Signal/Mom] {symbol}: no side decision "
                f"(up_trend={up_trend}, down_trend={down_trend}, RSI={last_rsi:.1f})"
            )
        return None

    # Base ATR stop
    if last_atr <= 0:
        return None

    if side == "long":
        stop = last_close - ATR_MULT_STOP * last_atr
        # RR_TARGET defines profit distance in multiples of (entry - stop)
        risk_per_share = max(last_close - stop, 1e-4)
        target = last_close + RR_TARGET * risk_per_share
    else:  # short
        stop = last_close + ATR_MULT_STOP * last_atr
        risk_per_share = max(stop - last_close, 1e-4)
        target = last_close - RR_TARGET * risk_per_share

    # Clamp stop/target to hard pct limits as a safety net
    max_stop_move = SINGLE_HARD_STOP_PCT * last_close
    max_tp_move = SINGLE_TAKE_PROFIT_PCT * last_close

    if side == "long":
        stop = max(stop, last_close - max_stop_move)
        target = min(target, last_close + max_tp_move)
    else:
        stop = min(stop, last_close + max_stop_move)
        target = max(target, last_close - max_tp_move)

    if LOG_SIGNALS_DEBUG:
        log.debug(
            f"[Signal/Mom] {symbol}: side={side}, entry={last_close:.2f}, "
            f"stop={stop:.2f}, target={target:.2f}, ATR={last_atr:.3f}, RSI={last_rsi:.1f}"
        )

    extra = {
        "ema_fast": last_ema_fast,
        "ema_med": last_ema_med,
        "ema_slow": last_ema_slow,
        "ema_trend": last_ema_trend,
    }

    return SingleSignal(
        symbol=symbol,
        side=side,
        mode="momentum",
        entry=last_close,
        stop=stop,
        target=target,
        atr=last_atr,
        rsi=last_rsi,
        created_at=now_eastern().isoformat(),
        extra=extra,
    )


def _compute_side_and_levels_reversal(
        symbol: str,
        df_1m: pd.DataFrame,
        ind: Dict[str, object],
        log: logging.Logger,
) -> Optional[SingleSignal]:
    """
    Decide side + levels for reversal mode.
    """
    close = df_1m["close"].astype(float)
    if len(close) < 20:
        return None

    rsi_val = ind["rsi"]
    atr_val = ind["atr"]
    wick_ratios = ind["wick_ratios"]

    last_close = float(close.iloc[-1])
    last_rsi = float(rsi_val.iloc[-1])
    last_atr = float(atr_val.iloc[-1])

    if last_atr <= 0:
        return None

    last_wr = wick_ratios.iloc[-1]
    upper_ratio = float(last_wr["upper_wick_ratio"])
    lower_ratio = float(last_wr["lower_wick_ratio"])

    from .config import (
        REVERSAL_RSI_DEEP_OVERSOLD,
        REVERSAL_RSI_DEEP_OVERBOUGHT,
        REVERSAL_MIN_WICK_RATIO,
    )

    side: Optional[str] = None

    # Deep oversold + big lower wick -> long reversal candidate
    if last_rsi <= REVERSAL_RSI_DEEP_OVERSOLD and lower_ratio >= REVERSAL_MIN_WICK_RATIO:
        side = "long"

    # Deep overbought + big upper wick -> short reversal candidate
    elif last_rsi >= REVERSAL_RSI_DEEP_OVERBOUGHT and upper_ratio >= REVERSAL_MIN_WICK_RATIO:
        side = "short"

    if side is None:
        if LOG_SIGNALS_DEBUG:
            log.debug(
                f"[Signal/Rev] {symbol}: no side decision "
                f"(RSI={last_rsi:.1f}, upper={upper_ratio:.2f}, lower={lower_ratio:.2f})"
            )
        return None

    # For reversals, we want tighter stops and nearer targets (quick snaps)
    if side == "long":
        stop = last_close - ATR_MULT_STOP * last_atr
        risk_per_share = max(last_close - stop, 1e-4)
        target = last_close + RR_TARGET * risk_per_share
    else:  # short
        stop = last_close + ATR_MULT_STOP * last_atr
        risk_per_share = max(stop - last_close, 1e-4)
        target = last_close - RR_TARGET * risk_per_share

    # Clamp to global hard pct stops/targets
    max_stop_move = SINGLE_HARD_STOP_PCT * last_close
    max_tp_move = SINGLE_TAKE_PROFIT_PCT * last_close

    if side == "long":
        stop = max(stop, last_close - max_stop_move)
        target = min(target, last_close + max_tp_move)
    else:
        stop = min(stop, last_close + max_stop_move)
        target = max(target, last_close - max_tp_move)

    if LOG_SIGNALS_DEBUG:
        log.debug(
            f"[Signal/Rev] {symbol}: side={side}, entry={last_close:.2f}, "
            f"stop={stop:.2f}, target={target:.2f}, ATR={last_atr:.3f}, RSI={last_rsi:.1f}, "
            f"upper={upper_ratio:.2f}, lower={lower_ratio:.2f}"
        )

    extra = {
        "upper_wick_ratio": upper_ratio,
        "lower_wick_ratio": lower_ratio,
    }

    return SingleSignal(
        symbol=symbol,
        side=side,
        mode="reversal",
        entry=last_close,
        stop=stop,
        target=target,
        atr=last_atr,
        rsi=last_rsi,
        created_at=now_eastern().isoformat(),
        extra=extra,
    )


def _compute_side_and_levels_trend(
        symbol: str,
        df_1m: pd.DataFrame,
        ind: Dict[str, object],
        log: logging.Logger,
) -> Optional[SingleSignal]:
    """
    Decide side + levels for trend mode (looser, swing-y).
    """
    close = df_1m["close"].astype(float)

    ema_fast = ind["ema_fast"]
    ema_med = ind["ema_med"]
    ema_slow = ind["ema_slow"]
    ema_trend = ind["ema_trend"]
    rsi_val = ind["rsi"]
    atr_val = ind["atr"]

    if len(close) < 50:
        return None

    last_close = float(close.iloc[-1])
    last_ema_fast = float(ema_fast.iloc[-1])
    last_ema_med = float(ema_med.iloc[-1])
    last_ema_slow = float(ema_slow.iloc[-1])
    last_ema_trend = float(ema_trend.iloc[-1])
    last_rsi = float(rsi_val.iloc[-1])
    last_atr = float(atr_val.iloc[-1])

    if last_atr <= 0:
        return None

    up_trend = (
            last_ema_fast > last_ema_med
            and last_ema_med > last_ema_slow
            and last_ema_slow > last_ema_trend
    )
    down_trend = (
            last_ema_fast < last_ema_med
            and last_ema_med < last_ema_slow
            and last_ema_slow < last_ema_trend
    )

    side: Optional[str] = None

    # For trend mode, we don't care as much about RSI extremes,
    # just avoid absolute crazy overbought/oversold.
    if up_trend and last_rsi < 90:
        side = "long"
    elif down_trend and last_rsi > 10:
        side = "short"

    if side is None:
        if LOG_SIGNALS_DEBUG:
            log.debug(
                f"[Signal/Trend] {symbol}: no side decision "
                f"(up_trend={up_trend}, down_trend={down_trend}, RSI={last_rsi:.1f})"
            )
        return None

    # Trend trades can afford slightly wider trailing stops
    if side == "long":
        stop = last_close - ATR_MULT_STOP * last_atr
        risk_per_share = max(last_close - stop, 1e-4)
        target = last_close + RR_TARGET * risk_per_share
    else:
        stop = last_close + ATR_MULT_STOP * last_atr
        risk_per_share = max(stop - last_close, 1e-4)
        target = last_close - RR_TARGET * risk_per_share

    # Hard caps
    max_stop_move = SINGLE_HARD_STOP_PCT * last_close
    max_tp_move = SINGLE_TAKE_PROFIT_PCT * last_close

    if side == "long":
        stop = max(stop, last_close - max_stop_move)
        target = min(target, last_close + max_tp_move)
    else:
        stop = min(stop, last_close + max_stop_move)
        target = max(target, last_close - max_tp_move)

    if LOG_SIGNALS_DEBUG:
        log.debug(
            f"[Signal/Trend] {symbol}: side={side}, entry={last_close:.2f}, "
            f"stop={stop:.2f}, target={target:.2f}, ATR={last_atr:.3f}, RSI={last_rsi:.1f}"
        )

    extra = {
        "ema_fast": last_ema_fast,
        "ema_med": last_ema_med,
        "ema_slow": last_ema_slow,
        "ema_trend": last_ema_trend,
    }

    return SingleSignal(
        symbol=symbol,
        side=side,
        mode="trend",
        entry=last_close,
        stop=stop,
        target=target,
        atr=last_atr,
        rsi=last_rsi,
        created_at=now_eastern().isoformat(),
        extra=extra,
    )


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def generate_single_signal(
        symbol: str,
        df_1m: pd.DataFrame,
        mode: str = "momentum",
        log: Optional[logging.Logger] = None,
) -> Optional[SingleSignal]:
    """
    Main entry point for single-stock signals.

    Steps:
        1. Compute indicator pack once.
        2. Use filter layer (filters.single_stock_filter) to check if
           symbol qualifies for the given `mode`.
        3. If filtered in, compute side + levels using a mode-specific
           helper.
    """
    logger_obj = log or logger

    if df_1m is None or df_1m.empty:
        return None

    mode = mode.lower().strip()
    if mode not in {"momentum", "reversal", "trend"}:
        logger_obj.warning(f"[Signal] Unknown mode '{mode}' for {symbol}")
        return None

    # Check high-level filter first
    passed = single_stock_filter(symbol, df_1m, mode=mode, log=logger_obj)
    if not passed:
        return None

    # Indicators
    ind = _build_indicator_pack(df_1m)

    # Mode-specific level + side logic
    if mode == "momentum":
        sig = _compute_side_and_levels_momentum(symbol, df_1m, ind, logger_obj)
    elif mode == "reversal":
        sig = _compute_side_and_levels_reversal(symbol, df_1m, ind, logger_obj)
    else:  # "trend"
        sig = _compute_side_and_levels_trend(symbol, df_1m, ind, logger_obj)

    if sig is None and LOG_SIGNALS_DEBUG:
        logger_obj.debug(f"[Signal] {symbol} passed filter but no side/levels chosen (mode={mode})")

    return sig


# Backwards-compatible alias for V3-style usage if any:
def generate_single_signals_for_universe(
        universe_bars_1m: Dict[str, pd.DataFrame],
        mode: str = "momentum",
        log: Optional[logging.Logger] = None,
) -> Dict[str, SingleSignal]:
    """
    Convenience function to generate signals for a dict of {symbol: df_1m}.

    Returns:
        dict {symbol: SingleSignal}
    """
    logger_obj = log or logger
    signals: Dict[str, SingleSignal] = {}

    for sym, df_1m in universe_bars_1m.items():
        sig = generate_single_signal(sym, df_1m, mode=mode, log=logger_obj)
        if sig is not None:
            signals[sym] = sig

    return signals
