"""
filters.py — Quant Bot V4.0

Signal / sanity filters for:
• Single-stock strategies (momentum, reversal, trend)
• Pair candidates (correlation + cointegration stats)
• Spread sanity checks
• Market regime pass/fail

These are "yes/no" gates that sit between raw signals and execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    # Single-stock config
    EMA_FAST,
    EMA_MEDIUM,
    EMA_SLOW,
    EMA_TREND,
    RSI_PERIOD,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    ATR_PERIOD,
    MOMENTUM_MIN_RVOL,
    MOMENTUM_MIN_INTRA_RANGE_PCT,
    MOMENTUM_MIN_TREND_SLOPE,
    REVERSAL_RSI_DEEP_OVERSOLD,
    REVERSAL_RSI_DEEP_OVERBOUGHT,
    REVERSAL_MIN_WICK_RATIO,
    SINGLE_HARD_STOP_PCT,
    SINGLE_TAKE_PROFIT_PCT,
    MAX_SINGLE_HOLD_MIN,
    # Pairs config
    MIN_PAIR_CORR,
    LOOKBACK_CORR_BARS,
    ADF_PVAL_MAX,
    MIN_SPREAD_STD,
    ZSCORE_ENTRY,
    ZSCORE_EXIT,
    ZSCORE_STOP,
    # Logging flags
    LOG_SIGNALS_DEBUG,
    LOG_PAIRS_DEBUG,
)

from .indicators import compute_indicator_pack

logger = logging.getLogger(__name__)


# ============================================================
# HELPER DATA STRUCTURES
# ============================================================

@dataclass
class SingleFilterContext:
    """Cache of indicators + stats for a symbol on a given timeframe."""
    df_1m: pd.DataFrame
    indicators: Dict[str, object]  # output of compute_indicator_pack
    rvol: float
    day_range_pct: float


# ============================================================
# BASIC STAT HELPERS
# ============================================================

def _intraday_stats(df_1m: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute:
        - rvol: simple relative volume vs mean of last N days intraday volume
        - day_range_pct: (high - low) / open * 100

    df_1m is expected to be today's 1m bars.
    """
    if df_1m is None or df_1m.empty:
        return 0.0, 0.0

    vol = df_1m["volume"].astype(float)
    if vol.empty:
        rvol = 0.0
    else:
        recent_mean = vol.rolling(60).mean().iloc[-1]  # last 60 minutes
        recent_mean = float(recent_mean) if not np.isnan(recent_mean) else 0.0
        curr_vol = float(vol.iloc[-60:].sum()) if len(vol) >= 60 else float(vol.sum())
        rvol = curr_vol / (recent_mean * 60 + 1e-9) if recent_mean > 0 else 1.0

    day_high = float(df_1m["high"].max())
    day_low = float(df_1m["low"].min())
    day_open = float(df_1m["open"].iloc[0])

    day_range_pct = 0.0
    if day_open > 0:
        day_range_pct = (day_high - day_low) / day_open * 100.0

    return rvol, day_range_pct


def _build_single_context(df_1m: pd.DataFrame) -> Optional[SingleFilterContext]:
    if df_1m is None or df_1m.empty:
        return None

    indicators = compute_indicator_pack(
        df_1m,
        ema_fast=EMA_FAST,
        ema_med=EMA_MEDIUM,
        ema_slow=EMA_SLOW,
        ema_trend=EMA_TREND,
        rsi_period=RSI_PERIOD,
        atr_period=ATR_PERIOD,
    )

    rvol, day_range_pct = _intraday_stats(df_1m)
    return SingleFilterContext(
        df_1m=df_1m,
        indicators=indicators,
        rvol=rvol,
        day_range_pct=day_range_pct,
    )


# ============================================================
# UNIVERSE / TIER-1 FILTER
# ============================================================

def tier1_pass(price: float, dollar_volume: float) -> bool:
    """
    Basic Tier-1 universe gate.

    Called from bot.build_universe() on each symbol after we already
    did liquidity filtering. To keep things robust and avoid more
    config dependencies, this is intentionally simple:

        - reject if price or dollar_volume missing / non-positive
        - otherwise, pass

    If you want stricter universe rules later, we can wire real
    thresholds here.
    """
    if price is None or dollar_volume is None:
        return False
    if price <= 0 or dollar_volume <= 0:
        return False
    return True


# ============================================================
# SINGLE-STOCK FILTERS
# ============================================================

def _momentum_filter(
        symbol: str,
        ctx: SingleFilterContext,
        log: logging.Logger,
) -> bool:
    """Momentum-style filter: trend + RVOL + range."""
    df = ctx.df_1m
    close = df["close"].astype(float)

    ema_fast = ctx.indicators["ema_fast"]
    ema_med = ctx.indicators["ema_med"]
    ema_slow = ctx.indicators["ema_slow"]
    ema_trend = ctx.indicators["ema_trend"]
    rsi_val = ctx.indicators["rsi"]
    atr_val = ctx.indicators["atr"]

    if len(close) < 50:
        if LOG_SIGNALS_DEBUG:
            log.debug(f"[Single/Mom] {symbol}: too few bars ({len(close)})")
        return False

    last_close = float(close.iloc[-1])
    last_ema_fast = float(ema_fast.iloc[-1])
    last_ema_med = float(ema_med.iloc[-1])
    last_ema_slow = float(ema_slow.iloc[-1])
    last_ema_trend = float(ema_trend.iloc[-1])
    last_rsi = float(rsi_val.iloc[-1])
    last_atr = float(atr_val.iloc[-1])

    # Trend alignment: EMA fast > EMA med > EMA slow > EMA trend for long
    trend_up = (
            last_ema_fast > last_ema_med
            and last_ema_med > last_ema_slow
            and last_ema_slow > last_ema_trend
            and last_close > last_ema_fast
    )

    trend_down = (
            last_ema_fast < last_ema_med
            and last_ema_med < last_ema_slow
            and last_ema_slow < last_ema_trend
            and last_close < last_ema_fast
    )

    # Basic volatility sanity
    if last_atr <= 0 or last_atr / last_close < 0.002:  # <0.2% ATR is too dead
        if LOG_SIGNALS_DEBUG:
            log.debug(f"[Single/Mom] {symbol}: ATR too low ({last_atr / last_close:.4f})")
        return False

    # Intraday range threshold
    if ctx.day_range_pct < MOMENTUM_MIN_INTRA_RANGE_PCT:
        if LOG_SIGNALS_DEBUG:
            log.debug(
                f"[Single/Mom] {symbol}: day range too small "
                f"({ctx.day_range_pct:.2f}% < {MOMENTUM_MIN_INTRA_RANGE_PCT:.2f}%)"
            )
        return False

    # Relative volume filter
    if ctx.rvol < MOMENTUM_MIN_RVOL:
        if LOG_SIGNALS_DEBUG:
            log.debug(
                f"[Single/Mom] {symbol}: rvol too low "
                f"({ctx.rvol:.2f} < {MOMENTUM_MIN_RVOL:.2f})"
            )
        return False

    # RSI confirmation: avoid overbought for long, oversold for short
    long_ok = trend_up and last_rsi < RSI_OVERBOUGHT
    short_ok = trend_down and last_rsi > RSI_OVERSOLD

    if not (long_ok or short_ok):
        if LOG_SIGNALS_DEBUG:
            log.debug(
                f"[Single/Mom] {symbol}: trend/RSI combo failed "
                f"(trend_up={trend_up}, trend_down={trend_down}, RSI={last_rsi:.1f})"
            )
        return False

    if LOG_SIGNALS_DEBUG:
        log.debug(
            f"[Single/Mom] {symbol}: PASS "
            f"trend_up={trend_up}, trend_down={trend_down}, "
            f"rvol={ctx.rvol:.2f}, range={ctx.day_range_pct:.2f}%, RSI={last_rsi:.1f}"
        )
    return True


def _reversal_filter(
        symbol: str,
        ctx: SingleFilterContext,
        log: logging.Logger,
) -> bool:
    """Reversal-style filter: RSI extremes + wick structure."""
    df = ctx.df_1m
    close = df["close"].astype(float)

    rsi_val = ctx.indicators["rsi"]
    wick_ratios = ctx.indicators["wick_ratios"]
    atr_val = ctx.indicators["atr"]

    if len(close) < 20:
        if LOG_SIGNALS_DEBUG:
            log.debug(f"[Single/Rev] {symbol}: too few bars")
        return False

    last_rsi = float(rsi_val.iloc[-1])
    last_atr = float(atr_val.iloc[-1])

    # Candle structure
    last_wr = wick_ratios.iloc[-1]
    upper_ratio = float(last_wr["upper_wick_ratio"])
    lower_ratio = float(last_wr["lower_wick_ratio"])

    # Vol sanity
    last_close = float(close.iloc[-1])
    if last_atr <= 0 or last_atr / last_close < 0.002:
        if LOG_SIGNALS_DEBUG:
            log.debug(f"[Single/Rev] {symbol}: ATR too low")
        return False

    # Oversold reversal (hammer)
    long_candidate = (
            last_rsi <= REVERSAL_RSI_DEEP_OVERSOLD
            and lower_ratio >= REVERSAL_MIN_WICK_RATIO
    )

    # Overbought reversal (shooting star)
    short_candidate = (
            last_rsi >= REVERSAL_RSI_DEEP_OVERBOUGHT
            and upper_ratio >= REVERSAL_MIN_WICK_RATIO
    )

    if not (long_candidate or short_candidate):
        if LOG_SIGNALS_DEBUG:
            log.debug(
                f"[Single/Rev] {symbol}: reversal conditions failed "
                f"(RSI={last_rsi:.1f}, upper={upper_ratio:.2f}, lower={lower_ratio:.2f})"
            )
        return False

    if LOG_SIGNALS_DEBUG:
        log.debug(
            f"[Single/Rev] {symbol}: PASS "
            f"RSI={last_rsi:.1f}, upper={upper_ratio:.2f}, lower={lower_ratio:.2f}"
        )
    return True


def _trend_filter(
        symbol: str,
        ctx: SingleFilterContext,
        log: logging.Logger,
) -> bool:
    """
    Trend-following filter, slightly looser than pure momentum:
    just requires EMA ordering + reasonable volatility.
    """
    df = ctx.df_1m
    close = df["close"].astype(float)

    ema_fast = ctx.indicators["ema_fast"]
    ema_med = ctx.indicators["ema_med"]
    ema_slow = ctx.indicators["ema_slow"]
    ema_trend = ctx.indicators["ema_trend"]
    atr_val = ctx.indicators["atr"]

    if len(close) < 50:
        if LOG_SIGNALS_DEBUG:
            log.debug(f"[Single/Trend] {symbol}: too few bars")
        return False

    last_close = float(close.iloc[-1])
    last_ema_fast = float(ema_fast.iloc[-1])
    last_ema_med = float(ema_med.iloc[-1])
    last_ema_slow = float(ema_slow.iloc[-1])
    last_ema_trend = float(ema_trend.iloc[-1])
    last_atr = float(atr_val.iloc[-1])

    if last_atr <= 0 or last_atr / last_close < 0.0015:
        if LOG_SIGNALS_DEBUG:
            log.debug(f"[Single/Trend] {symbol}: ATR too low")
        return False

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

    if not (up_trend or down_trend):
        if LOG_SIGNALS_DEBUG:
            log.debug(
                f"[Single/Trend] {symbol}: EMA stack not trending "
                f"(fast={last_ema_fast:.2f}, med={last_ema_med:.2f}, "
                f"slow={last_ema_slow:.2f}, trend={last_ema_trend:.2f})"
            )
        return False

    if LOG_SIGNALS_DEBUG:
        log.debug(
            f"[Single/Trend] {symbol}: PASS up_trend={up_trend}, down_trend={down_trend}"
        )
    return True


def single_stock_filter(
        symbol: str,
        df_1m: pd.DataFrame,
        mode: str,
        log: Optional[logging.Logger] = None,
) -> bool:
    """
    High-level filter entry point for single-stock signals.

    Args:
        symbol: ticker symbol
        df_1m: 1-minute bars for ~1–2 hours or today's session
        mode: "momentum", "reversal", or "trend"
    """
    logger_obj = log or logger

    ctx = _build_single_context(df_1m)
    if ctx is None:
        return False

    # Map mode to underlying filter
    mode = mode.lower()
    if mode == "momentum":
        return _momentum_filter(symbol, ctx, logger_obj)
    elif mode == "reversal":
        return _reversal_filter(symbol, ctx, logger_obj)
    elif mode == "trend":
        return _trend_filter(symbol, ctx, logger_obj)
    else:
        logger_obj.warning(f"[Single] Unknown mode '{mode}' for {symbol}")
        return False


# Backwards-compatible alias for V3-style naming
def tier2_single_pass(
        symbol: str,
        df_1m: pd.DataFrame,
        mode: str = "momentum",
        log: Optional[logging.Logger] = None,
) -> bool:
    """
    Alias kept for backward compatibility (V3 code may import this).
    Simply delegates to single_stock_filter().
    """
    return single_stock_filter(symbol, df_1m, mode=mode, log=log)


# ============================================================
# MARKET REGIME FILTER
# ============================================================

def market_regime_pass(
        spy_fast: float,
        spy_slow: float,
        vix: float,
        log: Optional[logging.Logger] = None,
) -> bool:
    """
    Simple market regime gate.

    Passes when:
        • SPY fast EMA > slow EMA  (uptrend)   OR
        • SPY fast EMA < slow EMA  (downtrend) with vix not extreme
    and VIX is not in a 'crisis' zone (> 40).

    You can refine this later, but this keeps the V3 semantics
    while adding a VIX safety cap.
    """
    logger_obj = log or logger

    crisis = vix >= 40.0
    if crisis:
        logger_obj.warning(
            f"[Regime] VIX={vix:.2f} too high → blocking new entries."
        )
        return False

    # We allow trading in both trends, but some strategy logic
    # may prefer up/down internally.
    trending = abs(spy_fast - spy_slow) / max(spy_slow, 1e-9) > 0.00001

    if not trending:
        if LOG_SIGNALS_DEBUG:
            logger_obj.debug(
                f"[Regime] SPY not trending enough "
                f"(fast={spy_fast:.2f}, slow={spy_slow:.2f})"
            )
        return False

    if LOG_SIGNALS_DEBUG:
        logger_obj.debug(
            f"[Regime] PASS fast={spy_fast:.2f}, slow={spy_slow:.2f}, vix={vix:.2f}"
        )
    return True


# ============================================================
# PAIRS FILTERS
# ============================================================

def pair_candidate_pass(
        symbol_a: str,
        symbol_b: str,
        corr: float,
        adf_pvalue: Optional[float],
        spread_std: float,
        log: Optional[logging.Logger] = None,
) -> bool:
    """
    Decide whether (symbol_a, symbol_b) is a good pair candidate.

    Args:
        corr: rolling correlation between log-prices over LOOKBACK_CORR_BARS
        adf_pvalue: p-value from ADF test on the spread (None if not available)
        spread_std: standard deviation of the spread over lookback
    """
    logger_obj = log or logger

    if corr < MIN_PAIR_CORR:
        if LOG_PAIRS_DEBUG:
            logger_obj.debug(
                f"[Pairs] {symbol_a}-{symbol_b}: corr {corr:.3f} < MIN_PAIR_CORR {MIN_PAIR_CORR:.3f}"
            )
        return False

    if adf_pvalue is not None and adf_pvalue > ADF_PVAL_MAX:
        if LOG_PAIRS_DEBUG:
            logger_obj.debug(
                f"[Pairs] {symbol_a}-{symbol_b}: ADF p-value {adf_pvalue:.3f} > {ADF_PVAL_MAX:.3f}"
            )
        return False

    if spread_std < MIN_SPREAD_STD:
        if LOG_PAIRS_DEBUG:
            logger_obj.debug(
                f"[Pairs] {symbol_a}-{symbol_b}: spread std {spread_std:.3f} < MIN_SPREAD_STD {MIN_SPREAD_STD:.3f}"
            )
        return False

    if LOG_PAIRS_DEBUG:
        logger_obj.debug(
            f"[Pairs] {symbol_a}-{symbol_b}: PASS corr={corr:.3f}, "
            f"adf_p={adf_pvalue}, spread_std={spread_std:.3f}"
        )
    return True


def spread_sanity_check(
        spread: pd.Series,
        log: Optional[logging.Logger] = None,
) -> bool:
    """
    Sanity check for the pair spread series.

    Used to avoid acting on broken data or clearly non-stationary
    series before taking signals.
    """
    logger_obj = log or logger

    if spread is None or len(spread) < LOOKBACK_CORR_BARS:
        if LOG_PAIRS_DEBUG:
            logger_obj.debug("[Pairs] Spread sanity: not enough history.")
        return False

    s = spread.astype(float)
    if s.isna().any():
        if LOG_PAIRS_DEBUG:
            logger_obj.debug("[Pairs] Spread sanity: NaNs detected.")
        return False

    std = float(s.std())
    if std == 0.0 or np.isnan(std):
        if LOG_PAIRS_DEBUG:
            logger_obj.debug("[Pairs] Spread sanity: zero or NaN std.")
        return False

    # Optional: check for extreme drift (simple heuristic)
    last = float(s.iloc[-1])
    mean = float(s.mean())
    if abs(last - mean) > 10.0 * std:
        if LOG_PAIRS_DEBUG:
            logger_obj.debug(
                f"[Pairs] Spread sanity: last value {last:.3f} "
                f"more than 10 std from mean {mean:.3f}."
            )
        return False

    return True
