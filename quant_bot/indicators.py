"""
indicators.py — Quant Bot V4.0

High-performance indicator utilities:

• EMA (fast, medium, slow, trend)
• ATR (with Wilder smoothing)
• RSI (6-period & fast reversal use-case)
• VWAP (intraday)
• Rolling volatility
• Candle helper metrics (wick/body ratios)
• Safe wrappers to avoid NaNs and crashes

All indicators operate on pandas Series / DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# BASIC SAFETY HELPERS
# ============================================================

def _safe_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        return x.astype(float)
    return pd.Series(x, dtype=float)


def _safe_df(df) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return pd.DataFrame(df)


# ============================================================
# MOVING AVERAGES (EMA)
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Standard exponential moving average.

    V4 uses pandas ewm() which is optimized in C.
    """
    s = _safe_series(series)
    if s.empty:
        return s
    return s.ewm(span=period, adjust=False).mean()


# ============================================================
# ATR (WILDER SMOOTHING)
# ============================================================

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) using classic Wilder smoothing.

    df must contain columns: high, low, close.
    """
    df = _safe_df(df)
    if df.empty:
        return pd.Series(dtype=float)

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing (EMA with alpha = 1/period)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


# ============================================================
# RSI
# ============================================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Used for:
    • Momentum confirmation
    • Reversals (RSI(6), RSI(2))
    """
    s = _safe_series(series)
    if s.empty:
        return s

    delta = s.diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain, index=s.index)
    loss = pd.Series(loss, index=s.index)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))

    return rsi_val


# ============================================================
# VWAP (INTRADAY)
# ============================================================

def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Intraday VWAP (no session resets here — the caller is expected
    to pass only today's bars if they want a clean VWAP).

    VWAP = cumulative(price * volume) / cumulative(volume)
    """
    df = _safe_df(df)
    if df.empty:
        return pd.Series(dtype=float)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].astype(float)

    cum_pv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum() + 1e-9

    return cum_pv / cum_vol


# ============================================================
# ROLLING VOLATILITY
# ============================================================

def rolling_volatility(series: pd.Series, window: int = 30) -> pd.Series:
    """
    Rolling standard deviation of returns.

    Used for:
        • Pair spread stability
        • Trend vs chop detection
        • Risk scaling
    """
    s = _safe_series(series)
    if s.empty:
        return s

    returns = s.pct_change()
    return returns.rolling(window).std()


# ============================================================
# CANDLE STRUCTURE HELPERS (REVERSAL STRATEGY)
# ============================================================

def wick_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute wick/body ratios for reversal setups:
        • upper_wick_ratio
        • lower_wick_ratio
        • body_size

    Used to detect hammer / shooting star candles.
    """
    df = _safe_df(df)
    if df.empty:
        return pd.DataFrame()

    open_ = df["open"].astype(float)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    body = (open_ - close).abs() + 1e-9

    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    return pd.DataFrame(
        {
            "upper_wick_ratio": upper_wick / body,
            "lower_wick_ratio": lower_wick / body,
            "body_size": body,
        },
        index=df.index,
    )


# ============================================================
# COMBINED INDICATOR PACK (EASY MODE)
# ============================================================

def compute_indicator_pack(
        df: pd.DataFrame,
        ema_fast: int,
        ema_med: int,
        ema_slow: int,
        ema_trend: int,
        rsi_period: int,
        atr_period: int,
):
    """
    Return a dict of all key indicators in one call.

    Used by the single-stock engine so we don’t repeatedly
    recalc EMAs, ATR, RSI separately for each filter.
    """
    df = _safe_df(df)
    if df.empty:
        return {}

    close = df["close"].astype(float)

    return {
        "ema_fast": ema(close, ema_fast),
        "ema_med": ema(close, ema_med),
        "ema_slow": ema(close, ema_slow),
        "ema_trend": ema(close, ema_trend),
        "rsi": rsi(close, period=rsi_period),
        "atr": atr(df, period=atr_period),
        "vwap": vwap(df),
        "volatility": rolling_volatility(close),
        "wick_ratios": wick_ratios(df),
    }
