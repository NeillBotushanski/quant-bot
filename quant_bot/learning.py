"""
learning.py — Quant Bot V4.0

Lightweight learning / adaptation layer.

Reads trade_log.csv and:
• Tracks per-symbol performance (wins / losses / winrate)
• Tracks global performance over a rolling window
• Decides whether a symbol should be traded (blacklist + cooldown)
• Computes per-symbol risk factors
• Computes a GLOBAL risk factor based on recent performance

This is NOT a heavy ML system — it's a statistical risk governor.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import pandas as pd

from .config import (
    TRADE_LOG_PATH,
    MIN_TRADES_FOR_DECISION,
    MIN_TRADES_FOR_BLACKLIST,
    GLOBAL_WINDOW,
    SYMBOL_COOLDOWN_DAYS,
    SYMBOL_RISK_BANDS,
    GLOBAL_RISK_BANDS,
    LOG_LEARNING_DEBUG,
)

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SymbolStats:
    symbol: str
    n_trades: int
    wins: int
    losses: int
    winrate: float
    avg_pnl: float
    total_pnl: float
    last_trade_time: Optional[datetime]
    last_outcome: Optional[float]  # last trade PnL


@dataclass
class GlobalStats:
    n_trades: int
    total_pnl: float
    avg_pnl: float
    window_start: Optional[datetime]
    window_end: Optional[datetime]


# ============================================================
# HELPERS: LOADING / PARSING TRADE LOG
# ============================================================

def _load_trade_log() -> pd.DataFrame:
    """
    Load trade_log.csv into a DataFrame.

    Expected minimal columns:
        • timestamp (ISO or %Y-%m-%d %H:%M:%S)
        • symbol
        • pnl  (absolute or relative, we treat it as '1R' style PnL)
    Extra columns (strategy, side, etc.) are ignored here.

    The function is defensive: if file doesn't exist or is empty,
    returns an empty DataFrame and logs a debug message.
    """
    if not os.path.exists(TRADE_LOG_PATH):
        if LOG_LEARNING_DEBUG:
            logger.debug(f"[Learning] Trade log not found at {TRADE_LOG_PATH}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(TRADE_LOG_PATH)
    except Exception as exc:
        logger.error(f"[Learning] Failed to read trade log: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    # Normalize columns if present
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        except Exception:
            df["timestamp"] = pd.NaT
    else:
        # Create dummy timestamps if missing
        df["timestamp"] = pd.NaT

    if "symbol" not in df.columns:
        logger.warning("[Learning] Trade log has no 'symbol' column; learning disabled.")
        return pd.DataFrame()

    if "pnl" not in df.columns:
        # Try common alternative names
        if "pnl_realized" in df.columns:
            df["pnl"] = df["pnl_realized"]
        elif "pl" in df.columns:
            df["pnl"] = df["pl"]
        else:
            logger.warning("[Learning] Trade log has no 'pnl' column; learning disabled.")
            return pd.DataFrame()

    df = df.dropna(subset=["symbol", "pnl"])
    return df


# ============================================================
# SYMBOL-LEVEL STATS
# ============================================================

def _compute_symbol_stats(df: pd.DataFrame, symbol: str) -> Optional[SymbolStats]:
    """
    Compute basic stats for a single symbol from the trade log.
    """
    if df.empty:
        return None

    sdf = df[df["symbol"] == symbol].copy()
    if sdf.empty:
        return None

    n_trades = len(sdf)
    # Define win as pnl > 0, loss as pnl < 0
    wins = int((sdf["pnl"] > 0).sum())
    losses = int((sdf["pnl"] < 0).sum())
    winrate = wins / max(n_trades, 1)

    avg_pnl = float(sdf["pnl"].mean())
    total_pnl = float(sdf["pnl"].sum())

    last_trade_time = None
    last_outcome = None

    if "timestamp" in sdf.columns:
        # pick last non-NaT timestamp if any
        sdf_valid_ts = sdf.dropna(subset=["timestamp"])
        if not sdf_valid_ts.empty:
            last_trade_time = sdf_valid_ts["timestamp"].iloc[-1].to_pydatetime()
            last_outcome = float(sdf_valid_ts["pnl"].iloc[-1])

    return SymbolStats(
        symbol=symbol,
        n_trades=n_trades,
        wins=wins,
        losses=losses,
        winrate=winrate,
        avg_pnl=avg_pnl,
        total_pnl=total_pnl,
        last_trade_time=last_trade_time,
        last_outcome=last_outcome,
    )


def get_symbol_stats(symbol: str) -> Optional[SymbolStats]:
    """
    Public helper to get stats for a symbol from the log.
    """
    df = _load_trade_log()
    return _compute_symbol_stats(df, symbol)


# ============================================================
# GLOBAL STATS
# ============================================================

def _compute_global_stats(df: pd.DataFrame) -> Optional[GlobalStats]:
    """
    Compute global PnL over a rolling window of the last GLOBAL_WINDOW trades.
    """
    if df.empty:
        return None

    # Sort by timestamp where available; else use index
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    else:
        df = df.sort_index()

    # Take last GLOBAL_WINDOW trades
    window_df = df.tail(GLOBAL_WINDOW)
    if window_df.empty:
        return None

    n_trades = len(window_df)
    total_pnl = float(window_df["pnl"].sum())
    avg_pnl = float(window_df["pnl"].mean())

    window_start = None
    window_end = None
    if "timestamp" in window_df.columns and window_df["timestamp"].notna().any():
        window_valid = window_df.dropna(subset=["timestamp"])
        if not window_valid.empty:
            window_start = window_valid["timestamp"].iloc[0].to_pydatetime()
            window_end = window_valid["timestamp"].iloc[-1].to_pydatetime()

    return GlobalStats(
        n_trades=n_trades,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        window_start=window_start,
        window_end=window_end,
    )


def get_global_stats() -> Optional[GlobalStats]:
    df = _load_trade_log()
    return _compute_global_stats(df)


# ============================================================
# RISK FACTORS
# ============================================================

def _map_winrate_to_risk_mult(winrate: float) -> float:
    """
    Use SYMBOL_RISK_BANDS from config to map winrate → risk multiplier.

    SYMBOL_RISK_BANDS is a list of (winrate_threshold, size_multiplier),
    sorted by ascending winrate_threshold, e.g.:

        [
            (0.20, 0.25),
            (0.40, 0.50),
            (0.55, 0.75),
            (0.70, 1.00),
            (0.80, 1.25),
        ]
    """
    bands = sorted(SYMBOL_RISK_BANDS, key=lambda x: x[0])
    chosen = 1.0
    for threshold, mult in bands:
        if winrate >= threshold:
            chosen = mult
    return chosen


def _map_global_pnl_to_risk_mult(total_pnl: float) -> float:
    """
    Use GLOBAL_RISK_BANDS to map cumulative PnL over GLOBAL_WINDOW trades
    → risk multiplier.

    GLOBAL_RISK_BANDS is a list of (pnl_threshold, mult), where
    pnl_threshold is expressed as fraction of something like starting equity
    or 1R units; here we assume your pnl column is in "R" units or %.

    Example:
        [
            (-0.15, 0.5),
            (-0.08, 0.75),
            (0.00, 1.00),
            (0.05, 1.25),
            (0.10, 1.5),
        ]
    """
    bands = sorted(GLOBAL_RISK_BANDS, key=lambda x: x[0])
    chosen = 1.0
    for threshold, mult in bands:
        if total_pnl >= threshold:
            chosen = mult
    return chosen


def get_symbol_risk_factor(symbol: str) -> float:
    """
    Compute a risk multiplier for this symbol based on its stats.

    Returns:
        A number typically between 0.25 and 1.25, used to scale
        base position size.

    If there is not enough data, returns 1.0 (neutral).
    """
    stats = get_symbol_stats(symbol)
    if stats is None:
        return 1.0

    if stats.n_trades < MIN_TRADES_FOR_DECISION:
        # Too few trades to trust stats → neutral
        return 1.0

    mult = _map_winrate_to_risk_mult(stats.winrate)

    if LOG_LEARNING_DEBUG:
        logger.debug(
            f"[Learning] Symbol {symbol}: trades={stats.n_trades}, "
            f"winrate={stats.winrate:.2f}, total_pnl={stats.total_pnl:.2f}, "
            f"risk_mult={mult:.2f}"
        )

    return mult


def get_global_risk_factor() -> float:
    """
    Compute a global risk multiplier based on recent performance.

    Returns:
        A number typically between 0.5 and 1.5, used to scale
        base risk per trade globally.
    """
    gstats = get_global_stats()
    if gstats is None or gstats.n_trades < 5:
        return 1.0

    mult = _map_global_pnl_to_risk_mult(gstats.total_pnl)

    if LOG_LEARNING_DEBUG:
        logger.debug(
            f"[Learning] Global: trades={gstats.n_trades}, "
            f"total_pnl={gstats.total_pnl:.2f}, avg_pnl={gstats.avg_pnl:.2f}, "
            f"risk_mult={mult:.2f}"
        )

    return mult


# ============================================================
# BLACKLIST / COOLDOWN / ELIGIBILITY
# ============================================================

def _symbol_in_cooldown(stats: SymbolStats, now_utc: datetime) -> bool:
    """
    Decide if this symbol is in cooldown.

    Simple rule:
        • If symbol has at least MIN_TRADES_FOR_BLACKLIST trades,
          and the last trade PnL was negative,
          then cooldown for SYMBOL_COOLDOWN_DAYS days.
    """
    if stats.last_trade_time is None or stats.last_outcome is None:
        return False

    if stats.n_trades < MIN_TRADES_FOR_BLACKLIST:
        return False

    # Convert last_trade_time to naive UTC if not already
    ltt = stats.last_trade_time
    if ltt.tzinfo is not None:
        ltt = ltt.astimezone(timezone.utc).replace(tzinfo=None)

    if stats.last_outcome >= 0:
        return False

    cooldown_end = ltt + timedelta(days=SYMBOL_COOLDOWN_DAYS)
    in_cd = now_utc < cooldown_end

    if in_cd and LOG_LEARNING_DEBUG:
        logger.debug(
            f"[Learning] Symbol {stats.symbol} in cooldown until {cooldown_end} "
            f"(last loss={stats.last_outcome:.2f})"
        )

    return in_cd


def should_trade_symbol(symbol: str) -> Tuple[bool, str]:
    """
    Decide whether a NEW trade can be opened for this symbol.

    Returns:
        (allowed: bool, reason: str)

    Rules:
        1. If not enough trades -> allow but neutral risk (we catch that in risk factor).
        2. If enough trades AND symbol in cooldown after a loss -> block.
        3. Otherwise -> allow, with risk scaled via get_symbol_risk_factor().
    """
    df = _load_trade_log()
    if df.empty:
        # No history → allow trading
        return True, "no_history"

    stats = _compute_symbol_stats(df, symbol)
    if stats is None:
        # No trades for this symbol yet
        return True, "no_symbol_history"

    now_utc = datetime.utcnow()

    # Enough trades to consider cooldown/blacklist?
    if stats.n_trades >= MIN_TRADES_FOR_BLACKLIST:
        if _symbol_in_cooldown(stats, now_utc):
            return False, "cooldown"

    # If we get here, symbol is allowed
    if stats.n_trades < MIN_TRADES_FOR_DECISION:
        return True, "insufficient_data"

    # Enough data and not in cooldown → allow
    return True, "ok"


# ============================================================
# BACKWARDS-COMPAT HELPERS
# ============================================================

def should_trade(symbol: str) -> bool:
    """
    Backwards-compatible helper if older code calls learning.should_trade(symbol).
    """
    allowed, _reason = should_trade_symbol(symbol)
    return allowed


def get_risk_multipliers(symbol: str) -> Tuple[float, float, float]:
    """
    Convenience helper:
        Returns (symbol_mult, global_mult, combined_mult)

    Where:
        combined_mult = symbol_mult * global_mult
    """
    sm = get_symbol_risk_factor(symbol)
    gm = get_global_risk_factor()
    return sm, gm, sm * gm
