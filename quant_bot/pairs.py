"""
pairs.py — Quant Bot V4.0

Pairs trading engine (beta-neutral, mean-reversion style).

Responsibilities:
• Discover statistically valid pairs using correlation on returns
• Compute hedge ratio (β) for each pair
• Build spread series and z-score
• Generate entry instructions (direction, beta, prices) for execution layer

This file is designed to be compatible with:
    from .pairs import discover_pair_candidates, generate_pair_signal, build_spread_series
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as cfg
from .filters import pair_candidate_pass, spread_sanity_check
from .utils import zscore, now_eastern

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# CONFIG WITH SAFE DEFAULTS (works with V3 or V4 config.py)
# ------------------------------------------------------------

# Correlation threshold for candidate pairs
MIN_CORR = getattr(cfg, "MIN_PAIR_CORR", getattr(cfg, "MIN_CORR", 0.80))

# Minimum bars to construct spread
LOOKBACK_BARS = getattr(cfg, "LOOKBACK_CORR_BARS", 80)

# Minimum spread volatility to avoid ultra-flat spreads
MIN_SPREAD_STD = getattr(cfg, "MIN_SPREAD_STD", 0.002)

# Z-score thresholds
ZSCORE_ENTRY = getattr(cfg, "ZSCORE_ENTRY", 1.5)
ZSCORE_EXIT = getattr(cfg, "ZSCORE_EXIT", 0.5)
ZSCORE_STOP = getattr(cfg, "ZSCORE_STOP", 4.0)

# Time stop in minutes for pairs
PAIR_TIME_STOP_MIN = getattr(cfg, "PAIR_TIME_STOP_MIN", 60)

# Debug toggle
LOG_PAIRS_DEBUG = getattr(cfg, "LOG_PAIRS_DEBUG", False)


# ============================================================
# CANDIDATE DISCOVERY (FROM CORR MATRIX)
# ============================================================

def discover_pair_candidates(
        returns_matrix,
        symbol_list: List[str],
        corr_matrix,
        max_pairs: int = 20,
) -> List[Dict]:
    """
    Select candidate pairs from a correlation matrix.

    Parameters
    ----------
    returns_matrix : np.ndarray or pd.DataFrame
        Not used directly here (left for future extensions), but kept
        in the signature for backwards compatibility with V3.
    symbol_list : list[str]
        List of symbols corresponding to matrix indices.
    corr_matrix : np.ndarray or pd.DataFrame
        Symmetric correlation matrix.
    max_pairs : int
        Max number of candidate pairs to return (highest correlation first).

    Returns
    -------
    list of dicts with keys:
        'sym_a', 'sym_b', 'corr'
    """
    # Normalize corr_matrix to numpy
    corr_arr = np.asarray(corr_matrix)
    n = len(symbol_list)
    pairs: List[Dict] = []

    if corr_arr.shape[0] != n or corr_arr.shape[1] != n:
        logger.warning("[Pairs] corr_matrix shape does not match symbol_list length.")
        return []

    for i in range(n):
        for j in range(i + 1, n):
            sym_a = symbol_list[i]
            sym_b = symbol_list[j]
            corr_val = corr_arr[i, j]

            if not pair_candidate_pass(
                    sym_a,
                    sym_b,
                    corr_val,
                    adf_pvalue=None,
                    spread_std=MIN_SPREAD_STD,
            ):
                continue

            pairs.append(
                {
                    "sym_a": sym_a,
                    "sym_b": sym_b,
                    "corr": float(corr_val),
                }
            )

    # Sort by correlation descending and clip to max_pairs
    pairs.sort(key=lambda x: x["corr"], reverse=True)
    if max_pairs is not None and max_pairs > 0:
        pairs = pairs[:max_pairs]

    logger.info("[Pairs] %d candidate pairs above corr %.2f", len(pairs), MIN_CORR)
    return pairs


# ============================================================
# SPREAD & HEDGE RATIO
# ============================================================

def _estimate_beta(log_a: np.ndarray, log_b: np.ndarray) -> float:
    """
    Estimate hedge ratio β via simple OLS:
        log(A) = alpha + β * log(B) + ε
    """
    if log_a.size != log_b.size or log_a.size < 10:
        return 1.0

    x = log_b.reshape(-1, 1)
    y = log_a.reshape(-1, 1)
    # Add constant term for alpha
    X = np.hstack([x, np.ones_like(x)])

    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        beta = float(coef[0, 0])
        if not np.isfinite(beta) or beta == 0:
            beta = 1.0
        return beta
    except Exception as e:
        logger.warning(f"[Pairs] Beta estimation failed: {e}")
        return 1.0


def build_spread_series(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
) -> Tuple[Optional[pd.Series], Optional[float], Optional[pd.Series], Optional[pd.Series]]:
    """
    Construct spread = log(A) - β * log(B), aligned on time index.

    Returns
    -------
    (spread_series, beta, closes_a_aligned, closes_b_aligned)

    If anything fails, returns (None, None, None, None).
    """
    if df_a is None or df_b is None:
        return None, None, None, None

    if "close" not in df_a.columns or "close" not in df_b.columns:
        return None, None, None, None

    # Align on time index (inner join)
    joined = df_a[["close"]].rename(columns={"close": "close_a"}).join(
        df_b[["close"]].rename(columns={"close": "close_b"}),
        how="inner",
    ).dropna()

    if len(joined) < LOOKBACK_BARS:
        return None, None, None, None

    closes_a = joined["close_a"].astype(float)
    closes_b = joined["close_b"].astype(float)

    # Log prices
    log_a = np.log(closes_a.values)
    log_b = np.log(closes_b.values)

    beta = _estimate_beta(log_a, log_b)

    spread_vals = log_a - beta * log_b
    spread_series = pd.Series(spread_vals, index=joined.index)

    return spread_series, beta, closes_a, closes_b


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_pair_signal(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        sym_a: str,
        sym_b: str,
) -> Optional[Dict]:
    """
    Generate a pair trade signal for (sym_a, sym_b) from two price DataFrames.

    Intended flow:
        1. build_spread_series → spread, β
        2. z-score on spread (using entire window)
        3. If |z| >= ZSCORE_ENTRY and spread sane → trade
        4. Direction:
            • z > 0  → spread rich → SHORT_SPREAD (short A, long B)
            • z < 0  → spread cheap → LONG_SPREAD  (long A, short B)

    Returns
    -------
    signal dict or None

    signal keys (used by execution & analytics):
        sym_a, sym_b, direction, zscore, beta, price_a, price_b,
        scale_in, z_exit, z_stop, open_time, time_stop_min,
        spread_vol, predictive_score
    """
    spread_series, beta, closes_a, closes_b = build_spread_series(df_a, df_b)
    if spread_series is None:
        return None

    # Basic sanity check on spread (length, std, finite)
    if not spread_sanity_check(spread_series):
        return None

    vol = float(np.std(np.asarray(spread_series)))
    if vol < MIN_SPREAD_STD:
        # Too flat, unlikely to mean-revert with enough edge
        if LOG_PAIRS_DEBUG:
            logger.debug(
                "[Pairs] %s-%s rejected: spread std %.6f < MIN_SPREAD_STD %.6f",
                sym_a, sym_b, vol, MIN_SPREAD_STD,
            )
        return None

    # z-score over full window
    z = float(zscore(spread_series))
    if abs(z) < ZSCORE_ENTRY:
        # Not far enough from mean
        if LOG_PAIRS_DEBUG:
            logger.debug(
                "[Pairs] %s-%s |z|=%.2f < entry=%.2f → no trade",
                sym_a, sym_b, abs(z), ZSCORE_ENTRY,
            )
        return None

    # Latest prices (aligned with spread index)
    try:
        price_a = float(closes_a.iloc[-1])
        price_b = float(closes_b.iloc[-1])
    except Exception:
        return None

    # Direction: spread = log(A) - β log(B)
    # If z > 0, spread is rich → short spread (short A, long B)
    # If z < 0, spread is cheap → long spread (long A, short B)
    if z > 0:
        direction = "short_spread"
    else:
        direction = "long_spread"

    # Optional scale-in flag: bigger z → more conviction
    scale_in = abs(z) >= (2.0 * ZSCORE_ENTRY)

    # Predictive score placeholder (could be extended later)
    predictive_score = abs(z) * vol

    signal = {
        "sym_a": sym_a,
        "sym_b": sym_b,
        "direction": direction,
        "zscore": float(z),
        "beta": float(beta),
        "price_a": price_a,
        "price_b": price_b,
        "scale_in": scale_in,
        "z_exit": float(ZSCORE_EXIT),
        "z_stop": float(ZSCORE_STOP),
        "open_time": now_eastern(),  # datetime object
        "time_stop_min": float(PAIR_TIME_STOP_MIN),
        "spread_vol": vol,
        "predictive_score": float(predictive_score),
    }

    if LOG_PAIRS_DEBUG:
        logger.debug(
            "[Pairs] Signal %s-%s dir=%s z=%.2f beta=%.3f "
            "pA=%.2f pB=%.2f vol=%.5f scale_in=%s",
            sym_a, sym_b, direction, z, beta, price_a, price_b, vol, scale_in,
        )

    return signal

