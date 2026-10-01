"""
execution.py — Quant Bot V4.0

Order sizing and execution for Quant Bot.

Handles:
• Single-stock position sizing & bracket orders
• Beta-neutral pairs sizing & both-leg orders
• Max position limits
• Circuit breaker checks
• Global exit management for singles and pairs

V4 changes:
• Single-stock sizing uses the learning layer risk multipliers
  (symbol + global) via learning.get_risk_multipliers().
• All public function signatures kept backwards-compatible:
    - get_account
    - get_open_positions
    - current_position_count
    - compute_single_size
    - submit_single_bracket
    - compute_pair_sizes
    - submit_pair_orders
    - manage_pair_exits
    - manage_single_exits
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from alpaca_trade_api.rest import REST

from . import config as cfg
from .utils import trigger_circuit_breaker, now_eastern
from .pairs import build_spread_series
from .analytics import log_pair_open, load_open_pairs, save_open_pairs, log_trade
from .learning import get_risk_multipliers

# ------------------------------------------------------------
# CONFIG WITH SAFE DEFAULTS (WORKS WITH V3 config.py)
# ------------------------------------------------------------

RISK_PER_TRADE = getattr(cfg, "RISK_PER_TRADE", 0.0015)
PAIRS_RISK_PER_TRADE = getattr(cfg, "PAIRS_RISK_PER_TRADE", 0.003)
MAX_POSITIONS = getattr(cfg, "MAX_POSITIONS", 6)
MAX_DRAWDOWN_PCT = getattr(cfg, "MAX_DRAWDOWN_PCT", 0.12)

SINGLE_HARD_STOP_PCT = getattr(cfg, "SINGLE_HARD_STOP_PCT", 0.04)
SINGLE_TAKE_PROFIT_PCT = getattr(cfg, "SINGLE_TAKE_PROFIT_PCT", 0.08)

ZSCORE_EXIT = getattr(cfg, "ZSCORE_EXIT", 0.5)
ZSCORE_STOP = getattr(cfg, "ZSCORE_STOP", 4.0)
PAIR_TIME_STOP_MIN = getattr(cfg, "PAIR_TIME_STOP_MIN", 60)

# Optional toggles, default ON if not defined
MANAGE_MANUAL_POSITIONS = getattr(cfg, "MANAGE_MANUAL_POSITIONS", True)
LIVE_TRADING_ENABLED = getattr(cfg, "LIVE_TRADING_ENABLED", True)

logger = logging.getLogger(__name__)


# ============================================================
# HELPER: GET ACCOUNT & POSITIONS
# ============================================================

def get_account(rest: REST):
    """
    Safely fetch Alpaca account object.
    """
    try:
        return rest.get_account()
    except Exception as e:
        logging.error(f"[Execution] Failed to fetch account: {e}")
        return None


def get_open_positions(rest: REST):
    """
    Return list of current Alpaca positions.
    """
    try:
        positions = rest.list_positions()
        return positions
    except Exception as e:
        logging.error(f"[Execution] Failed to fetch positions: {e}")
        return []


def current_position_count(rest: REST) -> int:
    """
    Simple count of open positions.
    """
    return len(get_open_positions(rest))


# ============================================================
# SINGLE-STOCK POSITION SIZING
# ============================================================

def compute_single_size(account, entry_price, stop_price, symbol: str = None) -> int:
    try:
        equity = float(account.equity)
    except Exception:
        logging.error("[Execution] Could not read account equity.")
        return 0

    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0 or entry_price <= 0:
        return 0

    base_risk = equity * RISK_PER_TRADE

    if symbol is not None:
        try:
            sym_mult, glob_mult, combined = get_risk_multipliers(symbol)
        except Exception as e:
            logging.warning(f"[Execution] get_risk_multipliers failed for {symbol}: {e}")
            combined = 1.0
    else:
        combined = 1.0

    combined = max(0.25, min(combined, 2.0))
    effective_risk = base_risk * combined

    # Size from risk
    qty = int(math.floor(effective_risk / risk_per_share))
    if qty <= 0:
        return 0

    # 🔒 New: cap notional to, say, 20% of equity
    MAX_EQUITY_PER_POSITION = 0.20  # 20%
    max_notional = equity * MAX_EQUITY_PER_POSITION
    max_qty_notional = int(max_notional // entry_price)

    qty = min(qty, max_qty_notional)
    qty = max(qty, 0)

    return qty


def submit_single_bracket(
        rest: REST,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        stop_price: float,
        target_price: float,
):
    """
    Submit a single-stock bracket order (market entry, TP, SL).

    side: 'buy' or 'sell'
    """
    if qty <= 0:
        logging.info(f"[Execution] Skipping {symbol}: qty <= 0")
        return

    if not LIVE_TRADING_ENABLED:
        logging.info(
            "[Execution] (DRY RUN) Would submit %s bracket for %s qty=%d "
            "stop=%.2f target=%.2f",
            side, symbol, qty, stop_price, target_price,
        )
        return

    try:
        rest.submit_order(
            symbol=symbol,
            side=side,
            type="market",
            qty=qty,
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": round(target_price, 2)},
            stop_loss={"stop_price": round(stop_price, 2)},
        )
        logging.info(
            f"[Execution] Submitted {side.upper()} bracket for {symbol}, "
            f"qty={qty}, stop={stop_price:.2f}, target={target_price:.2f}"
        )
    except Exception as e:
        logging.error(f"[Execution] Failed to submit single bracket for {symbol}: {e}")


# ============================================================
# PAIRS POSITION SIZING (BETA-NEUTRAL)
# ============================================================

def compute_pair_sizes(account, price_a: float, price_b: float, beta: float) -> Tuple[int, int]:
    """
    Beta-neutral sizing:

        - Risk per pair = PAIRS_RISK_PER_TRADE * equity
        - Approximate 'capital at risk' across both legs instead of
          modeling full spread ATR.
        - Allocate capital across A and B in beta proportion.

    Returns:
        (qty_a, qty_b)
    """
    try:
        equity = float(account.equity)
    except Exception:
        logging.error("[Execution] Could not read account equity for pairs.")
        return 0, 0

    if price_a <= 0 or price_b <= 0:
        return 0, 0

    total_capital = equity * PAIRS_RISK_PER_TRADE

    # weights: A weight = 1, B weight = |beta|
    w_a = 1.0
    w_b = abs(beta) if abs(beta) > 0 else 1.0
    w_sum = w_a + w_b

    cap_a = total_capital * (w_a / w_sum)
    cap_b = total_capital * (w_b / w_sum)

    qty_a = int(math.floor(cap_a / price_a))
    qty_b = int(math.floor(cap_b / price_b))

    return max(qty_a, 0), max(qty_b, 0)


def submit_pair_orders(
        rest: REST,
        signal: dict,
):
    """
    Execute a pair trade based on generated pair signal.

    signal keys:
        "sym_a", "sym_b", "direction", "beta", "price_a", "price_b"
        (plus optional: "z_entry", "z_exit", "z_stop", "time_stop_min")

    direction:
        "short_spread": short A, long β * B
        "long_spread":  long A, short β * B
    """
    sym_a = signal["sym_a"]
    sym_b = signal["sym_b"]
    direction = signal["direction"]
    beta = float(signal["beta"])
    price_a = float(signal["price_a"])
    price_b = float(signal["price_b"])

    account = get_account(rest)
    if account is None:
        logging.error("[Execution] Cannot read account for pair order.")
        return

    # Circuit breaker check
    if trigger_circuit_breaker(account, logger, MAX_DRAWDOWN_PCT):
        logging.error("[Execution] Circuit breaker active. Skipping pair trade.")
        return

    qty_a, qty_b = compute_pair_sizes(account, price_a, price_b, beta)
    if qty_a <= 0 or qty_b <= 0:
        logging.info(f"[Execution] Pair {sym_a}-{sym_b} size is zero → skipping.")
        return

    # Position limit (each leg counts as a position)
    pos_count = current_position_count(rest)
    if pos_count + 2 > MAX_POSITIONS:
        logging.info(
            f"[Execution] Max positions reached ({pos_count}), cannot open new pair."
        )
        return

    if direction == "short_spread":
        # short A, long B
        side_a = "sell"
        side_b = "buy"
    else:
        # long_spread
        side_a = "buy"
        side_b = "sell"

    logging.info(
        f"[Execution] Executing pair {sym_a}-{sym_b}, "
        f"dir={direction}, beta={beta:.2f}, qtyA={qty_a}, qtyB={qty_b}"
    )

    if not LIVE_TRADING_ENABLED:
        logging.info(
            "[Execution] (DRY RUN) Would submit pair: %s %s x%d, %s %s x%d",
            side_a, sym_a, qty_a, side_b, sym_b, qty_b,
        )
        return

    try:
        rest.submit_order(
            symbol=sym_a,
            side=side_a,
            type="market",
            qty=qty_a,
            time_in_force="day",
        )
        rest.submit_order(
            symbol=sym_b,
            side=side_b,
            type="market",
            qty=qty_b,
            time_in_force="day",
        )

        # Record in analytics / open_pairs registry
        log_pair_open(signal, qty_a, qty_b)

    except Exception as e:
        logging.error(f"[Execution] Failed to submit pair orders for {sym_a}-{sym_b}: {e}")


# ============================================================
# PAIR EXIT MANAGEMENT
# ============================================================

def manage_pair_exits(rest: REST, fast_bars_all: Dict[str, 'pd.DataFrame']):
    """
    Check all open pairs and close any that:
        • Have |z| <= z_exit  (mean reversion target hit)
        • Have |z| >= z_stop  (protective stop)
        • Have exceeded time_stop_min minutes since open
    """
    open_pairs = load_open_pairs()
    if not open_pairs:
        return

    updated_rows: List[dict] = []
    now = now_eastern()  # timezone-aware "now"

    for row in open_pairs:
        try:
            sym_a = row["sym_a"]
            sym_b = row["sym_b"]
            direction = row["direction"]
            beta = float(row["beta"])
            qty_a = int(row["qty_a"])
            qty_b = int(row["qty_b"])

            z_exit = float(row.get("z_exit", ZSCORE_EXIT))
            z_stop = float(row.get("z_stop", ZSCORE_STOP))
            time_stop_min = float(row.get("time_stop_min", PAIR_TIME_STOP_MIN))

            open_time_str = row.get("open_time", "")
            try:
                open_dt = datetime.fromisoformat(open_time_str)
            except Exception:
                open_dt = None

            # Get fresh data for both legs
            df_a = fast_bars_all.get(sym_a)
            df_b = fast_bars_all.get(sym_b)

            if df_a is None or df_b is None:
                # no fresh data → keep pair as-is
                updated_rows.append(row)
                continue

            spread_series, _, _, _ = build_spread_series(df_a, df_b)
            if spread_series is None or len(spread_series) < 10:
                updated_rows.append(row)
                continue

            mean = float(spread_series.mean())
            std = float(spread_series.std())
            if std == 0:
                updated_rows.append(row)
                continue

            last_val = float(spread_series.iloc[-1])
            z = (last_val - mean) / std

            # Age in minutes
            if open_dt is None:
                age_min = 0.0
            else:
                if open_dt.tzinfo is None:
                    # align now to naive
                    age_min = (now.replace(tzinfo=None) - open_dt).total_seconds() / 60.0
                else:
                    now_aligned = now.astimezone(open_dt.tzinfo)
                    age_min = (now_aligned - open_dt).total_seconds() / 60.0

            close_reason = None
            if abs(z) <= z_exit:
                close_reason = "z_target"
            elif abs(z) >= z_stop:
                close_reason = "z_stop"
            elif age_min >= time_stop_min:
                close_reason = "time_stop"

            if close_reason is None:
                updated_rows.append(row)
                continue

            logging.info(
                "[Pairs Exit] Closing pair %s-%s, reason=%s, z=%.2f, age=%.1fm",
                sym_a, sym_b, close_reason, z, age_min,
            )

            # Close sides opposite to entry
            if direction == "short_spread":
                # originally: short A, long B
                side_a_close = "buy"
                side_b_close = "sell"
            else:
                # originally: long A, short B
                side_a_close = "sell"
                side_b_close = "buy"

            if LIVE_TRADING_ENABLED:
                rest.submit_order(
                    symbol=sym_a,
                    side=side_a_close,
                    type="market",
                    qty=qty_a,
                    time_in_force="day",
                )
                rest.submit_order(
                    symbol=sym_b,
                    side=side_b_close,
                    type="market",
                    qty=qty_b,
                    time_in_force="day",
                )

            # Do NOT append to updated_rows → pair is closed

        except Exception as e:
            logging.error(f"[Pairs Exit] Error handling pair row {row}: {e}")
            # keep the row to avoid losing tracking on error
            updated_rows.append(row)

    # Write back remaining open pairs
    save_open_pairs(updated_rows)


# ============================================================
# SINGLE EXIT MANAGEMENT (INCL. MANUAL POSITIONS)
# ============================================================

def manage_single_exits(rest: REST):
    """
    Scan ALL open positions in the Alpaca account and close
    any that violate global risk rules:
        • loss worse than SINGLE_HARD_STOP_PCT
        • profit greater than SINGLE_TAKE_PROFIT_PCT

    This applies to:
        • positions opened by this bot (with or without brackets)
        • positions opened manually or by other scripts
    """
    if not MANAGE_MANUAL_POSITIONS:
        # Feature switch – kept for future use, but default is True.
        return

    try:
        positions = rest.list_positions()
    except Exception as e:
        logging.error(f"[Single Exit] Failed to fetch positions: {e}")
        return

    if not positions:
        return

    for pos in positions:
        try:
            symbol = pos.symbol
            side = pos.side  # 'long' or 'short'
            qty = int(abs(float(pos.qty)))
            if qty <= 0:
                continue

            avg_price = float(pos.avg_entry_price)
            current_price = float(pos.current_price)

            if avg_price <= 0 or current_price <= 0:
                continue

            # PnL % normalized: positive = good, negative = bad
            if side == "long":
                pl_pct = (current_price - avg_price) / avg_price
                close_side = "sell"
            else:  # short
                pl_pct = (avg_price - current_price) / avg_price
                close_side = "buy"

            reason = None
            if pl_pct <= -SINGLE_HARD_STOP_PCT:
                reason = "hard_stop"
            elif pl_pct >= SINGLE_TAKE_PROFIT_PCT:
                reason = "take_profit"

            if reason is None:
                continue

            logging.info(
                "[Single Exit] Closing %s qty=%d side=%s pl_pct=%.2f%% reason=%s",
                symbol, qty, side, pl_pct * 100.0, reason,
            )

            if LIVE_TRADING_ENABLED:
                rest.submit_order(
                    symbol=symbol,
                    side=close_side,
                    type="market",
                    qty=qty,
                    time_in_force="day",
                )
                try:
                    # Realized PnL in dollars (approx at current market price)
                    if side == "long":
                        pnl = (current_price - avg_price) * qty
                        trade_side = "long"
                    else:
                        pnl = (avg_price - current_price) * qty
                        trade_side = "short"

                    log_trade(
                        symbol=symbol,
                        side=trade_side,
                        qty=qty,
                        entry_price=avg_price,
                        exit_price=current_price,
                        pnl=pnl,
                        strategy="single_exit_" + reason,
                    )
                except Exception as e:
                    logging.error(f"[Single Exit] Failed to log trade for {symbol}: {e}")



        except Exception as e:
            logging.error(f"[Single Exit] Error handling position {pos}: {e}")
