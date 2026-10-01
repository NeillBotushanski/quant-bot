"""
backtester.py

Offline backtesting engine for Quant Bot V3.0 (single-stock strategy).

Usage pattern (example):

    from quant_bot.backtester import Backtester
    import pandas as pd

    # Load your 1m data for one or more symbols
    df_aapl = pd.read_csv("AAPL_1m.csv", parse_dates=["t"]).set_index("t")
    df_msft = pd.read_csv("MSFT_1m.csv", parse_dates=["t"]).set_index("t")

    data = {
        "AAPL": df_aapl,
        "MSFT": df_msft,
    }

    bt = Backtester(initial_equity=100_000)
    results = bt.run_universe(data)

    print(results["summary"])
    print(results["trades"].head())

Notes:
- Assumes columns: ["open", "high", "low", "close", "volume"] on a 1m index.
- One open position per symbol at a time (no pyramiding).
- Capital path is approximated by applying trade PnL sequentially.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import RISK_PER_TRADE
from .signals import generate_single_stock_signal

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class BacktestTrade:
    symbol: str
    direction: str  # "long" or "short"
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    pnl_pct: float
    reason: str


# ============================================================
# HELPER: RESAMPLE 1M → 5M
# ============================================================

def _resample_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute bars into 5-minute OHLCV bars.
    Keeps same column naming convention as live code.
    """
    if df_1m.empty:
        return df_1m.copy()

    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df_5m = df_1m.resample("5T").agg(ohlc_dict).dropna()
    return df_5m


# ============================================================
# CORE BACKTESTER
# ============================================================

class Backtester:
    def __init__(self, initial_equity: float = 100_000.0,
                 risk_per_trade: Optional[float] = None):
        """
        initial_equity: starting capital for the backtest.
        risk_per_trade: fraction of equity to risk per trade.
                        Defaults to config.RISK_PER_TRADE.
        """
        self.initial_equity = float(initial_equity)
        self.risk_per_trade = float(risk_per_trade) if risk_per_trade is not None else RISK_PER_TRADE

    # --------------------------------------------------------
    # SINGLE SYMBOL BACKTEST
    # --------------------------------------------------------
    def _backtest_symbol(
            self,
            symbol: str,
            df_1m: pd.DataFrame,
    ) -> List[BacktestTrade]:
        """
        Backtest the single-stock strategy on one symbol.

        Uses:
            - 1m data as df_fast
            - resampled 5m data as df_slow
        """
        if df_1m is None or df_1m.empty:
            return []

        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(df_1m.columns):
            logger.warning(
                "[Backtester] %s missing required columns; has=%s",
                symbol,
                df_1m.columns,
            )
            return []

        df_fast = df_1m.sort_index().copy()
        df_slow = _resample_to_5m(df_fast)

        trades: List[BacktestTrade] = []

        position_open = False
        pos_direction: Optional[str] = None
        entry_price: float = 0.0
        stop_price: float = 0.0
        target_price: float = 0.0
        qty: int = 0
        entry_time: Optional[pd.Timestamp] = None

        # Simplified capital handling: use initial equity for sizing.
        # If you want dynamic sizing, you can update equity after each trade.
        equity_for_sizing = self.initial_equity

        # We start after a warmup period to let indicators form
        start_idx = 60  # ~1 hour of 1m bars
        if len(df_fast) <= start_idx:
            return []

        for i in range(start_idx, len(df_fast)):
            t = df_fast.index[i]
            row = df_fast.iloc[i]

            # Align slow bars up to current time
            df_fast_slice = df_fast.iloc[: i + 1]
            df_slow_slice = df_slow.loc[:t]

            if df_slow_slice.empty:
                continue

            if not position_open:
                # Try to open a new trade (long or short)
                # You can bias to long-only by removing "short" here.
                open_signal: Optional[Tuple[str, float, float, float]] = None

                for direction in ("long", "short"):
                    sig = generate_single_stock_signal(
                        df_fast_slice,
                        df_slow_slice,
                        direction=direction,
                    )
                    if sig:
                        open_signal = sig
                        break

                if open_signal is None:
                    continue

                pos_direction, entry, stop, target = open_signal

                # Compute quantity from risk_per_trade * equity
                risk_per_share = abs(entry - stop)
                if risk_per_share <= 0:
                    continue

                max_loss = equity_for_sizing * self.risk_per_trade
                raw_qty = max_loss / risk_per_share
                # No over-leverage: cap by "all in" notionally
                max_qty_by_capital = equity_for_sizing / entry
                qty = int(np.floor(min(raw_qty, max_qty_by_capital)))

                if qty <= 0:
                    continue

                position_open = True
                entry_price = float(entry)
                stop_price = float(stop)
                target_price = float(target)
                entry_time = t

                logger.debug(
                    "[Backtester] %s OPEN %s at %.2f (stop=%.2f, target=%.2f, qty=%d)",
                    symbol,
                    pos_direction,
                    entry_price,
                    stop_price,
                    target_price,
                    qty,
                )
                continue

            # Manage existing position
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            exit_reason = None
            exit_price = None

            if pos_direction == "long":
                # Check stop first
                if low <= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop"
                # Then take-profit
                elif high >= target_price:
                    exit_price = target_price
                    exit_reason = "target"
            elif pos_direction == "short":
                # For shorts, stop if price trades above stop
                if high >= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop"
                elif low <= target_price:
                    exit_price = target_price
                    exit_reason = "target"

            # End-of-data forced exit is handled after the loop

            if exit_reason is not None and entry_time is not None:
                pnl = (
                    (exit_price - entry_price) * qty
                    if pos_direction == "long"
                    else (entry_price - exit_price) * qty
                )
                pnl_pct = pnl / (entry_price * qty)

                trade = BacktestTrade(
                    symbol=symbol,
                    direction=pos_direction,
                    entry_time=entry_time,
                    exit_time=t,
                    entry_price=entry_price,
                    exit_price=float(exit_price),
                    qty=qty,
                    pnl=float(pnl),
                    pnl_pct=float(pnl_pct),
                    reason=exit_reason,
                )
                trades.append(trade)

                position_open = False
                pos_direction = None
                qty = 0
                entry_time = None

        # Force close at last bar if position still open
        if position_open and entry_time is not None:
            last_row = df_fast.iloc[-1]
            last_time = df_fast.index[-1]
            last_close = float(last_row["close"])

            if pos_direction == "long":
                exit_price = last_close
                pnl = (exit_price - entry_price) * qty
            else:
                exit_price = last_close
                pnl = (entry_price - exit_price) * qty

            pnl_pct = pnl / (entry_price * qty)

            trade = BacktestTrade(
                symbol=symbol,
                direction=pos_direction or "long",
                entry_time=entry_time,
                exit_time=last_time,
                entry_price=entry_price,
                exit_price=float(exit_price),
                qty=qty,
                pnl=float(pnl),
                pnl_pct=float(pnl_pct),
                reason="eod",
            )
            trades.append(trade)

        return trades

    # --------------------------------------------------------
    # UNIVERSE BACKTEST
    # --------------------------------------------------------
    def run_universe(
            self,
            data_1m: Dict[str, pd.DataFrame],
    ) -> Dict[str, object]:
        """
        Backtest single-stock strategy across a dict of symbol → 1m DataFrame.

        Returns dict with:
            {
                "summary": { ...metrics... },
                "trades": pd.DataFrame of trades
            }
        """
        all_trades: List[BacktestTrade] = []

        for symbol, df in data_1m.items():
            logger.info("[Backtester] Running %s...", symbol)
            trades_sym = self._backtest_symbol(symbol, df)
            all_trades.extend(trades_sym)

        trades_df = self._trades_to_df(all_trades)
        summary = self._compute_summary(trades_df)

        return {
            "summary": summary,
            "trades": trades_df,
        }

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    @staticmethod
    def _trades_to_df(trades: List[BacktestTrade]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "direction",
                    "entry_time",
                    "exit_time",
                    "entry_price",
                    "exit_price",
                    "qty",
                    "pnl",
                    "pnl_pct",
                    "reason",
                ]
            )

        records = [
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "qty": t.qty,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "reason": t.reason,
            }
            for t in trades
        ]
        df = pd.DataFrame.from_records(records)
        df.sort_values(["entry_time", "exit_time"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _compute_summary(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        if trades_df.empty:
            return {
                "initial_equity": self.initial_equity,
                "final_equity": self.initial_equity,
                "total_pnl": 0.0,
                "total_return_pct": 0.0,
                "num_trades": 0,
                "win_rate": 0.0,
                "avg_pnl_per_trade": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_like": 0.0,
            }

        # Build equity curve by applying PnL sequentially
        equity = self.initial_equity
        equity_curve = []

        for _, row in trades_df.iterrows():
            equity += row["pnl"]
            equity_curve.append(equity)

        equity_curve = np.array(equity_curve, dtype=float)
        final_equity = float(equity_curve[-1])
        total_pnl = float(final_equity - self.initial_equity)
        total_return_pct = total_pnl / self.initial_equity

        num_trades = len(trades_df)
        wins = (trades_df["pnl"] > 0).sum()
        win_rate = wins / num_trades if num_trades > 0 else 0.0
        avg_pnl_per_trade = trades_df["pnl"].mean() if num_trades > 0 else 0.0

        # Max drawdown on equity curve
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_dd_pct = float(drawdowns.min()) if drawdowns.size > 0 else 0.0

        # Simple Sharpe-like ratio using per-trade returns (not annualized)
        trade_returns = trades_df["pnl"] / self.initial_equity
        if trade_returns.std() > 0:
            sharpe_like = float(trade_returns.mean() / trade_returns.std())
        else:
            sharpe_like = 0.0

        return {
            "initial_equity": self.initial_equity,
            "final_equity": final_equity,
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "num_trades": int(num_trades),
            "win_rate": float(win_rate),
            "avg_pnl_per_trade": float(avg_pnl_per_trade),
            "max_drawdown_pct": float(max_dd_pct),
            "sharpe_like": sharpe_like,
        }
