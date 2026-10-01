"""
sync_alpaca_trades.py

Sync trades from Alpaca into trade_log.csv, supporting SCALED trades.

Logic:
    - Fetch FILL activities from Alpaca for the last N days.
    - For each symbol, walk fills in time order and maintain a signed position:
          pos_qty > 0  => net long
          pos_qty < 0  => net short
          pos_qty = 0  => flat
    - Track:
          * avg_price (VWAP entry price for the open leg)
          * realized_pnl_cycle (PnL realized during this cycle)
          * closed_qty_total (total quantity closed in this cycle)
          * trade_side ("long" or "short")
    - Whenever a position goes from non-zero -> zero, log ONE trade row:
          timestamp = time of the final closing fill
          side      = "long" or "short" (direction of the cycle)
          qty       = total closed quantity in that cycle
          pnl       = realized_pnl_cycle
          entry_price = avg entry VWAP
          exit_price  = last close price (approx; exit VWAP could be added later)

This supports:
    - Scaling in (multiple entry buys/sells)
    - Scaling out (multiple exit sells/buys)
    - Direction flips (close old + open new in one big fill)
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import alpaca_trade_api as tradeapi

from .config import API_KEY, API_SECRET, BASE_URL, TRADE_LOG_PATH


# ============================================================
# LOCAL TRADE_LOG HELPERS
# ============================================================

def _ensure_trade_log_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "symbol",
                "side",
                "qty",
                "entry_price",
                "exit_price",
                "pnl",
                "strategy",
                "notes",
            ]
        )


def _load_existing_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "side",
                "qty",
                "entry_price",
                "exit_price",
                "pnl",
                "strategy",
                "notes",
            ]
        )
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "side",
                "qty",
                "entry_price",
                "exit_price",
                "pnl",
                "strategy",
                "notes",
            ]
        )
    return df


def _append_trades(path: Path, new_rows: List[Dict[str, Any]]) -> None:
    """
    Append new trade rows to trade_log.csv.
    """
    if not new_rows:
        return

    _ensure_trade_log_header(path)

    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        for row in new_rows:
            writer.writerow(
                [
                    row["timestamp"],
                    row["symbol"],
                    row["side"],
                    int(row["qty"]),
                    f"{row['entry_price']:.4f}",
                    f"{row['exit_price']:.4f}",
                    f"{row['pnl']:.6f}",
                    row.get("strategy", "alpaca_sync_scaled"),
                    row.get("notes", ""),
                ]
            )


# ============================================================
# ALPACA FETCH
# ============================================================

def _fetch_fills(days_back: int = 2) -> pd.DataFrame:
    """
    Fetch FILL activities from Alpaca for the last N days.
    """
    client = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

    after_dt = datetime.utcnow() - timedelta(days=days_back)
    after_str = after_dt.isoformat() + "Z"

    activities = client.get_activities(
        activity_types="FILL",
        after=after_str,
    )

    records: List[Dict[str, Any]] = []
    for a in activities:
        try:
            t = pd.to_datetime(a.transaction_time, utc=True)
            records.append(
                {
                    "time": t,
                    "symbol": a.symbol,
                    "side": a.side.lower(),  # "buy" or "sell"
                    "qty": float(a.qty),
                    "price": float(a.price),
                    "activity_id": a.id,
                }
            )
        except Exception:
            continue

    if not records:
        return pd.DataFrame(columns=["time", "symbol", "side", "qty", "price", "activity_id"])

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ============================================================
# POSITION-BASED ROUND-TRIP RECONSTRUCTION (SCALED)
# ============================================================

def _reconstruct_trades_scaled(df_fills: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Reconstruct trades allowing scaling in/out.

    For each symbol:
        - Maintain signed pos_qty (long>0, short<0).
        - Maintain avg_price for the open position.
        - Accumulate realized_pnl_cycle + closed_qty_total.
        - When pos_qty goes to zero => log one trade.
        - If a fill both closes old and opens new (direction flip), we:
            1) finalize old trade
            2) start a new cycle with the remaining qty.
    """
    trades: List[Dict[str, Any]] = []
    if df_fills.empty:
        return trades

    for symbol, sdf in df_fills.groupby("symbol"):
        sdf = sdf.sort_values("time").reset_index(drop=True)

        pos_qty = 0.0  # signed
        avg_price = 0.0
        trade_side: str | None = None  # "long" or "short"
        realized_pnl_cycle = 0.0
        closed_qty_total = 0.0
        last_exit_price = 0.0
        entry_time: datetime | None = None

        i = 0
        while i < len(sdf):
            row = sdf.iloc[i]
            time = row["time"]
            side = row["side"]           # "buy" or "sell"
            qty = float(row["qty"])
            price = float(row["price"])
            signed_qty = qty if side == "buy" else -qty

            # Case 1: currently flat -> this fill opens a new position
            if pos_qty == 0.0:
                pos_qty = signed_qty
                avg_price = price
                trade_side = "long" if pos_qty > 0 else "short"
                realized_pnl_cycle = 0.0
                closed_qty_total = 0.0
                last_exit_price = 0.0
                entry_time = time
                i += 1
                continue

            # Case 2: same direction (adding to existing position)
            if (pos_qty > 0 and signed_qty > 0) or (pos_qty < 0 and signed_qty < 0):
                # Update VWAP entry price
                new_total_qty_abs = abs(pos_qty) + abs(signed_qty)
                if new_total_qty_abs > 0:
                    avg_price = (
                        avg_price * abs(pos_qty) + price * abs(signed_qty)
                    ) / new_total_qty_abs
                pos_qty += signed_qty
                i += 1
                continue

            # Case 3: opposite direction => this fill is at least partially closing
            # existing position. It might:
            #   - partially close (still same direction afterwards)
            #   - fully close
            #   - close and reverse (flip direction)
            # Determine how much of this fill closes the old position.
            qty_to_close = min(abs(pos_qty), abs(signed_qty))

            if pos_qty > 0:
                # Closing part of a long
                pnl_close = (price - avg_price) * qty_to_close
            else:
                # Closing part of a short
                pnl_close = (avg_price - price) * qty_to_close

            realized_pnl_cycle += pnl_close
            closed_qty_total += qty_to_close
            last_exit_price = price

            # Update position after using qty_to_close to close the old leg
            if abs(signed_qty) > qty_to_close:
                # This fill is bigger than the remaining position:
                #   1) fully close the old
                #   2) start a new position with the remainder
                # Compute direction of the new position from the leftover
                leftover_signed = signed_qty + (-pos_qty)  # since qty_to_close = abs(pos_qty)
                # Finalize old trade (now flat)
                if trade_side is not None and closed_qty_total > 0:
                    trades.append(
                        {
                            "timestamp": time.isoformat(),
                            "symbol": symbol,
                            "side": trade_side,
                            "qty": closed_qty_total,
                            "entry_price": avg_price,
                            "exit_price": last_exit_price,
                            "pnl": realized_pnl_cycle,
                            "strategy": "alpaca_bracket_scaled",
                            "notes": "scaled_cycle_flip",
                        }
                    )

                # Start new cycle with leftover
                pos_qty = leftover_signed
                avg_price = price
                trade_side = "long" if pos_qty > 0 else "short"
                realized_pnl_cycle = 0.0
                closed_qty_total = 0.0
                last_exit_price = 0.0
                entry_time = time
            else:
                # This fill is smaller or equal to remaining position:
                # just reduce or flatten without reversing direction.
                new_pos = pos_qty + signed_qty
                if new_pos == 0.0:
                    # Fully flat -> finalize trade
                    if trade_side is not None and closed_qty_total > 0:
                        trades.append(
                            {
                                "timestamp": time.isoformat(),
                                "symbol": symbol,
                                "side": trade_side,
                                "qty": closed_qty_total,
                                "entry_price": avg_price,
                                "exit_price": last_exit_price,
                                "pnl": realized_pnl_cycle,
                                "strategy": "alpaca_bracket_scaled",
                                "notes": "scaled_cycle_flat",
                            }
                        )
                    pos_qty = 0.0
                    avg_price = 0.0
                    trade_side = None
                    realized_pnl_cycle = 0.0
                    closed_qty_total = 0.0
                    last_exit_price = 0.0
                    entry_time = None
                else:
                    # Partially closed, still same direction
                    pos_qty = new_pos
                    # avg_price remains the same for the remaining leg

            i += 1

        # If we end the loop with an open position (pos_qty != 0),
        # we do NOT log a trade yet (no round-trip).
        # You could optionally log open positions elsewhere if desired.

    return trades


# ============================================================
# MAIN ENTRY
# ============================================================

def sync_alpaca_trades(days_back: int = 2) -> None:
    """
    Main entry:
        1) Fetch FILLs from Alpaca for last N days
        2) Reconstruct scaled round-trip trades
        3) Deduplicate against existing trade_log.csv
        4) Append new trades
    """
    print(f"[sync_alpaca_trades] Fetching FILLs from Alpaca for last {days_back} day(s)...")
    df_fills = _fetch_fills(days_back=days_back)
    if df_fills.empty:
        print("[sync_alpaca_trades] No FILL activities found.")
        return

    print(f"[sync_alpaca_trades] Got {len(df_fills)} FILL rows. Reconstructing scaled trades...")
    trades = _reconstruct_trades_scaled(df_fills)
    if not trades:
        print("[sync_alpaca_trades] No completed round-trip trades reconstructed.")
        return

    print(f"[sync_alpaca_trades] Reconstructed {len(trades)} trade(s). Loading existing trade_log...")
    existing = _load_existing_trades(TRADE_LOG_PATH)

    # Deduplicate by (symbol, side, entry_price, exit_price, qty, timestamp)
    if not existing.empty:
        existing_keys = set(
            zip(
                existing.get("symbol", []),
                existing.get("side", []),
                existing.get("entry_price", []),
                existing.get("exit_price", []),
                existing.get("qty", []),
                existing.get("timestamp", []),
            )
        )
    else:
        existing_keys = set()

    new_rows: List[Dict[str, Any]] = []
    for tr in trades:
        key = (
            tr["symbol"],
            tr["side"],
            tr["entry_price"],
            tr["exit_price"],
            tr["qty"],
            tr["timestamp"],
        )
        if key in existing_keys:
            continue
        new_rows.append(tr)

    if not new_rows:
        print("[sync_alpaca_trades] No new trades to append (all already logged).")
        return

    print(f"[sync_alpaca_trades] Appending {len(new_rows)} new trade(s) to {TRADE_LOG_PATH} ...")
    _append_trades(TRADE_LOG_PATH, new_rows)
    print("[sync_alpaca_trades] Done.")


if __name__ == "__main__":
    # Default: sync last 2 days of fills.
    sync_alpaca_trades(days_back=2)
