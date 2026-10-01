"""
analytics.py — Quant Bot V4.0

Lightweight analytics + persistence helpers:

• Append realized trades to trade_log.csv
• Track currently open pairs in open_pairs.json
• Provide utilities for other modules (learning, execution, eod_report)

This module is intentionally simple and file-based so it works
on any environment without a database.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import TRADE_LOG_PATH, DATA_DIR

logger = logging.getLogger(__name__)

OPEN_PAIRS_PATH = DATA_DIR / "open_pairs.json"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TradeRecord:
    """
    One realized trade.

    This is what ends up in trade_log.csv.
    """
    timestamp: str  # ISO string
    symbol: str
    side: str  # "long" or "short"
    qty: int
    entry_price: float
    exit_price: float
    pnl: float  # can be absolute or R-units; learning is agnostic
    strategy: str  # e.g. "single_momentum", "pair", "manual"
    notes: str = ""


# ============================================================
# TRADE LOG APPEND
# ============================================================

def _ensure_trade_log_header(path: Path) -> None:
    """
    Make sure trade_log.csv has a header row if it doesn't exist.
    """
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


def append_trade_record(record: TradeRecord) -> None:
    """
    Append a realized trade to trade_log.csv.
    """
    _ensure_trade_log_header(TRADE_LOG_PATH)

    row = asdict(record)
    try:
        with TRADE_LOG_PATH.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    row["timestamp"],
                    row["symbol"],
                    row["side"],
                    row["qty"],
                    f"{row['entry_price']:.4f}",
                    f"{row['exit_price']:.4f}",
                    f"{row['pnl']:.6f}",
                    row["strategy"],
                    row.get("notes", ""),
                ]
            )
        logger.info(
            "[Analytics] Logged trade %s side=%s qty=%d pnl=%.4f",
            row["symbol"], row["side"], row["qty"], row["pnl"],
        )
    except Exception as e:
        logger.error(f"[Analytics] Failed to append trade record: {e}")


# Convenience helper if something wants to log a trade without
# constructing TradeRecord manually.
def log_trade(
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        strategy: str,
        notes: str = "",
        ts: Optional[datetime] = None,
) -> None:
    """
    Quick helper to log a realized trade.
    """
    if ts is None:
        ts = datetime.utcnow()
    rec = TradeRecord(
        timestamp=ts.isoformat(),
        symbol=symbol,
        side=side,
        qty=qty,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        strategy=strategy,
        notes=notes,
    )
    append_trade_record(rec)


# ============================================================
# OPEN PAIRS REGISTRY
# ============================================================

def load_open_pairs() -> List[Dict[str, Any]]:
    """
    Load currently open pairs from open_pairs.json.

    File format: list[dict], where each dict looks like:

        {
          "sym_a": "QQQ",
          "sym_b": "TQQQ",
          "direction": "short_spread",
          "beta": 0.34,
          "price_a": 430.12,
          "price_b": 75.23,
          "qty_a": 3,
          "qty_b": 13,
          "z_entry": 2.3,
          "z_exit": 0.5,
          "z_stop": 4.0,
          "time_stop_min": 60,
          "open_time": "2025-11-28T12:02:58.000000",
          "spread_vol": 0.0123,
          "predictive_score": 0.0456
        }

    Returns an empty list if file missing or invalid.
    """
    if not OPEN_PAIRS_PATH.exists():
        return []

    try:
        with OPEN_PAIRS_PATH.open("r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        logger.error(f"[Analytics] Failed to load open_pairs.json: {e}")
        return []


def save_open_pairs(rows: List[Dict[str, Any]]) -> None:
    """
    Save list of open pair rows to open_pairs.json.

    Ensures open_time is serialized as ISO string if it's a datetime.
    """
    # Normalize datetimes
    norm_rows: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        ot = r.get("open_time")
        if isinstance(ot, datetime):
            r["open_time"] = ot.isoformat()
        norm_rows.append(r)

    try:
        OPEN_PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OPEN_PAIRS_PATH.open("w") as f:
            json.dump(norm_rows, f, default=str, indent=2)
    except Exception as e:
        logger.error(f"[Analytics] Failed to save open_pairs.json: {e}")


def log_pair_open(signal: Dict[str, Any], qty_a: int, qty_b: int) -> None:
    """
    Record a newly-opened pair.

    Called from execution.submit_pair_orders(signal, ...).
    """
    rows = load_open_pairs()

    open_time = signal.get("open_time")
    if isinstance(open_time, datetime):
        open_time_str = open_time.isoformat()
    else:
        # If already string or missing, just cast to str
        open_time_str = str(open_time) if open_time is not None else datetime.utcnow().isoformat()

    row = {
        "sym_a": signal["sym_a"],
        "sym_b": signal["sym_b"],
        "direction": signal["direction"],
        "beta": float(signal["beta"]),
        "price_a": float(signal["price_a"]),
        "price_b": float(signal["price_b"]),
        "qty_a": int(qty_a),
        "qty_b": int(qty_b),
        "z_entry": float(signal.get("zscore", 0.0)),
        "z_exit": float(signal.get("z_exit", ZSCORE_EXIT := 0.5)),
        "z_stop": float(signal.get("z_stop", ZSCORE_STOP := 4.0)),
        "time_stop_min": float(signal.get("time_stop_min", 60.0)),
        "open_time": open_time_str,
        "spread_vol": float(signal.get("spread_vol", 0.0)),
        "predictive_score": float(signal.get("predictive_score", 0.0)),
    }

    rows.append(row)
    save_open_pairs(rows)

    logger.info(
        "[Analytics] Logged open pair %s-%s dir=%s beta=%.3f qtyA=%d qtyB=%d",
        row["sym_a"],
        row["sym_b"],
        row["direction"],
        row["beta"],
        row["qty_a"],
        row["qty_b"],
    )
