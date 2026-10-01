"""
eod_report.py — Quant Bot V4.0

End-of-day reporting.

Supports BOTH:
1) Legacy V3 event-style logs:
   • Rows with type='single_open' and type starting with 'single_close_'
   • Reconstructs single trades from open/close events.

2) V4 analytics-style logs (from analytics.append_trade_record / log_trade):
   Columns:
       timestamp, symbol, side, qty,
       entry_price, exit_price, pnl,
       strategy, notes

For a given day, produces:
• Per-symbol PnL and stats
• Overall performance metrics
• Approx equity curve over time
• Equity & PnL-by-symbol plots
• Excel report with multiple sheets
"""

import os
from datetime import datetime, date

import pandas as pd
import matplotlib.pyplot as plt

from .config import TRADE_LOG_PATH

REPORT_DIR = "reports"


# ============================================================
# LOAD TRADE LOG (V3 + V4 COMPATIBLE)
# ============================================================

def _load_trades() -> pd.DataFrame:
    """
    Load trade_log.csv and normalize time column:

    V4 format:
        - 'timestamp' column (ISO) → parsed to datetime (UTC) → 'time'

    Legacy V3 format:
        - 'time' column already present → parsed → 'time'

    Returns a DataFrame sorted by 'time'.
    """
    if not os.path.exists(TRADE_LOG_PATH):
        raise FileNotFoundError(f"{TRADE_LOG_PATH} not found")

    df = pd.read_csv(TRADE_LOG_PATH)

    if "timestamp" in df.columns:
        # New analytics format
        df["time"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    elif "time" in df.columns:
        # Legacy format
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df["time"] = pd.NaT

    return df.sort_values("time")


# ============================================================
# LEGACY V3: RECONSTRUCT SINGLE TRADES FROM EVENTS
# ============================================================

def _build_single_trade_pnl_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy V3 path.

    Approximate single-trade PnL by matching single_open with single_close_*.

    Assumes:
        - df has columns: type, symbol, side, qty, entry, time, (possibly exit/entry for closes)
        - For each 'single_close_*' row, we match the earliest unmatched 'single_open'
          with same symbol + side (FIFO per symbol & side).

    Returns DataFrame with columns:
        symbol, side, open_time, close_time, qty, entry, exit, pnl
    """
    opens = df[df["type"] == "single_open"].copy()
    closes = df[df["type"].astype(str).str.startswith("single_close_")].copy()

    if opens.empty or closes.empty:
        return pd.DataFrame(
            columns=["symbol", "side", "open_time", "close_time", "qty", "entry", "exit", "pnl"]
        )

    trades = []
    opens["used"] = False

    for _, row in closes.iterrows():
        sym = row["symbol"]
        side_close = row["side"]
        qty_close = float(row["qty"])

        cand = opens[
            (~opens["used"]) &
            (opens["symbol"] == sym) &
            (opens["side"] == side_close)
            ]
        if cand.empty:
            continue

        o = cand.iloc[0]
        idx = o.name

        entry = float(o["entry"])

        # Try to infer exit_price from various legacy layouts
        exit_price = entry
        if "entry" in closes.columns and "exit" in closes.columns:
            exit_price = float(row.get("exit", row.get("entry", entry)))
        elif "entry" in row and "exit" not in row:
            exit_price = float(row["entry"])
        if "entry" in row and "exit" in row:
            exit_price = float(row["exit"])
        if "entry" in row and "exit_price" in row:
            exit_price = float(row["exit_price"])

        # Fallback: we logged exit_price in log_single_close as 'entry' sometimes
        try:
            exit_price = float(row["entry"])
        except Exception:
            pass

        qty = float(o["qty"])

        if side_close == "sell":  # closing a long
            pnl = (exit_price - entry) * qty
        else:  # closing a short
            pnl = (entry - exit_price) * qty

        trades.append(
            {
                "symbol": sym,
                "side": side_close,
                "open_time": o["time"],
                "close_time": row["time"],
                "qty": qty,
                "entry": entry,
                "exit": exit_price,
                "pnl": pnl,
            }
        )

        opens.loc[idx, "used"] = True

    return pd.DataFrame(trades)


# ============================================================
# V4 PATH: ANALYTICS-STYLE TRADE LOG
# ============================================================

def _build_single_trade_pnl_v4(df: pd.DataFrame) -> pd.DataFrame:
    """
    V4 path: analytics-style trade log where each row is already
    ONE realized trade.

    Expected columns:
        timestamp, symbol, side, qty,
        entry_price, exit_price, pnl,
        strategy, notes

    We:
        - Optionally filter out pair trades (keep 'single' style only)
        - Map to a common schema: symbol, side, open_time, close_time,
          qty, entry, exit, pnl
    """
    df = df.copy()

    # Filter to single-stock strategies if 'strategy' exists
    if "strategy" in df.columns:
        # keep anything that is not clearly a pair
        mask_single = ~df["strategy"].astype(str).str.contains("pair", case=False, na=False)
        df = df[mask_single].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["symbol", "side", "open_time", "close_time", "qty", "entry", "exit", "pnl"]
        )

    # Normalize columns
    qty = df["qty"] if "qty" in df.columns else 1
    entry_col = "entry_price" if "entry_price" in df.columns else None
    exit_col = "exit_price" if "exit_price" in df.columns else None

    open_time = df["time"]
    close_time = df["time"]  # we only know close time; trade is per row

    if entry_col is not None:
        entry = df[entry_col].astype(float)
    else:
        entry = pd.Series([float("nan")] * len(df), index=df.index)

    if exit_col is not None:
        exit_price = df[exit_col].astype(float)
    else:
        exit_price = pd.Series([float("nan")] * len(df), index=df.index)

    if "pnl" in df.columns:
        pnl = df["pnl"].astype(float)
    else:
        # if we somehow lack pnl, approximate from entry/exit
        pnl = (exit_price - entry) * qty

    trades = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "side": df.get("side", "long"),
            "open_time": open_time,
            "close_time": close_time,
            "qty": qty.astype(float),
            "entry": entry,
            "exit": exit_price,
            "pnl": pnl,
        }
    )

    return trades


def _build_single_trade_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dispatch to V3 or V4 builder depending on trade_log structure.
    """
    if "type" in df.columns and df["type"].astype(str).str.contains("single_open").any():
        # Legacy V3 event-based log
        return _build_single_trade_pnl_legacy(df)
    else:
        # V4 analytics-style log (or simplified single-row-per-trade log)
        return _build_single_trade_pnl_v4(df)


# ============================================================
# MAIN REPORT GENERATOR
# ============================================================

def generate_eod_report(target_date: date = None):
    if target_date is None:
        target_date = date.today()

    df = _load_trades()

    # Filter to target_date based on 'time' (local date)
    if df["time"].isna().all():
        print("No valid time column in trade_log; cannot build EOD report.")
        return

    df["local_date"] = df["time"].dt.tz_convert(None).dt.date if df["time"].dt.tz is not None else df["time"].dt.date
    df_day = df[df["local_date"] == target_date].copy()

    if df_day.empty:
        print(f"No trades for {target_date}")
        return

    single_trades = _build_single_trade_pnl(df_day)

    # Aggregate stats
    total_pnl = single_trades["pnl"].sum() if not single_trades.empty else 0.0
    wins = (single_trades["pnl"] > 0).sum()
    losses = (single_trades["pnl"] < 0).sum()
    winrate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

    per_symbol = (
        single_trades.groupby("symbol")
        .agg(
            trades=("pnl", "count"),
            pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            winrate=("pnl", lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.0),
        )
        .reset_index()
    )

    # Approx equity curve: cumulative sum of PnL over time
    eq = single_trades.sort_values("close_time").copy()
    eq["cum_pnl"] = eq["pnl"].cumsum()

    # Create reports directory
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Plot equity curve
    plt.figure()
    plt.plot(eq["close_time"], eq["cum_pnl"])
    plt.title(f"Equity Curve {target_date}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL")
    eq_path = os.path.join(REPORT_DIR, f"equity_{target_date}.png")
    plt.tight_layout()
    plt.savefig(eq_path)
    plt.close()

    # Plot PnL by symbol
    plt.figure()
    per_symbol.plot(kind="bar", x="symbol", y="pnl", legend=False)
    plt.title(f"PnL by Symbol {target_date}")
    plt.xlabel("Symbol")
    plt.ylabel("Pnl")
    pnl_sym_path = os.path.join(REPORT_DIR, f"pnl_by_symbol_{target_date}.png")
    plt.tight_layout()
    plt.savefig(pnl_sym_path)
    plt.close()

    # Export to Excel
    report_path = os.path.join(REPORT_DIR, f"eod_{target_date}.xlsx")
    with pd.ExcelWriter(report_path, engine="xlsxwriter") as writer:
        # Raw daily log
        df_day.to_excel(writer, sheet_name="RawLog", index=False)

        # Per-trade single stats
        single_trades.to_excel(writer, sheet_name="SingleTrades", index=False)

        # Per-symbol stats
        per_symbol.to_excel(writer, sheet_name="SymbolStats", index=False)

        # Summary sheet
        summary = pd.DataFrame(
            [
                {
                    "date": target_date,
                    "total_pnl": total_pnl,
                    "wins": wins,
                    "losses": losses,
                    "winrate": winrate,
                }
            ]
        )
        summary.to_excel(writer, sheet_name="Summary", index=False)

        # Graphs sheet: embed PNGs
        workbook = writer.book
        sheet = workbook.add_worksheet("Graphs")
        writer.sheets["Graphs"] = sheet

        sheet.insert_image("B2", eq_path)
        sheet.insert_image("B30", pnl_sym_path)

    print(f"EOD report written to {report_path}")


if __name__ == "__main__":
    # Run manual EOD: python -m quant_bot.eod_report
    generate_eod_report()
