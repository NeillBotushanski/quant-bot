"""
analyze_trades.py

Quick analytics for Quant Bot V3.0.

Reads trade_log.csv and computes:
- Per-trade PnL & PnL%
- Overall summary stats
- Per-symbol performance
- Time-of-day performance (by entry hour)

Outputs:
- Prints tables to console
- Saves PNG plots:
    symbol_pnl.png
    symbol_winrate.png
    hourly_pnl.png
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quant_bot.config import TRADE_LOG_PATH


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_trades():
    if not os.path.exists(TRADE_LOG_PATH):
        raise FileNotFoundError(
            f"{TRADE_LOG_PATH} not found. Run the bot long enough to "
            f"generate some trades first."
        )

    df = pd.read_csv(TRADE_LOG_PATH)

    # Only keep single trade rows for now
    mask_single = df["type"].str.startswith("single_open") | df["type"].str.startswith(
        "single_close"
    )
    df = df[mask_single].copy()

    # Parse time
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.sort_values("time")

    return df


def build_trade_round_trips(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-trade 'round trips' using single_close rows.

    Note:
        Because analytics.log_single_close already logs entry_price & exit_price,
        we don't need to match against the open row to compute PnL.
        We still keep open time for approximate holding period if possible.
    """
    closes = df[df["type"].str.startswith("single_close")].copy()
    if closes.empty:
        raise ValueError("No single_close rows found in trade_log.csv yet.")

    # Ensure numeric
    closes["entry"] = pd.to_numeric(closes["entry"], errors="coerce")
    closes["stop"] = pd.to_numeric(closes["stop"], errors="coerce")
    closes["qty"] = pd.to_numeric(closes["qty"], errors="coerce")
    closes["qty"].fillna(0, inplace=True)

    # entry column holds entry_price, stop holds exit_price (by our logging convention)
    closes.rename(columns={"entry": "entry_price", "stop": "exit_price"}, inplace=True)

    # Compute PnL per share
    side = closes["side"].str.lower().fillna("")
    is_buy = side == "buy"

    closes["pnl_per_share"] = np.where(
        is_buy,
        closes["exit_price"] - closes["entry_price"],
        closes["entry_price"] - closes["exit_price"],
    )

    closes["pnl"] = closes["pnl_per_share"] * closes["qty"]
    closes["pnl_pct"] = closes["pnl_per_share"] / closes["entry_price"]

    # Approximate holding period:
    # For now we don't strictly match open rows; we just keep close time as exit_time.
    closes["exit_time"] = closes["time"]
    closes["entry_time"] = closes["exit_time"]  # placeholder; can be improved later

    return closes


def summarize_overall(trades: pd.DataFrame):
    total_trades = len(trades)
    wins = (trades["pnl"] > 0).sum()
    losses = (trades["pnl"] < 0).sum()
    breakeven = total_trades - wins - losses

    winrate = wins / total_trades * 100 if total_trades > 0 else 0.0
    avg_pnl = trades["pnl"].mean()
    avg_pnl_pct = trades["pnl_pct"].mean() * 100
    median_pnl_pct = trades["pnl_pct"].median() * 100

    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = trades.loc[trades["pnl"] < 0, "pnl"].sum()
    net_pnl = trades["pnl"].sum()

    print("\n===== OVERALL PERFORMANCE =====")
    print(f"Total trades : {total_trades}")
    print(f"Wins / Losses / BE : {wins} / {losses} / {breakeven}")
    print(f"Win rate     : {winrate:.2f}%")
    print(f"Avg PnL      : ${avg_pnl:,.2f}")
    print(f"Avg PnL%     : {avg_pnl_pct:.2f}%")
    print(f"Median PnL%  : {median_pnl_pct:.2f}%")
    print(f"Gross Profit : ${gross_profit:,.2f}")
    print(f"Gross Loss   : ${gross_loss:,.2f}")
    print(f"Net PnL      : ${net_pnl:,.2f}")


def summarize_by_symbol(trades: pd.DataFrame):
    grp = trades.groupby("symbol")

    summary = pd.DataFrame({
        "trades": grp.size(),
        "winrate": grp.apply(lambda g: (g["pnl"] > 0).mean() * 100),
        "avg_pnl_pct": grp["pnl_pct"].mean() * 100,
        "net_pnl": grp["pnl"].sum(),
    }).sort_values("net_pnl", ascending=False)

    print("\n===== PERFORMANCE BY SYMBOL =====")
    print(summary.head(20))

    # Plot avg pnl% for top/bottom symbols with enough trades
    min_trades = 5
    filt = summary[summary["trades"] >= min_trades]

    if not filt.empty:
        top = filt.sort_values("avg_pnl_pct", ascending=False).head(15)
        bottom = filt.sort_values("avg_pnl_pct", ascending=True).head(15)

        plt.figure(figsize=(12, 6))
        combined = pd.concat([top, bottom])
        combined = combined[~combined.index.duplicated(keep="first")]
        combined["avg_pnl_pct"].plot(kind="bar")
        plt.title("Avg PnL% by Symbol (top & bottom, >=5 trades)")
        plt.ylabel("Avg PnL%")
        plt.tight_layout()
        plt.savefig("symbol_pnl.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        filt["winrate"].sort_values(ascending=False).head(25).plot(kind="bar")
        plt.title("Win Rate by Symbol (>=5 trades)")
        plt.ylabel("Win rate (%)")
        plt.tight_layout()
        plt.savefig("symbol_winrate.png")
        plt.close()


def summarize_by_hour(trades: pd.DataFrame):
    """
    Time-of-day effect based on close time (proxy for trade timing).
    If you later log true entry_time, switch to that.
    """
    # Use entry_time if present, else fall back to exit_time/time column
    if "entry_time" in trades.columns:
        times = pd.to_datetime(trades["entry_time"], errors="coerce", utc=True)
    else:
        times = pd.to_datetime(trades["time"], errors="coerce", utc=True)

    trades = trades.copy()
    trades["hour"] = times.dt.hour

    grp = trades.groupby("hour")
    summary = pd.DataFrame({
        "trades": grp.size(),
        "winrate": grp.apply(lambda g: (g["pnl"] > 0).mean() * 100),
        "avg_pnl_pct": grp["pnl_pct"].mean() * 100,
        "net_pnl": grp["pnl"].sum(),
    }).sort_index()

    print("\n===== PERFORMANCE BY HOUR (UTC) =====")
    print(summary)

    # Plot avg pnl% by hour
    plt.figure(figsize=(10, 5))
    summary["avg_pnl_pct"].plot(kind="bar")
    plt.title("Avg PnL% by Hour (UTC)")
    plt.xlabel("Hour of Day (UTC)")
    plt.ylabel("Avg PnL%")
    plt.tight_layout()
    plt.savefig("hourly_pnl.png")
    plt.close()


def main():
    print(f"Loading trades from {TRADE_LOG_PATH} ...")
    df_raw = load_trades()
    trades = build_trade_round_trips(df_raw)

    summarize_overall(trades)
    summarize_by_symbol(trades)
    summarize_by_hour(trades)

    print("\nPlots saved as:")
    print("  symbol_pnl.png")
    print("  symbol_winrate.png")
    print("  hourly_pnl.png")


if __name__ == "__main__":
    main()
