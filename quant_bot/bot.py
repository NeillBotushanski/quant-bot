"""
bot.py

Main orchestrator for Quant Bot V4.0.

Responsibilities:
• Build Tier-1 liquid universe (full market → top N symbols)
• Intraday scan loop (every INTRADAY_SCAN_INTERVAL seconds)
• Market regime checks (SPY + VIX)
• Single-stock signal generation & execution (learning-aware, V4 signals)
• Pairs discovery, signal generation & execution (learning-aware sizing)
• Heartbeat + basic safety / stability
"""

import asyncio
import logging
from datetime import timedelta

from alpaca_trade_api.rest import REST

from .config import (
    INTRADAY_SCAN_INTERVAL,
    MAX_DEEP_SYMBOLS,
    SNAPSHOT_BATCH,
    USE_MARKET_FILTER,
    MAX_DRAWDOWN_PCT,
    MTF_FAST,
    MTF_SLOW,
    MAX_POSITIONS,
    SCAN_SYMBOL_LIMIT,
)

from .utils import heartbeat, now_eastern, trigger_circuit_breaker
from .data import (
    get_rest,
    fetch_snapshots,
    get_market_regime,
    compute_pairs_matrix,
)
from .filters import (
    tier1_pass,
    market_regime_pass,
)
from .signals import generate_single_signal
from .pairs import discover_pair_candidates, generate_pair_signal
from .execution import (
    compute_single_size,
    submit_single_bracket,
    submit_pair_orders,
    get_account,
    current_position_count,
    manage_pair_exits,
    manage_single_exits,
)
from .learning import should_trade_symbol

logger = logging.getLogger(__name__)


class QuantBot:
    """
    Intraday trading bot:
        • Universe builder
        • Scan loop
        • Routing to signals / execution / learning
    """

    def __init__(self):
        self.rest: REST = get_rest()
        self.symbols = []
        self.last_scan = None
        self.log = logger


    # --------------------------------------------------------
    # MARKET CLOCK
    # --------------------------------------------------------
    def is_market_open(self) -> bool:
        try:
            clock = self.rest.get_clock()
            return clock.is_open
        except Exception as e:
            logger.warning(f"[Clock] Failed to fetch market clock: {e}")
            return False

    # --------------------------------------------------------
    # UNIVERSE BUILD
    # --------------------------------------------------------
    def build_universe(self):
        """
        Build Tier-1 liquid universe:

            • Fetch all tradable assets from Alpaca
            • Filter via tier1_pass(price, dollar_volume)
            • Keep top MAX_DEEP_SYMBOLS by dollar volume
        """
        try:
            assets = self.rest.list_assets(status="active")
        except Exception as e:
            logger.error(f"[Universe] Failed to list assets: {e}")
            self.symbols = []
            return

        all_symbols = [
            a.symbol
            for a in assets
            if getattr(a, "tradable", False) and getattr(a, "class", "") == "us_equity"
        ]

        logger.info("[Universe] Fetching all tradable assets...")
        logger.info(f"[Universe] Total tradable symbols: {len(all_symbols)}")

        liquidity_map = {}  # sym -> (dollar_volume, price)

        for i in range(0, len(all_symbols), SNAPSHOT_BATCH):
            batch = all_symbols[i:i + SNAPSHOT_BATCH]
            snaps = fetch_snapshots(self.rest, batch)

            for sym, snap in snaps.items():
                try:
                    if snap is None or snap.daily_bar is None:
                        continue
                    price = float(snap.daily_bar.c)
                    vol = float(snap.daily_bar.v)
                    dollar_vol = price * vol

                    if tier1_pass(price, dollar_vol):
                        liquidity_map[sym] = (dollar_vol, price)
                except Exception:
                    continue

        pre_count = len(liquidity_map)
        logger.info(f"[Universe] Tier-1 liquid universe (pre-cut): {pre_count} symbols")

        if pre_count == 0:
            self.symbols = []
            return

        # sort by dollar volume desc & keep top MAX_DEEP_SYMBOLS
        sorted_syms = sorted(
            liquidity_map.items(),
            key=lambda kv: kv[1][0],
            reverse=True,
        )
        top_syms = [sym for sym, _ in sorted_syms[:MAX_DEEP_SYMBOLS]]

        self.symbols = top_syms
        logger.info(f"[Universe] Final Tier-1 list (deep analysis): {len(self.symbols)} symbols")

    # --------------------------------------------------------
    # INTRADAY SCAN
    # --------------------------------------------------------
    async def intraday_scan(self):
        """
        One scan cycle:
        • Throttle by INTRADAY_SCAN_INTERVAL
        • Market regime filter
        • Fetch 1m + 5m bars for Tier-1 symbols
        • Single-stock signals & execution (learning-aware)
        • Pairs discovery & execution (learning-aware sizing)
        • Exit management for singles & pairs
        """
        now = now_eastern()

        # throttle to INTRADAY_SCAN_INTERVAL
        if self.last_scan is not None:
            delta = (now - self.last_scan).total_seconds()
            if delta < INTRADAY_SCAN_INTERVAL:
                return
        self.last_scan = now

        heartbeat(self.log)

        if not self.symbols:
            logger.warning("[Scan] No symbols in universe. Rebuilding...")
            self.build_universe()
            if not self.symbols:
                return

        # --- Market regime (SPY + VIX) ---
        regime = "unknown"
        spy_close = spy_fast = spy_slow = vix_val = None

        if USE_MARKET_FILTER:
            spy_close, spy_fast, spy_slow, vix_val = get_market_regime(self.rest)

            if spy_fast is not None and spy_slow is not None:
                if spy_fast > spy_slow:
                    regime = "bullish"
                elif spy_fast < spy_slow:
                    regime = "bearish"
                else:
                    regime = "neutral"
            else:
                regime = "neutral"

            logger.info(
                f"[Scan] Market regime: {regime} (SPY fast={spy_fast}, slow={spy_slow}, VIX={vix_val})"
            )

            # Optional regime gate using numeric inputs
            vix_for_filter = vix_val if vix_val is not None else 0.0
            if not market_regime_pass(spy_fast, spy_slow, vix_for_filter, log=logger):
                logger.info(
                    f"[Scan] Market regime {regime} blocked by filter – skipping scan."
                )
                return
        else:
            regime = "bullish"  # default if filter disabled

        # --- Circuit breaker check globally ---
        account = get_account(self.rest)
        if account is None:
            logger.error("[Scan] Cannot read account, skipping.")
            return

        if trigger_circuit_breaker(account, logger, MAX_DRAWDOWN_PCT):
            logger.error("[Scan] Circuit breaker active. Skipping trading.")
            return

        # --- Fetch intraday bars (1m & 5m) for a subset of symbols ---
        end = now
        start = end - timedelta(minutes=120)  # last ~2 hours

        symbols_for_scan = self.symbols[:SCAN_SYMBOL_LIMIT]

        logger.info(
            f"[Scan] Fetching bars for {len(symbols_for_scan)} symbols (deep scan subset)..."
        )
        fast_bars_all = {}
        slow_bars_all = {}

        for sym in symbols_for_scan:
            try:
                bars_fast = self.rest.get_bars(
                    sym,
                    MTF_FAST,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=500,
                    feed="iex",
                ).df

                bars_slow = self.rest.get_bars(
                    sym,
                    MTF_SLOW,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=500,
                    feed="iex",
                ).df
            except Exception as e:
                logger.warning(f"[Scan] Error fetching bars for {sym}: {e}")
                continue

            if not bars_fast.empty:
                fast_bars_all[sym] = bars_fast.sort_index()
            if not bars_slow.empty:
                slow_bars_all[sym] = bars_slow.sort_index()

        logger.info(
            f"[Scan] Received bars for {len(fast_bars_all)} symbols (1m), "
            f"{len(slow_bars_all)} symbols (5m)"
        )

        # ====================================================
        # SINGLE-STOCK TRADING (V4 signals + learning-aware)
        # ====================================================
        pos_count = current_position_count(self.rest)
        single_trades = 0

        for sym, df_fast in fast_bars_all.items():
            if pos_count >= MAX_POSITIONS:
                break

            # Skip symbols the learning layer currently doesn't like
            allowed, reason = should_trade_symbol(sym)
            if not allowed:
                continue

            # Try modes in order: momentum → reversal → trend
            signal_obj = None
            for mode in ("momentum", "reversal", "trend"):
                signal_obj = generate_single_signal(sym, df_fast, mode=mode, log=logger)
                if signal_obj is not None:
                    break

            if signal_obj is None:
                continue

            # Optional: use regime to filter direction
            if regime == "bullish" and signal_obj.side == "short":
                continue
            if regime == "bearish" and signal_obj.side == "long":
                continue

            entry = float(signal_obj.entry)
            stop = float(signal_obj.stop)
            target = float(signal_obj.target)
            side = "buy" if signal_obj.side == "long" else "sell"

            qty = compute_single_size(account, entry, stop, symbol=sym)
            if qty <= 0:
                continue

            # Position limit safety
            if pos_count >= MAX_POSITIONS:
                break

            submit_single_bracket(
                self.rest,
                symbol=sym,
                side=side,
                qty=qty,
                entry_price=entry,
                stop_price=stop,
                target_price=target,
            )
            pos_count += 1
            single_trades += 1

            logger.info(
                f"[Single] Submitted {side.upper()} bracket for {sym}, "
                f"mode={signal_obj.mode}, entry={entry:.2f}, "
                f"stop={stop:.2f}, target={target:.2f}, qty={qty}"
            )

        logger.info(f"[Single] Executed {single_trades} single trade(s) this scan.")

        # ====================================================
        # PAIRS TRADING (LEARNING-AWARE SIZING)
        # ====================================================
        returns_matrix, symbols_list, corr_matrix = compute_pairs_matrix(fast_bars_all)
        if returns_matrix is None or symbols_list is None or corr_matrix is None:
            logger.info("[Pairs] Not enough data for correlation matrix.")
            # Even if no pairs, still manage existing exits below
        else:
            candidates = discover_pair_candidates(returns_matrix, symbols_list, corr_matrix)
            if not candidates:
                logger.info("[Pairs] No candidate pairs found this scan.")
            else:
                max_pairs_per_scan = 5
                handled = 0

                for cand in candidates:
                    if handled >= max_pairs_per_scan:
                        break

                    a = cand["sym_a"]
                    b = cand["sym_b"]
                    df_a = fast_bars_all.get(a)
                    df_b = fast_bars_all.get(b)
                    if df_a is None or df_b is None:
                        continue

                    pair_signal = generate_pair_signal(df_a, df_b, a, b)
                    if not pair_signal:
                        continue

                    submit_pair_orders(self.rest, pair_signal)
                    handled += 1

                logger.info(f"[Pairs] Executed {handled} pair(s) this scan.")

        # ====================================================
        # EXIT MANAGEMENT
        # ====================================================
        manage_pair_exits(self.rest, fast_bars_all)
        manage_single_exits(self.rest)

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------
    async def run(self):
        """
        Top-level event loop:
        • Build universe once at startup
        • While True:
            - If market open → intraday_scan
            - Else → sleep longer
        """
        logger.info("[Bot] Starting Quant Bot V4.0...")

        # initial universe
        self.build_universe()

        while True:
            if self.is_market_open():
                await self.intraday_scan()
                await asyncio.sleep(1)  # prevent spamming /clock
            else:
                logger.info("[Bot] Market closed — sleeping 60s...")
                await asyncio.sleep(60)


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        level=logging.INFO,
    )

    bot = QuantBot()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        logger.info("[Bot] Stopped by user.")


if __name__ == "__main__":
    main()
