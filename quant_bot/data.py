"""
data.py — Quant Bot V4.0

Centralized data access layer for Alpaca:
• REST client creation
• Tradable assets / universe helpers
• Snapshots for liquidity / dollar volume
• Async multi-symbol bar fetching with concurrency limits

Designed to work with:
    - alpaca_trade_api.REST (sync)
    - asyncio.to_thread for concurrency
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import alpaca_trade_api as tradeapi
except ImportError:  # pragma: no cover
    tradeapi = None  # type: ignore

from .config import (
    API_KEY,
    API_SECRET,
    BASE_URL,
    DATA_FEED,
    MIN_PRICE,
    MAX_PRICE,
    MIN_AVG_DOLLAR_VOL,
    MAX_CONCURRENT_DATA_TASKS,
    HTTP_TIMEOUT,
    LOOKBACK_CORR_BARS,
)
from .utils import now_eastern, retry_async
from .indicators import ema

logger = logging.getLogger(__name__)

# ============================================================
# REST CLIENT
# ============================================================

_rest_client: Optional["tradeapi.REST"] = None


def get_rest_client() -> "tradeapi.REST":
    """
    Lazily create and return a singleton Alpaca REST client.

    V4 still uses the familiar alpaca_trade_api.REST client, but
    wraps calls in asyncio.to_thread for concurrency.
    """
    global _rest_client
    if _rest_client is None:
        if tradeapi is None:
            raise RuntimeError(
                "alpaca_trade_api is not installed. "
                "Install it with `pip install alpaca-trade-api`."
            )
        _rest_client = tradeapi.REST(
            API_KEY,
            API_SECRET,
            BASE_URL,
            api_version="v2",
        )
    return _rest_client


# ============================================================
# SIMPLE CACHING (IN-MEMORY)
# ============================================================

@dataclass
class TimedCacheEntry:
    value: any
    expiry: datetime


_asset_cache: Optional[TimedCacheEntry] = None
ASSET_CACHE_TTL_SEC = 300  # 5 minutes


def _is_cache_valid(entry: Optional[TimedCacheEntry]) -> bool:
    return entry is not None and entry.expiry > now_eastern()


# ============================================================
# UNIVERSE HELPERS
# ============================================================

def fetch_tradable_assets(
        logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Fetch ALL tradable assets from Alpaca.

    Returns:
        DataFrame with at least columns:
            symbol, exchange, asset_class, tradable, marginable, shortable, easy_to_borrow
    """
    log = logger or globals().get("logger", logging.getLogger(__name__))

    global _asset_cache
    if _is_cache_valid(_asset_cache):
        return _asset_cache.value

    client = get_rest_client()
    log.info("[Universe] Fetching all tradable assets...")

    assets = client.list_assets(status="active")
    records = []
    for a in assets:
        if not getattr(a, "tradable", False):
            continue
        records.append(
            {
                "symbol": a.symbol,
                "exchange": a.exchange,
                "asset_class": a.asset_class,
                "tradable": bool(a.tradable),
                "marginable": bool(getattr(a, "marginable", False)),
                "shortable": bool(getattr(a, "shortable", False)),
                "easy_to_borrow": bool(getattr(a, "easy_to_borrow", False)),
            }
        )

    df = pd.DataFrame.from_records(records).sort_values("symbol").reset_index(drop=True)

    _asset_cache = TimedCacheEntry(
        value=df,
        expiry=now_eastern() + timedelta(seconds=ASSET_CACHE_TTL_SEC),
    )

    log.info(f"[Universe] Total tradable symbols: {len(df)}")
    return df


def fetch_snapshots_for_universe(
        symbols: Iterable[str],
        logger: Optional[logging.Logger] = None,
) -> Dict[str, "tradeapi.entity.Snapshot"]:
    """
    Fetch snapshot data (latest quote/trade) for a batch of symbols.

    Used to compute:
        • last price
        • volume
        • dollar volume
    """
    log = logger or globals().get("logger", logging.getLogger(__name__))

    client = get_rest_client()
    symbols = list(symbols)
    if not symbols:
        return {}

    log.info(f"[Universe] Fetching snapshots for {len(symbols)} symbols...")
    # tradeapi.REST has get_snapshots in v2
    snapshots = client.get_snapshots(symbols, feed=DATA_FEED)

    # snapshots is usually Dict[str, Snapshot]
    return snapshots


def build_liquid_universe(
        logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of liquid, tradeable stocks with basic filters:
        - price between MIN_PRICE and MAX_PRICE
        - 20d average dollar volume >= MIN_AVG_DOLLAR_VOL

    Returns:
        DataFrame with: symbol, last_price, volume, dollar_volume, ...
    """
    log = logger or globals().get("logger", logging.getLogger(__name__))

    assets_df = fetch_tradable_assets(log)
    symbols = assets_df["symbol"].tolist()

    # Fetch snapshots and construct a liquidity table
    snapshots = fetch_snapshots_for_universe(symbols, log)

    records: List[dict] = []
    for sym in symbols:
        snap = snapshots.get(sym)
        if not snap:
            continue

        latest_trade = getattr(snap, "latest_trade", None)
        daily_bar = getattr(snap, "daily_bar", None)

        price = None
        vol = None

        if latest_trade is not None:
            price = float(getattr(latest_trade, "p", 0.0))
        if daily_bar is not None:
            vol = float(getattr(daily_bar, "v", 0.0))

        if not price or price <= 0:
            continue

        if vol is None:
            vol = 0.0

        dollar_vol = price * vol

        # Filter by price and dollar volume
        if price < MIN_PRICE or price > MAX_PRICE:
            continue
        if dollar_vol < MIN_AVG_DOLLAR_VOL:
            continue

        records.append(
            {
                "symbol": sym,
                "last_price": price,
                "volume": vol,
                "dollar_volume": dollar_vol,
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        log.warning("[Universe] No assets passed liquidity filters!")
        return df

    df = df.sort_values("dollar_volume", ascending=False).reset_index(drop=True)
    return df


# ============================================================
# ASYNC BAR FETCHING (MULTI-SYMBOL)
# ============================================================

async def _fetch_bars_symbol_sync(
        symbol: str,
        timeframe: str,
        limit: int,
        end: Optional[datetime] = None,
        adjustment: str = "raw",
) -> pd.DataFrame:
    """
    Sync helper that runs in a worker thread via asyncio.to_thread.
    """
    client = get_rest_client()
    # The Alpaca get_bars API returns a BarSet-like object with .df in older versions
    bars = client.get_bars(
        symbol,
        timeframe,
        limit=limit,
        end=end,
        adjustment=adjustment,
    )
    # Support both new + old versions of alpaca_trade_api
    if hasattr(bars, "df"):
        df = bars.df
        # If multi-symbol, bars.df index has 'symbol' level; isolate ours
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        df = df.copy()
    else:
        # Newer style might return a list-like; normalize to DataFrame
        df = pd.DataFrame([b._raw for b in bars])  # type: ignore[attr-defined]

    return df


async def _fetch_bars_symbol(
        symbol: str,
        timeframe: str,
        limit: int,
        end: Optional[datetime],
        logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """
    Async wrapper with retry for a single symbol's bars.
    """

    async def _call():
        return await asyncio.to_thread(
            _fetch_bars_symbol_sync,
            symbol,
            timeframe,
            limit,
            end,
        )

    df = await retry_async(_call, logger=logger)
    if df is None:
        logger.warning(f"[Bars] Failed to fetch bars for {symbol}")
        return None

    if df.empty:
        logger.debug(f"[Bars] No bars for {symbol}")
        return None

    return df


async def fetch_bars_for_symbols(
        symbols: Iterable[str],
        timeframe: str,
        limit: int = 500,
        end: Optional[datetime] = None,
        logger: Optional[logging.Logger] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch bars for multiple symbols concurrently using asyncio.

    Args:
        symbols: list/iterable of symbols
        timeframe: e.g. "1Min", "5Min", "15Min"
        limit: number of bars to request per symbol
        end: optional end datetime (market time)
        logger: optional logger

    Returns:
        dict: {symbol: DataFrame of OHLCV bars}
    """
    log = logger or globals().get("logger", logging.getLogger(__name__))

    symbols = list(symbols)
    if not symbols:
        return {}

    log.info(f"[Scan] Fetching bars for {len(symbols)} symbols (timeframe={timeframe}, limit={limit})...")

    # Throttle concurrency to avoid hammering Alpaca
    sem = asyncio.Semaphore(MAX_CONCURRENT_DATA_TASKS)
    result: Dict[str, pd.DataFrame] = {}

    async def _task(sym: str):
        async with sem:
            df = await _fetch_bars_symbol(sym, timeframe, limit, end, log)
            if df is not None and not df.empty:
                result[sym] = df

    tasks = [asyncio.create_task(_task(s)) for s in symbols]
    await asyncio.gather(*tasks)

    log.info(
        f"[Scan] Received bars for {len(result)} symbols ({timeframe}); "
        f"requested {len(symbols)}"
    )
    return result


def fetch_bars_for_symbols_sync(
        symbols: Iterable[str],
        timeframe: str,
        limit: int = 500,
        end: Optional[datetime] = None,
        logger: Optional[logging.Logger] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Synchronous wrapper around fetch_bars_for_symbols().

    Lets existing sync-style code call into the async engine without
    refactoring the entire bot at once.

    NOTE: This runs an event loop internally. In an already-async
    environment, call fetch_bars_for_symbols() directly instead.
    """
    log = logger or globals().get("logger", logging.getLogger(__name__))
    try:
        return asyncio.run(
            fetch_bars_for_symbols(symbols, timeframe, limit=limit, end=end, logger=log)
        )
    except RuntimeError:
        # In case there's already an event loop running (e.g. inside Jupyter)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            fetch_bars_for_symbols(symbols, timeframe, limit=limit, end=end, logger=log)
        )


# ============================================================
# COMPATIBILITY SHIMS FOR bot.py V4
# ============================================================

def get_rest() -> "tradeapi.REST":
    """
    Compatibility wrapper expected by bot.py V4.
    Returns the singleton REST client.
    """
    return get_rest_client()


def fetch_snapshots(rest: "tradeapi.REST", symbols: Iterable[str]) -> Dict[str, "tradeapi.entity.Snapshot"]:
    """
    Compatibility wrapper expected by bot.py V4.

    Ignores the passed `rest` (we use our singleton client) and
    delegates to fetch_snapshots_for_universe().
    """
    # We don't actually need `rest` here because get_rest_client() uses the same creds.
    return fetch_snapshots_for_universe(symbols)


def get_market_regime(rest: "tradeapi.REST") -> Tuple[
    Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute a simple market regime from SPY:

        • spy_close: latest close
        • fast EMA (e.g., 20)
        • slow EMA (e.g., 50)
        • VIX placeholder (currently None)

    Returns:
        (spy_close, fast_ema, slow_ema, vix_value)
    """
    try:
        bars = rest.get_bars("SPY", "1Min", limit=200, feed=DATA_FEED).df
    except Exception as e:
        logger.warning(f"[Data] SPY bars fetch error: {e}")
        return None, None, None, None

    if bars is None or bars.empty:
        return None, None, None, None

    bars = bars.sort_index()
    close = bars["close"].astype(float)

    fast = ema(close, 20).iloc[-1]
    slow = ema(close, 50).iloc[-1]
    spy_close = float(close.iloc[-1])

    vix_val = None  # placeholder; wire real VIX later if you want

    return spy_close, float(fast), float(slow), vix_val


def compute_pairs_matrix(
        fast_bars_dict: Dict[str, pd.DataFrame],
) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray]]:
    """
    Build correlation matrix for pairs trading from 1-minute returns.

    Inputs:
        fast_bars_dict: dict[symbol] -> DataFrame with 'close' column

    Steps:
        • For each symbol, compute % returns on 'close'
        • Take last LOOKBACK_CORR_BARS returns
        • Stack into matrix (rows = symbols, cols = time)
        • Compute corr = np.corrcoef(matrix)

    Returns:
        (returns_matrix, symbol_list, corr_matrix)
        or (None, None, None) if not enough usable symbols.
    """
    CORR_LOOKBACK = LOOKBACK_CORR_BARS

    symbols: List[str] = []
    rows: List[np.ndarray] = []

    for sym, df in fast_bars_dict.items():
        if df is None or df.empty or "close" not in df.columns:
            continue

        close = df["close"].astype(float)
        rets = close.pct_change().dropna().tail(CORR_LOOKBACK).values

        if len(rets) < CORR_LOOKBACK:
            continue

        symbols.append(sym)
        rows.append(rets)

    if len(rows) < 5:
        # Not enough symbols for a meaningful matrix
        return None, None, None

    mat = np.array(rows)
    corr = np.corrcoef(mat)

    return mat, symbols, corr
