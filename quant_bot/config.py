"""
config.py

GLOBAL CONFIGURATION FOR QUANT BOT V4.0
Balanced risk mode + full-market scanning + adaptive learning layer.
"""

import os
from pathlib import Path

# ============================================================
# API (read from environment variables; never hard-code keys)
# ============================================================

API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# If you use live at some point, you’d switch BASE_URL appropriately.


# ============================================================
# PATHS / DIRECTORIES
# ============================================================

# Base directory = folder containing this file
BASE_DIR = Path(__file__).resolve().parent

# Where we keep data, logs, open_pairs.json, trade_log.csv, etc.
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Main trade log used by learning, analytics, and eod_report
TRADE_LOG_PATH = DATA_DIR / "trade_log.csv"

# ============================================================
# GENERAL BOT SETTINGS
# ============================================================

TIMEZONE = "America/New_York"  # or "America/Toronto" if you prefer local

# Which Alpaca data feed to use
# Options: "iex" (free), "sip" (paid)
DATA_FEED = "iex"

# Live vs dry-run toggle. When False, orders are just logged.
LIVE_TRADING_ENABLED = True

# How often we send a heartbeat log (seconds)
HEARTBEAT_INTERVAL = 10

# Max number of retries for Alpaca requests
MAX_RETRIES = 3
RETRY_SLEEP = 2  # seconds

# Circuit breaker: if equity drawdown from high-water > this fraction, stop trading
MAX_DRAWDOWN_PCT = 0.12  # 12%

# ============================================================
# UNIVERSE / SCAN SETTINGS (USED BY bot.py & data.py)
# ============================================================

# Time between full scans (seconds)
INTRADAY_SCAN_INTERVAL = 15  # ~ every 15s

# How many of the most liquid symbols to keep in the Tier-1 universe
MAX_DEEP_SYMBOLS = 1000  # matches your log: "Final Tier-1 list (deep analysis): 1000 symbols"

# How many symbols per snapshot batch when building the initial universe
SNAPSHOT_BATCH = 200

# Whether to use SPY/VIX market-regime filter
USE_MARKET_FILTER = True

# Bar resolutions for intraday scan
MTF_FAST = "1Min"
MTF_SLOW = "5Min"

# Max symbols to deeply scan per intraday loop (subset of Tier-1)
SCAN_SYMBOL_LIMIT = 150  # matches your log: "Fetching bars for 150 symbols"

# Max total open positions (singles + pair legs count individually)
MAX_POSITIONS = 6

# ============================================================
# DATA / ASYNC SETTINGS (USED BY data.py)
# ============================================================

# Maximum number of concurrent async tasks when fetching data
MAX_CONCURRENT_DATA_TASKS = 20

# HTTP timeout for data requests (seconds)
HTTP_TIMEOUT = 30

# ============================================================
# UNIVERSE LIQUIDITY FILTERS (USED BY data.py / filters.py)
# ============================================================

# Minimum and maximum price for symbols in the tradeable universe
MIN_PRICE = 5.0  # skip sub-$5 penny stocks
MAX_PRICE = 1000.0  # ignore ultra-expensive illiquid stuff

# Minimum daily dollar volume to be considered Tier-1 liquid
# (price * volume from yesterday's daily bar)
MIN_DOLLAR_VOL = 5_000_000  # $5M+
# Some versions use MIN_AVG_DOLLAR_VOL instead; define both.
MIN_AVG_DOLLAR_VOL = 5_000_000

# ============================================================
# SINGLE-STOCK RISK / POSITION SIZING (USED BY execution.py)
# ============================================================

# Risk per single-stock trade as fraction of account equity
RISK_PER_TRADE = 0.0015  # 0.15% of equity per single trade

# SINGLE-STOCK GLOBAL EXIT RULES (for any open position, incl. manual)
SINGLE_HARD_STOP_PCT = 0.04  # -4% hard stop
SINGLE_TAKE_PROFIT_PCT = 0.08  # +8% take profit

# Should the bot also manage / close **manually opened** positions?
MANAGE_MANUAL_POSITIONS = True

# ============================================================
# PAIRS RISK / SETTINGS (USED BY execution.py & pairs.py)
# ============================================================

# Risk budget for a pair (both legs together) as fraction of equity
PAIRS_RISK_PER_TRADE = 0.003  # 0.3% of equity per pair

# Min correlation for candidate pairs
MIN_PAIR_CORR = 0.80

# Lookback bars for spread / correlation calculations
LOOKBACK_CORR_BARS = 80

# Minimum spread volatility (std) to consider a pair tradable
MIN_SPREAD_STD = 0.002

# Z-score thresholds
ZSCORE_ENTRY = 1.5
ZSCORE_EXIT = 0.5
ZSCORE_STOP = 4.0

# Time stop for pairs (minutes)
PAIR_TIME_STOP_MIN = 60

# Extra debug logging for pairs
LOG_PAIRS_DEBUG = False

# ============================================================
# INDICATOR SETTINGS (USED BY indicators.py & signals.py)
# ============================================================

# EMAs for intraday trend and momentum
EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 50
EMA_TREND = 100

# RSI settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# ATR for volatility-based stops/targets
ATR_PERIOD = 14
ATR_MULT_STOP = 1.5
ATR_MULT_TRAIL = 2.0

# Reward-to-risk target multiplier (used after ATR stop)
RR_TARGET = 2.0

# Max hold time for single trades (minutes) – used by signals metadata
MAX_SINGLE_HOLD_MIN = 60

# Reversal-specific thresholds
REVERSAL_RSI_DEEP_OVERSOLD = 20
REVERSAL_RSI_DEEP_OVERBOUGHT = 80
REVERSAL_MIN_WICK_RATIO = 1.5  # how “long” wick must be vs body to count as reversal candle

# Extra debug logging for signal generation
LOG_SIGNALS_DEBUG = False
# Extra filters for momentum / reversal strategies
MOMENTUM_MIN_RVOL = 1.5  # min relative volume (1.5x normal)
MOMENTUM_MIN_INTRA_RANGE_PCT = 1.0  # min intraday range as % of open
MOMENTUM_MIN_TREND_SLOPE = 0.0005  # min EMA trend slope (not used heavily but required)

# Pairs / cointegration threshold
ADF_PVAL_MAX = 0.10  # max p-value for ADF test on spread

# ============================================================
# LEARNING / ADAPTATION (USED BY learning.py)
# ============================================================

# How many trades for a symbol before we trust its stats at all
MIN_TRADES_FOR_DECISION = 5

# How many trades before we allow cooldown / blacklist behavior
MIN_TRADES_FOR_BLACKLIST = 15

# Global risk lookback window (last N trades)
GLOBAL_WINDOW = 30

# Cooldown in days after a losing trade for symbols with enough history
SYMBOL_COOLDOWN_DAYS = 3

# Per-symbol risk bands: (winrate_threshold, size_multiplier)
# Sorted ascending by threshold.
SYMBOL_RISK_BANDS = [
    (0.20, 0.25),
    (0.40, 0.50),
    (0.55, 0.75),
    (0.70, 1.00),
    (0.80, 1.25),
]

# Global risk bands based on cumulative PnL over last GLOBAL_WINDOW trades.
# Here pnl is assumed to be in "R" units or %; tune thresholds as you wish.
GLOBAL_RISK_BANDS = [
    (-0.15, 0.50),  # big losing streak → halve risk
    (-0.08, 0.75),  # moderate drawdown → 75% risk
    (0.00, 1.00),  # flat or positive → normal risk
    (0.05, 1.25),  # good performance → slightly increase risk
    (0.10, 1.50),  # very good performance → more aggressive
]

# Extra debug logging for learning / risk engine
LOG_LEARNING_DEBUG = False

# ============================================================
# MISC / OTHER FLAGS (FOR FUTURE USE)
# ============================================================

# For VIX / SPY regime calculations, you may define VIX symbol, etc., in data.py.
# Nothing else is strictly required here for V4.
