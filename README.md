# Quant Bot

An intraday U.S. equities trading bot written in Python for the [Alpaca](https://alpaca.markets/) paper-trading API. It scans the most liquid stocks in the market, generates single-stock and pairs-trading signals, sizes each position from a fixed risk budget, and adjusts that risk based on its own recent results.

This is a personal learning project. It was built and run against a paper (simulated) account.

## What it does

- **Universe building:** pulls all tradable assets, filters by price ($5 to $1,000) and daily dollar volume ($5M+), and keeps the most liquid names for deep scanning.
- **Single-stock signals:** computes EMA, RSI, ATR and VWAP on 1-minute and 5-minute bars, then applies momentum, reversal and trend filters to propose an entry, stop and target.
- **Pairs trading:** finds highly correlated pairs (correlation of at least 0.80 on 1-minute returns), estimates a hedge ratio, and trades the spread on z-score thresholds (enter at 1.5, exit at 0.5, stop at 4.0) with a 60-minute time stop.
- **Market regime filter:** checks the SPY trend (fast vs. slow EMA) before taking new trades.
- **Risk management:** risks 0.15% of equity per single-stock trade and 0.30% per pair, caps the number of open positions, and stops trading if the account draws down 12% from its high-water mark.
- **Adaptive risk layer:** reads the trade log to track win rates per symbol and overall, scales position size up or down accordingly, and puts symbols on cooldown after losses. It is a statistical risk governor, not a machine-learning model.
- **Reporting:** syncs fills from Alpaca into a trade log and produces end-of-day reports with per-symbol P&L, an equity curve and an Excel summary.
- **Backtesting:** an offline backtester for the single-stock strategy that runs on 1-minute OHLCV data.

## Project structure

```text
live_bot.py              entry point
quant_bot/
  bot.py                 main loop: universe build, intraday scan, heartbeat
  config.py              all settings (risk, thresholds, scan intervals)
  data.py                Alpaca data access, async bar fetching, market regime
  indicators.py          EMA, ATR, RSI, VWAP, volatility, candle metrics
  signals.py             single-stock signal generation
  filters.py             pass/fail gates for signals, pairs and market regime
  pairs.py               pair discovery, hedge ratio, spread and z-score signals
  execution.py           position sizing, bracket orders, exits, circuit breaker
  learning.py            adaptive risk multipliers, cooldowns and blacklist
  analytics.py           trade log and open-pairs persistence
  alpaca_trades.py       sync fills from Alpaca into the trade log
  analyze_trades.py      per-symbol and time-of-day performance
  eod_report.py          end-of-day report with plots and Excel output
  backtester.py          offline backtester for the single-stock strategy
  utils.py               time helpers, retries, circuit breaker, z-score
```

## Setup

Requires Python 3.9+ and an Alpaca paper-trading account.

```bash
pip install -r requirements.txt
```

API keys are read from environment variables, so they never live in the code:

```bash
export ALPACA_API_KEY="your-paper-trading-key"
export ALPACA_API_SECRET="your-paper-trading-secret"
```

Then run the bot:

```bash
python live_bot.py
```

All strategy and risk settings are in `quant_bot/config.py`.

## Disclaimer

This project is for learning and research. It is not investment advice, and it has only been run against a paper-trading account. Use it at your own risk.
