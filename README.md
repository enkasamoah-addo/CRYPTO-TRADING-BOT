# Trading bot (learning project)

A multi-agent crypto trading bot: Scout screens pairs, Charts proposes trades,
the Skeptic can veto any trade, Chief places approved orders. Full plan is in the
project plan doc. Profit is a bonus; this is a learning and resume project.

## Status

- [x] Step 1: project structure, config, data pipeline, decision log
- [x] Step 2: Scout (volume and volatility screen)
- [x] Step 3: Charts (moving average crossover, long only)
- [x] Step 4: Skeptic (8 rule-based vetoes) and position sizing
- [x] Step 5: Backtest (naive vs Skeptic, whole year plus four slices)
- [x] Step 6a: Chief in PAPER mode (simulated orders on real prices)
- [x] Step 6b: strategy research. Daily trend filter passed dev and the final exam (modest edge; see results)
- [x] Step 7: paper trading of the daily trend filter (`paper_trend.yaml`)
- [ ] Step 6c: Binance Testnet executor (only worth building once a strategy passes research)

## Setup

Needs Python 3.10 or newer.

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python -m pytest                 # all tests should pass

## Download historical data

Uses Binance public market data, so no API key is needed.

    python -m bot.download

This saves one CSV per symbol in `data/` and prints a quality report (gaps,
bad candles, missing values). If it reports PROBLEMS FOUND, fix that before
backtesting. Settings (symbols, timeframe, days of history) are in `config.yaml`.

## Preview what Scout and Charts see

After downloading data:

    python -m bot.preview

Prints which pairs Scout picks and how many crossover signals Charts found in
the downloaded history, with the last few. It places no orders and judges no
trades; that is the Skeptic's job (Step 4).

## Backtest

    python -m bot.backtest

Replays the downloaded history through Charts and the Skeptic with simulated
money, twice: "naive" (every signal taken) and with the Skeptic. Prints results
for the full period and for four equal slices, plus buy-and-hold for comparison,
and which Skeptic rules blocked how many signals. Trade-by-trade details are saved
as CSV files in `data/`.

How to read it: judge consistency across the slices, not one big number. Profit
factor (PF) above 1 means winners outweighed losers after fees. If the strategy
loses to buy-and-hold, or only wins in one slice, treat it as no edge found.
Do not tune settings to make a slice look better; that is overfitting.

## Strategy research (daily candles, BTC and ETH)

Three fixed candidates, tested under rules written down in advance (`research.yaml`):
A breakout (20-day high in, 10-day low out), B trend filter (above 100-day average),
C 20/50 crossover as a control. Risk 1% of equity per trade, 2 x ATR protective stop.

    python -m bot.download --config research.yaml      daily candles from Sept 2017
    python -m bot.research dev --config research.yaml  development test (before 2023-10-22)
    python -m bot.research final --config research.yaml  final exam, ONE run only

Do not edit `research.yaml` after seeing results. The final exam refuses to run a second time.

## Paper trading the trend filter (the strategy that passed research)

    python -m bot.chief --config paper_trend.yaml --once      one cycle now
    python -m bot.chief --config paper_trend.yaml --loop      one cycle after every daily close (UTC midnight)
    python -m bot.chief --config paper_trend.yaml --status    equity, positions, versus buy-and-hold

Simulated orders on real prices; nothing touches an exchange. It keeps its own state in
`data/paper_trend.sqlite3`, separate from the earlier crossover experiment. Rules match
`bot/research.py` exactly (a test checks trade-for-trade parity). The kill switch is set to 30%
because the tested strategy normally suffers drawdowns up to ~25%; if it fires, stop and look.
Expect long quiet stretches and mostly small losing trades. Judge it after months, not days.

## Dashboard

    python -m bot.dashboard --config paper_trend.yaml

Writes `data/dashboard.html`; open it in your browser. Equity against buy-and-hold, open positions
with stops, candlestick charts with the 100-day average and simulated buys and sells, closed trades.
It only reads the database (opened read-only), so it cannot change a trade. Re-run it any time to
refresh. Charts load a small library from the internet; the numbers and tables show without it.

## Running in the cloud (GitHub)

`.github/workflows/daily.yml` runs one paper cycle every day at 00:10 UTC, using
`paper_trend_cloud.yaml` (same rules as `paper_trend.yaml`, but the bot's memory lives in `state/`).
Each run first checks that Binance is reachable (`python -m bot.netcheck`), then trades, rebuilds
`state/dashboard.html` and saves everything back to the repository. Run it by hand from the
repository's Actions tab ("Run workflow"). Only ONE place should run the bot: stop any local
`--loop` once the cloud job is live, or the two copies will keep different books.

## Chief (paper trading, old crossover experiment)

    python -m bot.chief --once          run one cycle now
    python -m bot.chief --loop          run at every candle close; leave the window open
    python -m bot.chief --status        equity, open positions, recent decisions
    python -m bot.chief --reset-halt    clear the kill switch after reviewing what happened

Paper mode only: orders are simulated on real market prices and nothing is sent to an
exchange. State is saved in `data/decisions.sqlite3`, so stopping and restarting is safe.
The kill switch (10% drawdown) blocks new entries until you reset it by hand.

IMPORTANT: Chief currently runs the moving average crossover, which the backtests showed
has no usable edge after costs. Paper trading it tests the machinery (data, agents, orders,
kill switch), not a money-making strategy. Do not put real money behind it.

## Chief (paper trading)

Chief runs the whole pipeline once per candle close and places SIMULATED orders at
real market prices. Nothing is ever sent to an exchange, and there is no real-money mode.

    python -m bot.chief --once         run one cycle now
    python -m bot.chief --loop         run at every candle close (Ctrl+C to stop; safe to restart)
    python -m bot.chief --status       account, open positions, recent decisions
    python -m bot.chief --reset-halt   clear the kill switch after reviewing what happened

State is kept in `data/decisions.sqlite3`, so a restart resumes where it stopped and a
signal can never be acted on twice. The kill switch (max drawdown) blocks new entries
until you reset it. The strategy it runs is the one that FAILED backtesting, so this
is a test of the system's plumbing, not a profit attempt.

## Layout

    config.yaml            all tunable settings, including the Skeptic's rules
    bot/data.py            download, quality checks, CSV storage, synthetic test data
    bot/models.py          Proposal: the trade idea passed from Charts to the Skeptic
    bot/decision_log.py    SQLite log of every proposal and verdict
    bot/download.py        command-line data download
    bot/indicators.py      moving average and ATR (no lookahead)
    bot/charts.py          Charts agent: crossover signals -> Proposal objects
    bot/scout.py           Scout agent: volume and volatility screen
    bot/preview.py         command-line preview of Scout and Charts
    bot/skeptic.py         Skeptic agent: rule-based vetoes, every failed rule reported
    bot/sizing.py          position size from risk, concentration and cash limits
    bot/backtest.py        backtester and its command-line report
    bot/netcheck.py        checks Binance prices are reachable (cloud job)
    bot/dashboard.py       read-only HTML dashboard for the paper bot
    bot/trend_chief.py     Chief variant running the daily trend filter
    paper_trend.yaml       settings for paper trading the trend filter
    bot/research.py        daily-candle research: 3 candidates, pass test, one-shot final exam
    research.yaml          fixed research settings (do not edit after seeing results)
    bot/chief.py           Chief agent: runs the pipeline each candle, paper orders
    bot/executor.py        paper executor (simulated fills with fees and slippage)
    bot/feed.py            live public market data for Chief
    bot/state.py           Chief's memory in SQLite (positions, trades, kill switch)
    bot/chief.py           Chief: runs the pipeline each candle, paper orders, kill switch
    bot/state.py           SQLite memory for Chief (account, positions, trades, seen signals)
    bot/executor.py        PaperExecutor: simulated fills with fees and slippage
    bot/feed.py            live public market data for Chief
    tests/                 unit tests (run on generated data, no network needed)

## Security

Never commit `.env`. When you create Testnet keys later, give them trading
permission only, never withdrawals.
