"""Chief agent: runs the whole pipeline once per candle and places approved orders.

PAPER MODE ONLY: orders are simulated (see executor.py); nothing is sent to an exchange.

Each cycle, in this order (it mirrors the backtester so live behaviour matches what was tested):
  1. Fetch complete candles. If data is missing or stale, new entries are blocked.
  2. Check every open position for a stop or target hit since it was last checked.
  3. Mark the account to market; update daily P&L, peak equity and the kill switch.
  4. For each new signal: Skeptic judges it; if approved, size it and "buy".

Safety properties:
- State lives in SQLite, so a restart resumes exactly where it stopped.
- A signal is recorded before any order, so it can never be acted on twice.
- The kill switch (max drawdown) is latched: it blocks new entries until a human runs
  `python -m bot.chief --reset-halt`. Open positions keep their stops.

Run from the project folder:
    python -m bot.chief --once        one cycle now
    python -m bot.chief --loop        one cycle at every candle close, forever (Ctrl+C to stop)
    python -m bot.chief --status      show account, positions and recent decisions
    python -m bot.chief --reset-halt  clear the kill switch after you have reviewed what happened
"""
import argparse
import sqlite3
import sys
import time
from typing import List

import pandas as pd

from .backtest import check_exit
from .charts import latest_proposal
from .config import load_config
from .data import timeframe_ms
from .decision_log import init_db, log_decision
from .executor import PaperExecutor
from .sizing import position_size
from .skeptic import AccountState, Skeptic
from .state import Store


class Chief:
    def __init__(self, cfg: dict, conn: sqlite3.Connection, feed, executor):
        self.cfg = cfg
        self.conn = conn
        self.feed = feed
        self.executor = executor
        self.skeptic = Skeptic(cfg)
        self.store = Store(conn, float(cfg["backtest"]["starting_equity"]))
        self.tf = pd.Timedelta(milliseconds=timeframe_ms(cfg["data"]["timeframe"]))
        self.symbols = cfg["data"]["symbols"]

    @staticmethod
    def _equity(cash, positions, last_close) -> float:
        return cash + sum(p["qty"] * last_close.get(p["symbol"], p["entry_fill"]) for p in positions)

    def _new_signals(self, frames, equity, cash, last_close, day_start, drawdown, data_ok, notes):
        cfg, store = self.cfg, self.store
        for symbol, df in frames.items():
            proposal = latest_proposal(df, symbol, cfg)
            if proposal is None or store.signal_seen(symbol, proposal.created_at):
                continue
            store.mark_signal(symbol, proposal.created_at)
            positions = store.open_positions()
            if any(p["symbol"] == symbol for p in positions):
                notes.append(f"{symbol}: signal ignored, already holding it")
                continue
            swing_n = cfg["skeptic"]["swing_lookback_candles"]
            swing = df["low"].rolling(swing_n).min().iloc[-1]
            state = AccountState(
                equity=equity, open_positions=len(positions),
                daily_pnl_pct=(equity - day_start) / day_start * 100, drawdown_pct=drawdown,
                halted=bool(store.get("halted")), data_ok=data_ok)
            verdict = self.skeptic.judge(proposal, state, None if pd.isna(swing) else float(swing))
            log_decision(self.conn, proposal, "approved" if verdict.approved else "blocked", verdict.reason)
            if not verdict.approved:
                notes.append(f"{symbol}: signal BLOCKED - {verdict.reason}")
                continue

            price = self.feed.price(symbol)
            if not (proposal.stop < price < proposal.target):
                notes.append(f"{symbol}: approved but price {price:,.2f} is already outside stop/target, skipped")
                continue
            ref = self.executor.expected_buy_price(price)
            mark = self._equity(cash, positions, last_close)
            qty = position_size(ref, proposal.stop, mark, cash, cfg, cfg["skeptic"]["max_open_positions"])
            if qty <= 0 or qty * ref < 1.0:
                notes.append(f"{symbol}: approved but position size is too small, skipped")
                continue
            fill = self.executor.buy(symbol, qty, price)
            cost = fill.qty * fill.price + fill.fee
            cash -= cost
            store.set("cash", cash)
            store.add_position(symbol, fill.qty, fill.price, proposal.stop, proposal.target,
                               proposal.created_at, cost, df["timestamp"].iloc[-1].isoformat())
            notes.append(f"{symbol}: BOUGHT {fill.qty:.6f} at {fill.price:,.2f} "
                         f"(stop {proposal.stop:,.2f}, target {proposal.target:,.2f})")

    def run_cycle(self, now: pd.Timestamp) -> List[str]:
        """One pass of the pipeline. Returns a list of plain-language notes about what happened."""
        cfg, store = self.cfg, self.store
        notes: List[str] = []

        # 1) Data
        frames, data_ok = {}, True
        for symbol in self.symbols:
            try:
                df = self.feed.candles(symbol)
                if df.empty:
                    raise ValueError("no candles returned")
                if now - (df["timestamp"].iloc[-1] + self.tf) > 2 * self.tf:
                    data_ok = False
                    notes.append(f"{symbol}: market data is stale, new entries blocked")
                frames[symbol] = df
            except Exception as exc:
                data_ok = False
                notes.append(f"{symbol}: could not get data ({exc}); new entries blocked")

        cash = store.get("cash")
        last_close = dict(store.get("last_close", {}))

        # 2) Exits: replay every candle since the last check against each open position
        for pos in store.open_positions():
            df = frames.get(pos["symbol"])
            if df is None:
                continue
            fresh = df[df["timestamp"] > pd.Timestamp(pos["last_checked_ts"])]
            for row in fresh.itertuples():
                hit = check_exit(float(row.open), float(row.high), float(row.low), pos["stop"], pos["target"])
                if hit is None:
                    continue
                price, reason = hit
                fill = self.executor.sell(pos["symbol"], pos["qty"], price)
                proceeds = fill.qty * fill.price
                cash += proceeds - fill.fee
                pnl = proceeds - fill.fee - pos["entry_cost"]
                store.close_position(pos["id"], {
                    "symbol": pos["symbol"], "entry_time": pos["entry_time"],
                    "exit_time": row.timestamp.isoformat(), "entry": pos["entry_fill"], "exit": fill.price,
                    "qty": pos["qty"], "pnl": pnl, "return_pct": pnl / pos["entry_cost"] * 100,
                    "exit_reason": reason})
                notes.append(f"{pos['symbol']}: closed on {reason}, P&L {pnl:+.2f} USDT")
                break
            else:
                if not fresh.empty:
                    store.set_last_checked(pos["id"], df["timestamp"].iloc[-1].isoformat())

        # 3) Mark to market; daily P&L, peak equity, kill switch
        for symbol, df in frames.items():
            last_close[symbol] = float(df["close"].iloc[-1])
        positions = store.open_positions()
        equity = self._equity(cash, positions, last_close)
        candle_time = max((df["timestamp"].iloc[-1] for df in frames.values()), default=now)
        if candle_time.date().isoformat() != store.get("day"):
            store.set("day", candle_time.date().isoformat())
            store.set("day_start_equity", store.get("prev_equity"))
        day_start = store.get("day_start_equity")
        peak = max(store.get("peak"), equity)
        drawdown = (peak - equity) / peak * 100
        if drawdown >= cfg["skeptic"]["max_drawdown_pct"] and not store.get("halted"):
            store.set("halted", True)
            notes.append(f"KILL SWITCH: drawdown {drawdown:.1f}%. No new entries until --reset-halt.")
        store.set("peak", peak)
        store.set("prev_equity", equity)
        store.set("last_close", last_close)
        store.set("cash", cash)
        store.log_equity(now.isoformat(), equity, cash, len(positions))

        # 4) New signals (what counts as a signal depends on the strategy; see _new_signals)
        self._new_signals(frames, equity, cash, last_close, day_start, drawdown, data_ok, notes)
        return notes


# ---------------------------------------------------------------- command line

def _status(conn, cfg) -> None:
    store = Store(conn, float(cfg["backtest"]["starting_equity"]))
    positions = store.open_positions()
    last_close = store.get("last_close", {})
    cash = store.get("cash")
    equity = Chief._equity(cash, positions, last_close)
    start = float(cfg["backtest"]["starting_equity"])
    print(f"Equity {equity:,.2f} USDT ({(equity / start - 1) * 100:+.2f}% since start)  cash {cash:,.2f}  "
          f"peak {store.get('peak'):,.2f}  kill switch: {'ON' if store.get('halted') else 'off'}")
    print(f"Open positions: {len(positions)}")
    for p in positions:
        print(f"  {p['symbol']}: {p['qty']:.6f} @ {p['entry_fill']:,.2f}  stop {p['stop']:,.2f}  "
              f"target {'none (trend exit)' if p['target'] >= 1e11 else format(p['target'], ',.2f')}")
    trades = store.trades()
    print(f"Closed trades: {len(trades)}  total P&L {sum(t['pnl'] for t in trades):+.2f} USDT")
    bh = store.get("bh_start")
    if bh:
        ratio = sum(last_close.get(sym, px) / px for sym, px in bh.items()) / len(bh)
        print(f"Buy-and-hold since the first cycle (no fees): {(ratio - 1) * 100:+.2f}%  "
              f"| this bot: {(equity / start - 1) * 100:+.2f}%")
    print("Recent decisions:")
    for row in conn.execute("SELECT ts, symbol, verdict, reason FROM decisions ORDER BY id DESC LIMIT 5"):
        print(f"  {row[0][:16]}  {row[1]}  {row[2]}  {row[3]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Chief: run the trading pipeline (paper mode)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["paper"], default="paper",
                        help="only paper mode exists; real-money execution is deliberately not built")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true")
    group.add_argument("--loop", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--reset-halt", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    conn = init_db(cfg["logging"]["db_path"])
    if args.status:
        _status(conn, cfg)
        return 0
    if args.reset_halt:
        Store(conn, float(cfg["backtest"]["starting_equity"])).set("halted", False)
        print("Kill switch cleared. Review the recent decisions before letting it trade again.")
        return 0

    from .exchange import make_exchange
    from .feed import LiveFeed
    exchange = make_exchange(cfg)
    chief_class = Chief
    if cfg.get("strategy", {}).get("name") == "trend":
        from .trend_chief import TrendChief
        chief_class = TrendChief
    chief = chief_class(cfg, conn, LiveFeed(exchange, cfg["data"]["timeframe"]), PaperExecutor(cfg))
    tf_ms = timeframe_ms(cfg["data"]["timeframe"])

    def cycle():
        try:
            for note in chief.run_cycle(pd.Timestamp.now(tz="UTC")) or ["no action"]:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {note}", flush=True)
        except Exception as exc:  # never let one bad cycle kill the loop
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] cycle failed: {exc}", flush=True)

    if args.once:
        cycle()
        return 0
    print("Paper trading loop started. Ctrl+C to stop. State is saved, so you can restart any time.")
    try:
        while True:
            now_ms = int(time.time() * 1000)
            wake_ms = (now_ms // tf_ms + 1) * tf_ms + 15_000  # 15s after the candle closes
            time.sleep(max(1, (wake_ms - now_ms) / 1000))
            cycle()
    except KeyboardInterrupt:
        print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
