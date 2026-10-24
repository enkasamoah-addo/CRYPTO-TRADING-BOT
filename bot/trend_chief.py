"""Chief variant that runs the daily trend filter (candidate B from the research).

Rules, identical to bot/research.py (the tested version):
  - After each daily close: if flat and close > 100-day average, buy at the next open.
  - If holding and close < 100-day average, sell at the next open.
  - Protective stop 2 x ATR below the fill, checked on every candle (Chief's normal exit step).
  - Size: 1% of equity at risk, capped at equity / number of symbols.
The Skeptic still guards entries (kill switch, daily loss, max positions, stale data). Rules that
only make sense for a profit-target strategy are switched off in config: skeptic.disabled_rules.
Paper mode only.
"""
import pandas as pd

from .chief import Chief
from .decision_log import log_decision
from .models import Proposal
from .research import add_signals
from .sizing import position_size
from .skeptic import AccountState

NO_TARGET = 1000.0   # target = entry * 1000 stands for "no profit target"; never reachable


class TrendChief(Chief):
    def _new_signals(self, frames, equity, cash, last_close, day_start, drawdown, data_ok, notes):
        cfg, store, rcfg = self.cfg, self.store, self.cfg["research"]
        if store.get("bh_start") is None and frames:
            store.set("bh_start", dict(last_close))   # reference for the buy-and-hold comparison
        for symbol, df in frames.items():
            if len(df) < rcfg["trend_days"]:      # same warm-up as the researched backtest
                notes.append(f"{symbol}: not enough history yet")
                continue
            sig = add_signals(df, rcfg).iloc[-1]
            if pd.isna(sig["atr"]):
                continue
            ts = df["timestamp"].iloc[-1]
            close_time = (ts + self.tf).isoformat()
            holding = next((p for p in store.open_positions() if p["symbol"] == symbol), None)

            if holding is not None:
                key = f"trend_out@{ts.isoformat()}"
                if bool(sig["trend_out"]) and not store.signal_seen(symbol, key):
                    store.mark_signal(symbol, key)
                    fill = self.executor.sell(symbol, holding["qty"], self.feed.price(symbol))
                    proceeds = fill.qty * fill.price
                    cash += proceeds - fill.fee
                    pnl = proceeds - fill.fee - holding["entry_cost"]
                    store.close_position(holding["id"], {
                        "symbol": symbol, "entry_time": holding["entry_time"], "exit_time": close_time,
                        "entry": holding["entry_fill"], "exit": fill.price, "qty": holding["qty"], "pnl": pnl,
                        "return_pct": pnl / holding["entry_cost"] * 100, "exit_reason": "trend_exit"})
                    store.set("cash", cash)
                    notes.append(f"{symbol}: SOLD, price closed below its {rcfg['trend_days']}-day average. "
                                 f"P&L {pnl:+.2f} USDT")
                continue

            key = f"trend_in@{ts.isoformat()}"
            if not bool(sig["trend_in"]) or store.signal_seen(symbol, key):
                continue
            store.mark_signal(symbol, key)
            price = self.feed.price(symbol)
            ref = self.executor.expected_buy_price(price)
            stop = ref - rcfg["atr_stop_mult"] * float(sig["atr"])
            if not stop < ref:
                continue
            proposal = Proposal(symbol, "long", ref, stop, ref * NO_TARGET, close_time)
            positions = store.open_positions()
            state = AccountState(
                equity=equity, open_positions=len(positions),
                daily_pnl_pct=(equity - day_start) / day_start * 100, drawdown_pct=drawdown,
                halted=bool(store.get("halted")), data_ok=data_ok)
            verdict = self.skeptic.judge(proposal, state)
            log_decision(self.conn, proposal, "approved" if verdict.approved else "blocked", verdict.reason)
            if not verdict.approved:
                notes.append(f"{symbol}: trend entry BLOCKED - {verdict.reason}")
                continue
            mark = self._equity(cash, positions, last_close)
            qty = position_size(ref, stop, mark, cash, cfg, cap_divisor=len(self.symbols))
            if qty <= 0 or qty * ref < 1.0:
                notes.append(f"{symbol}: approved but position size is too small, skipped")
                continue
            fill = self.executor.buy(symbol, qty, price)
            cash -= fill.qty * fill.price + fill.fee
            store.set("cash", cash)
            store.add_position(symbol, fill.qty, fill.price, stop, proposal.target, close_time,
                               fill.qty * fill.price + fill.fee, ts.isoformat())
            notes.append(f"{symbol}: BOUGHT {fill.qty:.6f} at {fill.price:,.2f} "
                         f"(trend is up; protective stop {stop:,.2f}, no profit target)")
        return cash
