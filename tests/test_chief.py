import copy

import pandas as pd
import pytest

from bot.backtest import run_backtest
from bot.chief import Chief
from bot.config import load_config
from bot.data import make_synthetic
from bot.decision_log import init_db
from bot.executor import PaperExecutor
from bot.state import Store

from test_step2_3 import v_shaped

CFG = load_config()
CFG["data"]["timeframe"] = "1h"  # these tests use 1-hour test data whatever config.yaml says
TF = pd.Timedelta(hours=1)
WINDOW = 300  # candles the live feed provides, as in LiveFeed.limit


class ReplayFeed:
    """Plays history back one candle at a time, like a live exchange would reveal it."""

    def __init__(self, frames):
        self.frames, self.k = frames, 0

    def candles(self, symbol):
        df = self.frames[symbol]
        return df.iloc[max(0, self.k - WINDOW + 1): self.k + 1].reset_index(drop=True)

    def price(self, symbol):
        # An order placed just after candle k closes fills around candle k+1's open.
        df = self.frames[symbol]
        return float(df["open"].iloc[self.k + 1]) if self.k + 1 < len(df) else float(df["close"].iloc[self.k])


def cfg_for(frames, base=CFG):
    """Config whose symbol list matches the test data (a missing symbol blocks all entries)."""
    cfg = copy.deepcopy(base)
    cfg["data"]["symbols"] = list(frames)
    return cfg


def make_chief(frames, cfg=CFG, conn=None):
    cfg = cfg_for(frames, cfg)
    conn = conn or init_db(":memory:")
    feed = ReplayFeed(frames)
    return Chief(cfg, conn, feed, PaperExecutor(cfg)), feed, conn


def replay(chief, feed, frames, start=0, stop=None):
    n = min(len(df) for df in frames.values())
    notes = []
    for k in range(start, stop if stop is not None else n):
        feed.k = k
        t = next(iter(frames.values()))["timestamp"].iloc[k]
        notes += chief.run_cycle(t + TF + pd.Timedelta(seconds=15))
    return notes


def compare_trades(chief_trades, bt_trades):
    assert len(chief_trades) == len(bt_trades) and len(bt_trades) > 3
    for c, b in zip(chief_trades, bt_trades):
        assert c["symbol"] == b["symbol"]
        assert pd.Timestamp(c["entry_time"]) == pd.Timestamp(b["entry_time"])
        assert pd.Timestamp(c["exit_time"]) == pd.Timestamp(b["exit_time"])
        assert c["exit_reason"] == b["exit_reason"]
        assert c["entry"] == pytest.approx(b["entry"], rel=1e-9)
        assert c["qty"] == pytest.approx(b["qty"], rel=1e-3)
        assert c["pnl"] == pytest.approx(b["pnl"], rel=1e-2, abs=0.05)


def two_symbol_frames(n=1000):
    return {"BTC/USDT": make_synthetic(n=n, vol=0.02, seed=5),
            "ETH/USDT": make_synthetic(n=n, vol=0.02, seed=6, start_price=50.0)}


def test_chief_replay_matches_the_backtester_trade_for_trade():
    """The live engine must behave like the thing we tested. Same data -> same trades."""
    frames = two_symbol_frames()
    chief, feed, _ = make_chief(frames)
    replay(chief, feed, frames)
    bt = run_backtest(frames, CFG, use_skeptic=True)
    compare_trades(chief.store.trades(), bt.trades)


def test_restart_in_the_middle_changes_nothing():
    """Stop the program, start a fresh one on the same database: results must be identical."""
    frames = two_symbol_frames()
    conn = init_db(":memory:")
    first, feed, _ = make_chief(frames, conn=conn)
    replay(first, feed, frames, 0, 500)
    second = Chief(cfg_for(frames), conn, feed, PaperExecutor(CFG))   # a brand-new process
    replay(second, feed, frames, 500)
    uninterrupted, feed2, _ = make_chief(frames)
    replay(uninterrupted, feed2, frames)
    compare_trades(second.store.trades(), uninterrupted.store.trades())
    assert second.store.get("cash") == pytest.approx(uninterrupted.store.get("cash"))


def test_the_same_candle_twice_never_places_a_second_order():
    df = v_shaped()
    frames = {"BTC/USDT": df}
    chief, feed, conn = make_chief(frames)
    from bot.charts import all_proposals
    i, _ = all_proposals(df, "BTC/USDT", CFG)[0]
    feed.k = i
    now = df["timestamp"].iloc[i] + TF + pd.Timedelta(seconds=15)
    first = chief.run_cycle(now)
    again = chief.run_cycle(now)
    again2 = chief.run_cycle(now)
    assert any("BOUGHT" in n for n in first)
    assert not any("BOUGHT" in n for n in again + again2)
    assert len(chief.store.open_positions()) == 1
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1


def test_money_is_conserved_live():
    frames = two_symbol_frames()
    chief, feed, _ = make_chief(frames)
    replay(chief, feed, frames)
    store = chief.store
    realized = sum(t["pnl"] for t in store.trades())
    positions = store.open_positions()
    unrealized = sum(p["qty"] * store.get("last_close")[p["symbol"]] - p["entry_cost"] for p in positions)
    equity = store.get("cash") + sum(p["qty"] * store.get("last_close")[p["symbol"]] for p in positions)
    assert len(store.trades()) > 3
    assert equity == pytest.approx(CFG["backtest"]["starting_equity"] + realized + unrealized, abs=1e-6)


def test_kill_switch_latches_blocks_entries_and_needs_a_manual_reset():
    harsh = copy.deepcopy(CFG)
    harsh["skeptic"]["max_drawdown_pct"] = 0.05
    frames = {"BTC/USDT": make_synthetic(n=1500, vol=0.02, seed=9)}
    chief, feed, conn = make_chief(frames, cfg=harsh)
    notes = replay(chief, feed, frames)
    assert chief.store.get("halted") is True
    assert any("KILL SWITCH" in n for n in notes)
    first_halt = next(i for i, n in enumerate(notes) if "KILL SWITCH" in n)
    assert not any("BOUGHT" in n for n in notes[first_halt + 1:])
    # a restart does not clear it
    assert Chief(cfg_for(frames, harsh), conn, feed, PaperExecutor(harsh)).store.get("halted") is True
    chief.store.set("halted", False)  # what --reset-halt does
    assert chief.store.get("halted") is False


def test_data_failure_never_crashes_and_never_trades():
    class BrokenFeed:
        def candles(self, symbol):
            raise ConnectionError("exchange unreachable")

        def price(self, symbol):
            raise ConnectionError("exchange unreachable")

    chief = Chief(CFG, init_db(":memory:"), BrokenFeed(), PaperExecutor(CFG))
    notes = chief.run_cycle(pd.Timestamp("2026-01-01", tz="UTC"))
    assert any("could not get data" in n for n in notes)
    assert chief.store.open_positions() == [] and chief.store.trades() == []


def test_stale_data_blocks_a_new_entry_and_says_why():
    df = v_shaped()
    frames = {"BTC/USDT": df}
    chief, feed, conn = make_chief(frames)
    from bot.charts import all_proposals
    i, _ = all_proposals(df, "BTC/USDT", CFG)[0]
    feed.k = i
    much_later = df["timestamp"].iloc[i] + pd.Timedelta(days=3)
    notes = chief.run_cycle(much_later)
    assert any("stale" in n for n in notes)
    assert not any("BOUGHT" in n for n in notes)
    reason = conn.execute("SELECT verdict, reason FROM decisions").fetchone()
    assert reason[0] == "blocked" and "stale" in reason[1]
