import copy
from pathlib import Path

import pandas as pd
import pytest

from bot import research
from bot.chief import Chief
from bot.config import load_config
from bot.data import make_synthetic
from bot.decision_log import init_db
from bot.executor import PaperExecutor
from bot.skeptic import AccountState, Skeptic
from bot.trend_chief import TrendChief

ROOT = Path(__file__).resolve().parent.parent
TF = pd.Timedelta(days=1)


class DailyReplay:
    def __init__(self, frames):
        self.frames, self.k = frames, 0

    def candles(self, symbol):
        df = self.frames[symbol]
        return df.iloc[max(0, self.k - 299): self.k + 1].reset_index(drop=True)

    def price(self, symbol):
        df = self.frames[symbol]
        return float(df["open"].iloc[self.k + 1]) if self.k + 1 < len(df) else float(df["close"].iloc[self.k])


def frames(seed=1, n=1500):
    return {"BTC/USDT": make_synthetic(n=n, timeframe="1d", vol=0.03, seed=seed, start="2018-01-01"),
            "ETH/USDT": make_synthetic(n=n, timeframe="1d", vol=0.035, seed=seed + 1, start="2018-01-01",
                                       start_price=50.0)}


def paper_cfg():
    return load_config(ROOT / "paper_trend.yaml")


def run(fr, cfg, stop=None):
    conn = init_db(":memory:")
    feed = DailyReplay(fr)
    chief = TrendChief(cfg, conn, feed, PaperExecutor(cfg))
    n = min(len(d) for d in fr.values())
    for k in range(stop or n):
        feed.k = k
        t = fr["BTC/USDT"]["timestamp"].iloc[k]
        chief.run_cycle(t + TF + pd.Timedelta(seconds=15))
    return chief, conn


@pytest.mark.parametrize("seed", [1, 4])
def test_paper_trend_chief_matches_the_researched_backtest(seed):
    fr = frames(seed)
    cfg = paper_cfg()
    cfg["skeptic"]["max_daily_loss_pct"] = 100.0     # the backtest has no such rule; compare like with like
    cfg["skeptic"]["max_drawdown_pct"] = 100.0
    chief, _ = run(fr, cfg)
    bt = research.simulate(fr, cfg, "trend")
    done = chief.store.trades()
    # Both sides leave positions open at the end; the researched backtest's CLOSED trades must match.
    assert len(bt["trades"]) > 5
    # The first cycle starts once 100+ days exist in the backtest too, so compare after warm-up.
    assert len(done) == len(bt["trades"])
    for c, b in zip(done, bt["trades"]):
        assert c["symbol"] == b["symbol"]
        assert pd.Timestamp(c["entry_time"]) == pd.Timestamp(b["entry_time"])
        assert pd.Timestamp(c["exit_time"]) == pd.Timestamp(b["exit_time"])
        assert c["entry"] == pytest.approx(b["entry"], rel=1e-9)
        assert c["exit"] == pytest.approx(b["exit"], rel=1e-9)
        assert c["qty"] == pytest.approx(b["qty"], rel=1e-3)


def test_a_restart_never_repeats_an_order():
    fr = frames(2)
    cfg = paper_cfg()
    chief, conn = run(fr, cfg, stop=700)
    before = (len(chief.store.trades()), len(chief.store.open_positions()), chief.store.get("cash"))
    chief.run_cycle(fr["BTC/USDT"]["timestamp"].iloc[699] + TF + pd.Timedelta(seconds=30))  # same candle again
    after = (len(chief.store.trades()), len(chief.store.open_positions()), chief.store.get("cash"))
    assert before == after


def test_disabled_rules_only_switch_off_the_named_rules():
    cfg = paper_cfg()
    from bot.models import Proposal
    p = Proposal("BTC/USDT", "long", 100.0, 90.0, 100000.0, "2026-01-01T00:00:00+00:00")
    ok = AccountState(equity=1000.0, open_positions=0)
    assert Skeptic(cfg).judge(p, ok).approved
    assert "halted" in Skeptic(cfg).judge(p, AccountState(1000.0, 0, halted=True)).failed
    assert "max_positions" in Skeptic(cfg).judge(p, AccountState(1000.0, 3)).failed
    assert "daily_loss" in Skeptic(cfg).judge(p, AccountState(1000.0, 0, daily_pnl_pct=-5.0)).failed
    assert "bad_data" in Skeptic(cfg).judge(p, AccountState(1000.0, 0, data_ok=False)).failed
    # Switching a rule off must matter: with a rule that WOULD veto this proposal (stop 90.0 is a
    # round number), the proposal is blocked when the rule is on and approved only when it is named off.
    strict = copy.deepcopy(cfg)
    strict["skeptic"].update(reject_stop_on_round_number=True, round_number_tolerance_pct=0.15, disabled_rules=[])
    assert "round_number_stop" in Skeptic(strict).judge(p, ok).failed
    strict["skeptic"]["disabled_rules"] = ["round_number_stop"]
    assert Skeptic(strict).judge(p, ok).approved


def test_kill_switch_blocks_new_trend_entries():
    fr = frames(3)
    cfg = paper_cfg()
    cfg["skeptic"]["max_drawdown_pct"] = 0.5
    chief, conn = run(fr, cfg)
    blocked = conn.execute("SELECT COUNT(*) FROM decisions WHERE verdict='blocked'").fetchone()[0]
    assert chief.store.get("halted") and blocked > 0
