import copy

import pandas as pd
import pytest

from bot.backtest import check_exit, run_backtest, split_frames
from bot.config import load_config
from bot.data import make_synthetic
from bot.models import Proposal
from bot.sizing import position_size
from bot.skeptic import AccountState, Skeptic, round_number_distance_pct

from test_step2_3 import v_shaped

CFG = load_config()
CFG["data"]["timeframe"] = "1h"  # these tests use 1-hour test data whatever config.yaml says
OK_STATE = AccountState(equity=1000.0, open_positions=0)


def good_proposal(**kw):
    base = dict(symbol="BTC/USDT", side="long", entry=100.0, stop=91.3, target=117.4,
                created_at="2026-01-01T00:00:00+00:00")
    base.update(kw)
    return Proposal(**base)


# ---------- Skeptic ----------

def test_skeptic_approves_a_clean_proposal():
    v = Skeptic(CFG).judge(good_proposal(), OK_STATE)
    assert v.approved and v.failed == ()


def test_each_rule_blocks_on_its_own():
    s = Skeptic(CFG)
    cases = {
        "reward_risk": (good_proposal(target=105.0), OK_STATE, None),
        "round_number_stop": (good_proposal(stop=90.0, target=118.0), OK_STATE, None),
        "swing_low_stop": (good_proposal(), OK_STATE, 91.35),
        "max_positions": (good_proposal(), AccountState(1000.0, open_positions=3), None),
        "daily_loss": (good_proposal(), AccountState(1000.0, 0, daily_pnl_pct=-3.5), None),
        "drawdown_kill": (good_proposal(), AccountState(1000.0, 0, drawdown_pct=10.5), None),
        "halted": (good_proposal(), AccountState(1000.0, 0, halted=True), None),
        "bad_data": (good_proposal(), AccountState(1000.0, 0, data_ok=False), None),
        "fees_vs_target": (good_proposal(stop=99.7, target=100.6), OK_STATE, None),
    }
    for rule, (proposal, state, low) in cases.items():
        v = s.judge(proposal, state, low)
        assert not v.approved, rule
        assert rule in v.failed, (rule, v.failed)


def test_skeptic_reports_every_failed_rule_not_just_the_first():
    v = Skeptic(CFG).judge(good_proposal(target=105.0), AccountState(1000.0, 3, daily_pnl_pct=-4.0))
    assert {"reward_risk", "max_positions", "daily_loss"} <= set(v.failed)
    assert v.reason.count(";") >= 2


def test_reward_risk_exactly_at_the_minimum_passes_despite_float_rounding():
    # Built the way Charts builds them, so reward/risk is 2.0 only up to rounding error.
    entry, atr = 2497.42, 8.4275
    stop = entry - CFG["strategy"]["atr_stop_mult"] * atr
    target = entry + CFG["strategy"]["reward_risk"] * (entry - stop)
    p = Proposal("ETH/USDT", "long", entry, stop, target, "2026-01-01T00:00:00+00:00")
    v = Skeptic(CFG).judge(p, OK_STATE)
    assert "reward_risk" not in v.failed


def test_every_proposal_charts_makes_passes_the_reward_risk_rule():
    from bot.charts import all_proposals
    skeptic = Skeptic(CFG)
    proposals = all_proposals(make_synthetic(n=6000, vol=0.02, seed=21), "BTC/USDT", CFG)
    assert len(proposals) > 20
    for _, p in proposals:
        assert "reward_risk" not in skeptic.judge(p, OK_STATE).failed


def test_round_number_distance():
    assert round_number_distance_pct(90.0) == 0.0
    assert round_number_distance_pct(2400.0) == 0.0
    assert round_number_distance_pct(81_000.0) == 0.0
    assert round_number_distance_pct(2416.53) == pytest.approx(0.6846, abs=1e-3)


# ---------- Sizing ----------

def test_sizing_uses_the_smallest_limit():
    # Wide stop: risk limit binds. 1% of 1000 = 10 USDT risk, 5 per unit -> 2 units.
    assert position_size(100.0, 95.0, 1000.0, 1000.0, CFG, cap_divisor=1) == pytest.approx(2.0)
    # Tight stop: risk limit would ask for 20 units (2000 USDT) -> concentration cap (1000/3) wins.
    assert position_size(100.0, 99.5, 1000.0, 1000.0, CFG, cap_divisor=3) == pytest.approx(1000 / 3 / 100)
    # Little cash left: cash limit wins and the cost (with fees) never exceeds it.
    qty = position_size(100.0, 95.0, 1000.0, 50.0, CFG, cap_divisor=1)
    assert qty * 100.0 * (1 + 0.0015) <= 50.0 + 1e-9
    assert position_size(100.0, 105.0, 1000.0, 1000.0, CFG, 1) == 0.0  # stop above entry


# ---------- Backtester ----------

def test_check_exit_stop_wins_when_a_candle_touches_both():
    assert check_exit(100, 125, 90, stop=95, target=120) == (95, "stop")
    assert check_exit(100, 125, 98, stop=95, target=120) == (120, "target")
    assert check_exit(100, 110, 98, stop=95, target=120) is None
    assert check_exit(94, 99, 90, stop=95, target=120) == (94, "stop_gap")   # gapped below the stop
    assert check_exit(121, 125, 119, stop=95, target=120) == (121, "target_gap")


def test_split_frames_covers_history_without_overlap():
    frames = {"A/USDT": make_synthetic(n=1000), "B/USDT": make_synthetic(n=1000, seed=1)}
    parts = split_frames(frames, 4)
    assert len(parts) == 4
    assert sum(len(p["A/USDT"]) for p in parts) == 1000
    stamps = pd.concat([p["A/USDT"]["timestamp"] for p in parts])
    assert stamps.is_unique and stamps.is_monotonic_increasing
    assert parts[1]["A/USDT"]["timestamp"].iloc[0] == parts[1]["B/USDT"]["timestamp"].iloc[0]


def test_money_is_conserved():
    """final equity == start + realized P&L + unrealized P&L, to the cent, with fees included."""
    frames = {"BTC/USDT": make_synthetic(n=3000, vol=0.02, seed=5),
              "ETH/USDT": make_synthetic(n=3000, vol=0.02, seed=6, start_price=50.0)}
    for use_skeptic in (False, True):
        r = run_backtest(frames, CFG, use_skeptic=use_skeptic)
        realized = sum(t["pnl"] for t in r.trades)
        unrealized = sum(o["unrealized_pnl"] for o in r.open_positions)
        assert len(r.trades) > 5
        assert r.final_equity == pytest.approx(r.start_equity + realized + unrealized, abs=1e-6)
        assert r.fees_paid > 0


def test_orders_fill_at_the_next_open_not_at_the_signal_price():
    df = v_shaped()
    r = run_backtest({"BTC/USDT": df}, CFG, use_skeptic=False)
    assert len(r.trades) == 1  # the V-shaped data produces exactly one trade
    first_decision = pd.Timestamp(r.decisions[0]["time"])          # = signal candle close time
    fill_row = df[df["timestamp"] == first_decision].iloc[0]       # = next candle (open time == signal close)
    slip = CFG["fees"]["slippage_pct"] / 100
    assert r.trades[0]["entry_time"] == first_decision
    assert r.trades[0]["entry"] == pytest.approx(fill_row["open"] * (1 + slip))
    assert r.trades[0]["entry"] != pytest.approx(r.decisions[0]["entry"])  # not the signal-candle close


def test_skeptic_that_blocks_everything_takes_no_trades():
    strict = copy.deepcopy(CFG)
    strict["skeptic"]["min_reward_risk"] = 99.0
    frames = {"BTC/USDT": make_synthetic(n=2000, vol=0.02, seed=7)}
    naive = run_backtest(frames, strict, use_skeptic=False)
    guarded = run_backtest(frames, strict, use_skeptic=True)
    assert len(naive.trades) > 0
    assert guarded.trades == [] and guarded.open_positions == []
    assert guarded.final_equity == guarded.start_equity
    assert all(not d["approved"] and "reward_risk" in d["failed"] for d in guarded.decisions)


def test_kill_switch_latches_and_stops_new_entries():
    harsh = copy.deepcopy(CFG)
    harsh["skeptic"]["max_drawdown_pct"] = 0.05  # trips almost immediately after any loss
    frames = {"BTC/USDT": make_synthetic(n=4000, vol=0.02, seed=9)}
    r = run_backtest(frames, harsh, use_skeptic=True)
    halted_blocks = [d for d in r.decisions if "halted" in d["failed"]]
    assert halted_blocks, "the kill switch never engaged"
    first = next(i for i, d in enumerate(r.decisions) if "halted" in d["failed"])
    assert all(not d["approved"] for d in r.decisions[first:])  # once on, always on


def test_decisions_are_written_to_the_log_database():
    from bot.decision_log import init_db
    conn = init_db(":memory:")
    frames = {"BTC/USDT": make_synthetic(n=1500, vol=0.02, seed=3)}
    r = run_backtest(frames, CFG, use_skeptic=True, db_conn=conn)
    n = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    assert n == len(r.decisions) > 0
