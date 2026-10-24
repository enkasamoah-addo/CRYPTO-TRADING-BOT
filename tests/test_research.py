import json

import numpy as np
import pandas as pd
import pytest

from bot import research
from bot.config import load_config
from bot.data import make_synthetic, save_csv

def cfg():
    from pathlib import Path
    return load_config(Path(__file__).resolve().parent.parent / "research.yaml")


def frames(n=2200, seed=1):
    return {"BTC/USDT": make_synthetic(n=n, timeframe="1d", vol=0.03, seed=seed, start="2018-01-01"),
            "ETH/USDT": make_synthetic(n=n, timeframe="1d", vol=0.035, seed=seed + 1, start="2018-01-01",
                                       start_price=50.0)}


@pytest.mark.parametrize("name", research.CANDIDATES)
def test_money_is_conserved_including_open_positions(name):
    for seed in (1, 4, 7):
        r = research.simulate(frames(seed=seed), cfg(), name)
        realized = sum(t["pnl"] for t in r["trades"])
        assert r["n_trades"] > 5
        assert r["final_equity"] == pytest.approx(r["start_equity"] + realized + r["unrealized"], abs=1e-6)
        assert r["fees_paid"] > 0


def test_signals_never_change_when_future_data_arrives():
    df = make_synthetic(n=900, timeframe="1d", vol=0.03, seed=3, start="2018-01-01")
    full = research.add_signals(df, cfg()["research"])
    part = research.add_signals(df.iloc[:500].copy(), cfg()["research"])
    for col in ["breakout_in", "breakout_out", "trend_in", "trend_out", "crossover_in", "crossover_out", "atr"]:
        pd.testing.assert_series_equal(full[col].iloc[:500].reset_index(drop=True),
                                       part[col].reset_index(drop=True), check_names=False)


def _flat_then_jump():
    # 60 flat days, then a strong rise: every candidate must enter, filling at the NEXT open.
    closes = np.concatenate([np.full(120, 100.0), 100 + 2.0 * np.arange(1, 81)])
    opens = np.concatenate([[100.0], closes[:-1]])
    ts = pd.date_range("2020-01-01", periods=len(closes), freq="1D", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": opens, "high": np.maximum(opens, closes) + 0.5,
                         "low": np.minimum(opens, closes) - 0.5, "close": closes, "volume": 1.0})


def test_entry_fills_at_next_open_with_slippage():
    c = cfg()
    df = _flat_then_jump()
    r = research.simulate({"BTC/USDT": df}, c, "breakout")
    sig = research.add_signals(df, c["research"])
    i = int(sig.index[sig["breakout_in"]][0])        # first signal candle
    assert r["curve"].index[0] == df["timestamp"].iloc[0]
    # the position opens on candle i+1 at its open (+ slippage)
    state = research.simulate({"BTC/USDT": df.iloc[: i + 2].copy()}, c, "breakout")
    assert state["open_positions"] == 1
    held = state["curve"].iloc[-1]
    assert held != state["start_equity"]           # money moved into the position
    one_before = research.simulate({"BTC/USDT": df.iloc[: i + 1].copy()}, c, "breakout")
    assert one_before["open_positions"] == 0       # not filled on the signal candle itself


def test_stop_wins_and_gaps_fill_at_the_open():
    c = cfg()
    df = _flat_then_jump()
    last = len(df)
    # crash: gap far below any stop
    crash = df.copy()
    crash.loc[last - 1, ["open", "high", "low", "close"]] = [20.0, 21.0, 19.0, 20.0]
    r = research.simulate({"BTC/USDT": crash}, c, "trend")
    stops = [t for t in r["trades"] if t["reason"] == "stop"]
    assert stops and stops[-1]["exit"] == pytest.approx(20.0 * (1 - c["fees"]["slippage_pct"] / 100))
    assert stops[-1]["pnl"] < 0


def test_pass_test_requires_every_check():
    rc = cfg()["research"]
    bh = {"return_pct": 50.0, "max_dd_pct": 60.0}
    good = {"return_pct": 40.0, "max_dd_pct": 30.0, "n_trades": 40}
    assert research.judge(good, bh, rc)["passed"]
    for bad in ({"n_trades": 29}, {"return_pct": -1.0}, {"max_dd_pct": 45.0},
                {"return_pct": 10.0, "max_dd_pct": 39.0}):
        assert not research.judge({**good, **bad}, bh, rc)["passed"], bad


def test_final_exam_runs_once_only(tmp_path, monkeypatch):
    c = cfg()
    c["data"]["dir"] = str(tmp_path)
    f = frames(n=2300)
    for s, df in f.items():
        save_csv(df, tmp_path, s, "1d")
    # dev cut falls inside the synthetic history
    c["research"]["dev_end"] = "2022-06-01"
    # We are testing the one-shot guard, not the strategies: let every candidate pass.
    monkeypatch.setattr(research, "judge", lambda res, bh, rc: {"checks": {"stub": True}, "passed": True})
    import yaml
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(c))
    assert research.main(["final", "--config", str(p)]) == 1      # dev not run yet
    assert research.main(["dev", "--config", str(p)]) == 0
    dev = json.loads((tmp_path / "research_dev_result.json").read_text())
    assert any(v["passed"] for v in dev.values())
    assert research.main(["final", "--config", str(p)]) == 0
    assert (tmp_path / "research_final_DONE.json").exists()
    assert research.main(["final", "--config", str(p)]) == 1      # second attempt refused
