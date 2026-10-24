import numpy as np
import pandas as pd
import pytest

from bot.charts import add_indicators, all_proposals, latest_proposal
from bot.config import load_config
from bot.data import make_synthetic
from bot.indicators import atr, sma
from bot.scout import screen

CFG = load_config()
CFG["data"]["timeframe"] = "1h"  # these tests use 1-hour test data whatever config.yaml says


def v_shaped(down=80, up=80, start=200.0, step=1.0):
    """Price falls then rises: the fast average must cross above the slow one on the way up."""
    closes = np.concatenate([start - step * np.arange(down), start - step * down + step * np.arange(1, up + 1)])
    opens = np.concatenate([[start], closes[:-1]])
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    ts = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": np.full(len(closes), 10.0)})


def test_sma_matches_manual_average():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, 3)
    assert out.isna().sum() == 2
    assert out.iloc[2] == 2.0 and out.iloc[4] == 4.0


def test_atr_is_positive_and_needs_warmup():
    df = make_synthetic(n=200)
    out = atr(df, 14)
    assert out.iloc[:13].isna().all()
    assert (out.dropna() > 0).all()


def test_signal_fires_on_the_way_up_with_valid_proposal():
    df = v_shaped()
    signals = all_proposals(df, "BTC/USDT", CFG)
    assert len(signals) >= 1
    i, p = signals[0]
    assert df.loc[i, "close"] == p.entry
    assert p.stop < p.entry < p.target
    assert p.reward_risk == pytest.approx(CFG["strategy"]["reward_risk"])
    expected_stop = p.entry - CFG["strategy"]["atr_stop_mult"] * add_indicators(df, CFG["strategy"]).loc[i, "atr"]
    assert p.stop == pytest.approx(expected_stop)
    # timestamped at the candle's CLOSE (open + 1h), not its open
    assert p.created_at == (df.loc[i, "timestamp"] + pd.Timedelta(hours=1)).isoformat()


def test_no_signal_in_a_pure_downtrend():
    df = v_shaped(down=150, up=0)
    assert all_proposals(df, "BTC/USDT", CFG) == []


def test_no_signal_before_both_averages_exist():
    df = v_shaped()
    data = add_indicators(df, CFG["strategy"])
    first_valid = CFG["strategy"]["slow_ma"] - 1
    assert not data["cross_up"].iloc[:first_valid + 1].any()


def test_no_lookahead_signals_never_change_when_future_data_arrives():
    df = make_synthetic(n=1500, vol=0.02, seed=3)
    full = {p.created_at: p for _, p in all_proposals(df, "BTC/USDT", CFG)}
    assert len(full) > 3  # the test is meaningless without signals
    for cut in (400, 800, 1200):
        partial = all_proposals(df.iloc[:cut].copy(), "BTC/USDT", CFG)
        for _, p in partial:
            assert full[p.created_at] == p


def test_latest_proposal_only_on_the_signal_candle():
    df = v_shaped()
    i, p = all_proposals(df, "BTC/USDT", CFG)[0]
    assert latest_proposal(df.iloc[: i + 1].copy(), "BTC/USDT", CFG) == p
    assert latest_proposal(df.iloc[: i + 2].copy(), "BTC/USDT", CFG) is None
    assert latest_proposal(df.iloc[:0], "BTC/USDT", CFG) is None


def _scout_cfg(**overrides):
    cfg = {"scout": dict(CFG["scout"], lookback_candles=24, top_n=5)}
    cfg["scout"].update(overrides)
    return cfg


def test_scout_ranks_by_volatility_and_respects_filters():
    frames = {
        "CALM/USDT": make_synthetic(n=100, vol=0.002, seed=1),
        "WILD/USDT": make_synthetic(n=100, vol=0.03, seed=2),
        "SHORT/USDT": make_synthetic(n=10, vol=0.05, seed=3),  # too little history
    }
    result = screen(frames, _scout_cfg())
    assert [c.symbol for c in result] == ["WILD/USDT", "CALM/USDT"]

    assert [c.symbol for c in screen(frames, _scout_cfg(top_n=1))] == ["WILD/USDT"]
    assert [c.symbol for c in screen(frames, _scout_cfg(min_volatility_pct=1.0))] == ["WILD/USDT"]
    assert screen(frames, _scout_cfg(min_quote_volume=1e12)) == []
