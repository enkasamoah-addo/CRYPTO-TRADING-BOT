import pandas as pd
import pytest

from bot.config import load_config
from bot.data import (check_quality, fetch_ohlcv_history, load_csv, make_synthetic,
                      save_csv, timeframe_ms)
from bot.decision_log import init_db, log_decision
from bot.models import Proposal

HOUR = 3_600_000


class FakeExchange:
    """Serves candles like a real exchange: from `since`, at most `limit` per call."""

    def __init__(self, df):
        self.rows = [[int(r.timestamp.timestamp() * 1000), r.open, r.high, r.low, r.close, r.volume]
                     for r in df.itertuples()]
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls += 1
        out = [r for r in self.rows if since is None or r[0] >= since]
        return out[:limit]


def test_timeframe_ms():
    assert timeframe_ms("1h") == HOUR
    assert timeframe_ms("15m") == 900_000
    with pytest.raises(ValueError):
        timeframe_ms("1x")


def test_pagination_returns_every_candle_once():
    df = make_synthetic(n=2500, timeframe="1h")
    ex = FakeExchange(df)
    since = int(df["timestamp"].iloc[0].timestamp() * 1000)
    far_future = int(df["timestamp"].iloc[-1].timestamp() * 1000) + 10 * HOUR
    out = fetch_ohlcv_history(ex, "BTC/USDT", "1h", since, now_ms=far_future, limit=1000)
    assert len(out) == 2500
    assert ex.calls == 3  # 1000 + 1000 + 500
    assert out["timestamp"].is_monotonic_increasing and out["timestamp"].is_unique
    assert check_quality(out, "1h")["ok"]


def test_forming_candle_is_dropped():
    df = make_synthetic(n=100, timeframe="1h")
    ex = FakeExchange(df)
    since = int(df["timestamp"].iloc[0].timestamp() * 1000)
    last_open = int(df["timestamp"].iloc[-1].timestamp() * 1000)
    # "now" is 30 minutes into the final candle, so it is not complete yet
    out = fetch_ohlcv_history(ex, "BTC/USDT", "1h", since, now_ms=last_open + HOUR // 2)
    assert len(out) == 99


def test_quality_check_finds_gaps_and_bad_rows():
    df = make_synthetic(n=50)
    assert check_quality(df, "1h")["ok"]

    gapped = df.drop(index=[20, 21]).reset_index(drop=True)
    report = check_quality(gapped, "1h")
    assert not report["ok"] and len(report["gaps"]) == 1

    bad = df.copy()
    bad.loc[10, "high"] = bad.loc[10, "low"] - 1  # high below low
    assert check_quality(bad, "1h")["bad_ohlc_rows"] == 1

    nan = df.copy()
    nan.loc[5, "close"] = float("nan")
    assert check_quality(nan, "1h")["nan_rows"] == 1


def test_csv_roundtrip(tmp_path):
    df = make_synthetic(n=30)
    save_csv(df, tmp_path, "BTC/USDT", "1h")
    back = load_csv(tmp_path, "BTC/USDT", "1h")
    assert (tmp_path / "BTC-USDT_1h.csv").exists()
    pd.testing.assert_frame_equal(df, back, check_dtype=False, check_freq=False)


def test_proposal_math_and_validation():
    p = Proposal("BTC/USDT", "long", entry=100.0, stop=95.0, target=110.0,
                 created_at="2026-10-19T12:00:00+00:00")
    assert p.risk == 5.0 and p.reward == 10.0 and p.reward_risk == 2.0
    with pytest.raises(ValueError):
        Proposal("BTC/USDT", "long", entry=100.0, stop=105.0, target=110.0, created_at="x")
    with pytest.raises(ValueError):
        Proposal("BTC/USDT", "short", entry=100.0, stop=105.0, target=90.0, created_at="x")


def test_decision_log_records_verdicts():
    conn = init_db(":memory:")
    p = Proposal("ETH/USDT", "long", 100.0, 95.0, 110.0, "2026-10-19T12:00:00+00:00")
    log_decision(conn, p, "blocked", "reward-to-risk below minimum")
    log_decision(conn, p, "approved", "all rules passed")
    rows = conn.execute("SELECT verdict, reason FROM decisions ORDER BY id").fetchall()
    assert rows == [("blocked", "reward-to-risk below minimum"), ("approved", "all rules passed")]
    with pytest.raises(Exception):
        conn.execute("INSERT INTO decisions (ts,symbol,side,entry,stop,target,verdict,reason) "
                     "VALUES ('t','s','long',1,0.5,2,'maybe','r')")


def test_config_loads_with_expected_sections():
    cfg = load_config()
    for key in ("exchange", "data", "strategy", "skeptic", "fees", "logging"):
        assert key in cfg
    assert cfg["strategy"]["fast_ma"] < cfg["strategy"]["slow_ma"]
    assert cfg["skeptic"]["min_reward_risk"] <= cfg["strategy"]["reward_risk"]
