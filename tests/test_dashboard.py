import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from bot import dashboard
from bot.config import load_config
from bot.data import make_synthetic, save_csv
from bot.decision_log import init_db
from bot.executor import PaperExecutor
from bot.trend_chief import TrendChief

from test_trend_chief import DailyReplay, frames, TF

ROOT = Path(__file__).resolve().parent.parent


def cfg_for(tmp_path):
    cfg = load_config(ROOT / "paper_trend.yaml")
    cfg["logging"]["db_path"] = str(tmp_path / "paper.sqlite3")
    cfg["data"]["dir"] = str(tmp_path)
    return cfg


def run_bot(cfg, fr, cycles):
    conn = init_db(cfg["logging"]["db_path"])
    feed = DailyReplay(fr)
    chief = TrendChief(cfg, conn, feed, PaperExecutor(cfg))
    for k in range(cycles):
        feed.k = k
        chief.run_cycle(fr["BTC/USDT"]["timestamp"].iloc[k] + TF + pd.Timedelta(seconds=15))
    return chief, conn, feed


def test_collect_matches_the_bots_own_books(tmp_path):
    cfg = cfg_for(tmp_path)
    fr = frames(4, n=700)
    chief, conn, feed = run_bot(cfg, fr, 600)
    d = dashboard.collect(conn, cfg, {s: f.iloc[:600] for s, f in fr.items()})
    store = chief.store
    assert d["equity"] == pytest.approx(store.get("cash") + d["invested"])
    assert d["n_trades"] == len(store.trades()) > 3
    assert d["total_pnl"] == pytest.approx(sum(t["pnl"] for t in store.trades()))
    assert d["return_pct"] == pytest.approx((d["equity"] / 1000 - 1) * 100)
    assert 0 <= d["drawdown_pct"] <= 100
    for s, c in d["charts"].items():
        assert len(c["candles"]) == dashboard.CANDLES_SHOWN
        days = {x["time"] for x in c["candles"]}
        assert all(m["time"] in days for m in c["markers"])      # every marker lands on a drawn candle
    assert len(d["bot_curve"]) > 100 and len(d["hold_curve"]) > 100


def test_render_contains_the_key_facts_and_escapes_script_end(tmp_path):
    cfg = cfg_for(tmp_path)
    fr = frames(4, n=700)
    chief, conn, _ = run_bot(cfg, fr, 600)
    d = dashboard.collect(conn, cfg, {s: f.iloc[:600] for s, f in fr.items()})
    html = dashboard.render(d)
    assert "Simulated trading" in html and "Closed trades" in html and "lightweight-charts@4.1.3" in html
    assert f"{d['equity']:,.2f}" in html
    d["halted"] = True
    assert "KILL SWITCH ON" in dashboard.render(d)
    d["charts"]["BTC/USDT"]["markers"].append({"time": "2020-01-01", "kind": "buy", "text": "</script><b>x"})
    assert "</script><b>x" not in dashboard.render(d)


def test_empty_bot_renders_without_errors(tmp_path):
    cfg = cfg_for(tmp_path)
    conn = init_db(cfg["logging"]["db_path"])
    from bot.state import Store
    Store(conn, 1000.0)
    html = dashboard.render(dashboard.collect(conn, cfg, {}))
    assert "No open positions" in html and "No closed trades yet" in html


def test_main_is_read_only_and_works_offline(tmp_path, capsys):
    cfg = cfg_for(tmp_path)
    fr = frames(4, n=700)
    run_bot(cfg, fr, 600)
    for s, f in fr.items():
        save_csv(f.iloc[:600], tmp_path, s, "1d")
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    db = Path(cfg["logging"]["db_path"])
    before = db.read_bytes()
    assert dashboard.main(["--config", str(p), "--offline"]) == 0
    assert db.read_bytes() == before                           # the database was not touched
    assert (tmp_path / "dashboard.html").exists()
    ro = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("DELETE FROM trades")
    cfg["logging"]["db_path"] = str(tmp_path / "missing.sqlite3")
    p.write_text(yaml.safe_dump(cfg))
    assert dashboard.main(["--config", str(p), "--offline"]) == 1
