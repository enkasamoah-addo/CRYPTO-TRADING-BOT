from pathlib import Path

import urllib.error
import yaml

from bot import netcheck
from bot.config import load_config

ROOT = Path(__file__).resolve().parent.parent


def test_cloud_config_has_the_same_rules_as_the_tested_config():
    local, cloud = load_config(ROOT / "paper_trend.yaml"), load_config(ROOT / "paper_trend_cloud.yaml")
    for section in ("strategy", "research", "skeptic", "fees", "backtest", "exchange"):
        assert local[section] == cloud[section], section          # only the file locations may differ
    assert cloud["data"]["symbols"] == local["data"]["symbols"] and cloud["data"]["timeframe"] == "1d"
    assert cloud["logging"]["db_path"].startswith("state/") and cloud["data"]["dir"] == "state"


def test_workflow_runs_daily_checks_binance_first_and_saves_state():
    w = yaml.safe_load((ROOT / ".github/workflows/daily.yml").read_text())
    triggers = w.get("on") or w.get(True)
    assert triggers["schedule"][0]["cron"] == "10 0 * * *" and "workflow_dispatch" in triggers
    runs = [s.get("run", "") for s in w["jobs"]["cycle"]["steps"]]
    order = [next(i for i, r in enumerate(runs) if key in r)
             for key in ("bot.netcheck", "bot.chief", "bot.dashboard", "git push")]
    assert order == sorted(order)                                  # check, trade, dashboard, save
    assert "--once" in runs[order[1]] and "paper_trend_cloud.yaml" in runs[order[1]]


def test_netcheck_reports_blocked_locations_and_sets_the_exit_code(capsys):
    def blocked(url, timeout):
        raise urllib.error.HTTPError(url, 451, "blocked", {}, None)
    assert "451" in netcheck.probe("https://x", opener=blocked)
    assert netcheck.main(candles=lambda: "ok", prober=lambda u: "ok") == 0
    assert netcheck.main(candles=lambda: "failed (x)", prober=lambda u: "HTTP 451") == 1
    assert "cannot get prices" in capsys.readouterr().out
