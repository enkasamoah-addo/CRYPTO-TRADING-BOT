"""Pre-registered strategy research on daily candles.

Three candidates, fixed rules, one development test, one final exam.
    python -m bot.research dev   --config research.yaml
    python -m bot.research final --config research.yaml     (can only be run once)

Realism: signal at the daily close, fill at the NEXT open, fees and slippage on both
sides, protective stop checked on every candle (stop wins if unsure), no leverage,
no lookahead. Money is conserved: equity = cash + open positions at market value.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import load_config
from .data import csv_path, load_csv
from .indicators import atr, sma
from .sizing import position_size

# Listed from simplest to most complicated. If several survive, the earliest one wins.
CANDIDATES = ["trend", "breakout", "crossover"]
LABELS = {"trend": "B. Trend filter (price above 100-day average)",
          "breakout": "A. Breakout (20-day high in, 10-day low out)",
          "crossover": "C. Crossover control (20/50-day averages)"}


def add_signals(df: pd.DataFrame, rcfg: dict) -> pd.DataFrame:
    """Add entry/exit flags for every candidate. Row i uses only rows up to i."""
    d = df.copy()
    d["atr"] = atr(d, rcfg["atr_period"])
    # A: breakout
    hi = d["high"].shift(1).rolling(rcfg["breakout_entry_days"]).max()
    lo = d["low"].shift(1).rolling(rcfg["breakout_exit_days"]).min()
    d["breakout_in"] = d["close"] > hi
    d["breakout_out"] = d["close"] < lo
    # B: trend filter
    ma = sma(d["close"], rcfg["trend_days"])
    d["trend_in"] = d["close"] > ma
    d["trend_out"] = d["close"] < ma
    # C: crossover
    fast, slow = sma(d["close"], rcfg["cross_fast"]), sma(d["close"], rcfg["cross_slow"])
    d["crossover_in"] = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    d["crossover_out"] = fast < slow
    return d


def simulate(frames: Dict[str, pd.DataFrame], cfg: dict, strategy: str,
             start: Optional[str] = None, end: Optional[str] = None) -> dict:
    """Run one strategy over [start, end). Indicators may look back before `start`."""
    rcfg, fee, slip = cfg["research"], cfg["fees"]["taker_pct"] / 100, cfg["fees"]["slippage_pct"] / 100
    data = {s: add_signals(f, rcfg).reset_index(drop=True) for s, f in frames.items()}
    t0 = pd.Timestamp(start, tz="UTC") if start else None
    t1 = pd.Timestamp(end, tz="UTC") if end else None
    days = sorted(set().union(*[set(d["timestamp"]) for d in data.values()]))
    days = [t for t in days if (t0 is None or t >= t0) and (t1 is None or t < t1)]
    index = {s: {t: i for i, t in enumerate(d["timestamp"])} for s, d in data.items()}

    cash = float(cfg["backtest"]["starting_equity"])
    start_equity = cash
    pos: Dict[str, dict] = {}        # symbol -> {qty, entry, stop, entry_time}
    pending: Dict[str, dict] = {}    # symbol -> {"action": "enter"/"exit", "atr": ...}
    trades: List[dict] = []
    curve = []
    fees_paid = 0.0
    n_syms = len(data)

    def equity_at(mark):  # mark: symbol -> price
        return cash + sum(p["qty"] * mark[s] for s, p in pos.items())

    last_close = {s: float(d["close"].iloc[0]) for s, d in data.items()}
    for t in days:
        # 1) orders decided at yesterday's close fill at today's open
        for s in list(pending):
            i = index[s].get(t)
            if i is None:
                continue
            row, order = data[s].iloc[i], pending.pop(s)
            if order["action"] == "enter" and s not in pos:
                fill = float(row["open"]) * (1 + slip)
                stop = fill - rcfg["atr_stop_mult"] * order["atr"]
                mark = {k: last_close[k] for k in data}
                eq = equity_at(mark)
                qty = position_size(fill, stop, eq, cash, cfg, cap_divisor=n_syms)
                if qty > 0:
                    cost = qty * fill
                    f = cost * fee
                    cash -= cost + f
                    fees_paid += f
                    pos[s] = {"qty": qty, "entry": fill, "stop": stop, "entry_time": t, "fees": f}
            elif order["action"] == "exit" and s in pos:
                cash, fees_paid = _close(pos, s, float(row["open"]) * (1 - slip), t, "signal",
                                         trades, cash, fees_paid, fee)
        # 2) protective stops, checked on every candle (also the candle we entered on)
        for s in list(pos):
            i = index[s].get(t)
            if i is None:
                continue
            row = data[s].iloc[i]
            if float(row["low"]) <= pos[s]["stop"]:
                px = min(float(row["open"]), pos[s]["stop"]) * (1 - slip)   # gap below stop fills at the open
                cash, fees_paid = _close(pos, s, px, t, "stop", trades, cash, fees_paid, fee)
                pending.pop(s, None)
        # 3) at the close: record equity, then decide tomorrow's orders
        for s, d in data.items():
            i = index[s].get(t)
            if i is not None:
                last_close[s] = float(d["close"].iloc[i])
        curve.append((t, equity_at(last_close)))
        for s, d in data.items():
            i = index[s].get(t)
            if i is None or pd.isna(d["atr"].iloc[i]):
                continue
            row = d.iloc[i]
            if s in pos and bool(row[f"{strategy}_out"]):
                pending[s] = {"action": "exit"}
            elif s not in pos and bool(row[f"{strategy}_in"]):
                pending[s] = {"action": "enter", "atr": float(row["atr"])}

    unreal = sum(p["qty"] * (last_close[s] - p["entry"]) - p["fees"] for s, p in pos.items())
    final = curve[-1][1] if curve else start_equity
    eq = pd.Series([v for _, v in curve], index=[t for t, _ in curve])
    return {"strategy": strategy, "start_equity": start_equity, "final_equity": final,
            "trades": trades, "open_positions": len(pos), "fees_paid": fees_paid,
            "unrealized": unreal, "curve": eq, **_stats(eq, trades, start_equity)}


def _close(pos, s, px, t, why, trades, cash, fees_paid, fee):
    p = pos.pop(s)
    proceeds = p["qty"] * px
    f = proceeds * fee
    pnl = proceeds - f - p["qty"] * p["entry"] - p["fees"]
    trades.append({"symbol": s, "entry_time": p["entry_time"], "exit_time": t, "entry": p["entry"],
                   "exit": px, "qty": p["qty"], "pnl": pnl, "reason": why})
    return cash + proceeds - f, fees_paid + f


def _stats(eq: pd.Series, trades: List[dict], start_equity: float) -> dict:
    if eq.empty:
        return {"return_pct": 0.0, "max_dd_pct": 0.0, "n_trades": 0, "win_rate": 0.0, "pf": 0.0}
    peak = eq.cummax()
    max_dd = float(((peak - eq) / peak).max() * 100)
    ret = (float(eq.iloc[-1]) / start_equity - 1) * 100
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [-t["pnl"] for t in trades if t["pnl"] <= 0]
    return {"return_pct": ret, "max_dd_pct": max_dd, "n_trades": len(trades),
            "win_rate": 100 * len(wins) / len(trades) if trades else 0.0,
            "pf": sum(wins) / sum(losses) if losses and sum(losses) > 0 else (float("inf") if wins else 0.0)}


def buy_and_hold(frames, cfg, start=None, end=None) -> dict:
    """Equal split across the symbols, bought at the first open (fees and slippage paid), held."""
    fee, slip = cfg["fees"]["taker_pct"] / 100, cfg["fees"]["slippage_pct"] / 100
    t0 = pd.Timestamp(start, tz="UTC") if start else None
    t1 = pd.Timestamp(end, tz="UTC") if end else None
    eq0 = float(cfg["backtest"]["starting_equity"])
    parts = []
    for s, f in frames.items():
        d = f if t0 is None else f[f["timestamp"] >= t0]
        d = d if t1 is None else d[d["timestamp"] < t1]
        px = float(d["open"].iloc[0]) * (1 + slip)
        qty = eq0 / len(frames) / (px * (1 + fee))
        parts.append(pd.Series((qty * d["close"]).values, index=d["timestamp"].values))
    eq = sum(parts).dropna()
    eq.index = pd.to_datetime(eq.index, utc=True)
    return {"curve": eq, "start_equity": eq0, **_stats(eq, [], eq0)}


def judge(res: dict, bh: dict, rcfg: dict) -> dict:
    """The fixed pass test. Every check must pass; there is no 'close enough'."""
    ratio = lambda x: x["return_pct"] / x["max_dd_pct"] if x["max_dd_pct"] > 0 else float("inf")
    checks = {
        f"at least {rcfg['min_trades']} closed trades": res["n_trades"] >= rcfg["min_trades"],
        "net return above zero": res["return_pct"] > 0,
        "max drawdown at most two thirds of buy-and-hold's":
            res["max_dd_pct"] <= rcfg["max_drawdown_vs_buyhold"] * bh["max_dd_pct"],
        "better return-to-drawdown than buy-and-hold": ratio(res) > ratio(bh),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _load(cfg) -> Dict[str, pd.DataFrame]:
    d = cfg["data"]
    frames = {}
    for s in d["symbols"]:
        if not csv_path(d["dir"], s, d["timeframe"]).exists():
            raise SystemExit(f"Missing data for {s}. Run: python -m bot.download --config research.yaml")
        frames[s] = load_csv(d["dir"], s, d["timeframe"])
    return frames


def _report(title, window, frames, cfg, start, end, only=None):
    bh = buy_and_hold(frames, cfg, start, end)
    print(f"\n=== {title}: {window} ===")
    print(f"Buy-and-hold ({'/'.join(frames)} equal split): return {bh['return_pct']:+.1f}%, "
          f"worst drawdown {bh['max_dd_pct']:.1f}%\n")
    out = {}
    for name in (only or CANDIDATES):
        r = simulate(frames, cfg, name, start, end)
        j = judge(r, bh, cfg["research"])
        out[name] = {"passed": j["passed"], "return_pct": r["return_pct"], "max_dd_pct": r["max_dd_pct"],
                     "n_trades": r["n_trades"], "win_rate": r["win_rate"], "pf": r["pf"]}
        print(f"{LABELS[name]}")
        print(f"  return {r['return_pct']:+.1f}%  worst drawdown {r['max_dd_pct']:.1f}%  trades {r['n_trades']}  "
              f"win rate {r['win_rate']:.0f}%  profit factor {r['pf']:.2f}  fees paid {r['fees_paid']:.1f}")
        for check, ok in j["checks"].items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {check}")
        print(f"  => {'PASSED' if j['passed'] else 'did not pass'}")
        pd.DataFrame(r["trades"]).to_csv(Path(cfg["data"]["dir"]) / f"research_{title.split()[0].lower()}_{name}_trades.csv", index=False)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered strategy research")
    ap.add_argument("stage", choices=["dev", "final"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    frames = _load(cfg)
    cut = cfg["research"]["dev_end"]
    out_dir = Path(cfg["data"]["dir"])
    dev_file, lock_file = out_dir / "research_dev_result.json", out_dir / "research_final_DONE.json"
    first = min(f["timestamp"].iloc[0] for f in frames.values()).date()
    last = max(f["timestamp"].iloc[-1] for f in frames.values()).date()

    if args.stage == "dev":
        res = _report("Development test", f"{first} to before {cut}", frames, cfg, None, cut)
        dev_file.write_text(json.dumps(res, indent=2))
        survivors = [n for n in CANDIDATES if res[n]["passed"]]
        print("\nSurvivors of the development test:", survivors or "none")
        if survivors:
            print("Next step: python -m bot.research final --config research.yaml   (you only get ONE run)")
        else:
            print("No candidate passed. The honest conclusion is: no edge found in this family at this size. "
                  "Do not change the rules and retry on the same data.")
        return 0

    if lock_file.exists():
        print(f"The final exam was already run ({lock_file}). Running it again would turn it into a "
              "second development test. Refusing.")
        return 1
    if not dev_file.exists():
        print("Run the development test first: python -m bot.research dev --config research.yaml")
        return 1
    survivors = [n for n, v in json.loads(dev_file.read_text()).items() if v["passed"]]
    if not survivors:
        print("Nothing passed the development test, so there is nothing to take into the final exam.")
        return 1
    res = _report("Final exam", f"{cut} to {last}", frames, cfg, cut, None, only=survivors)
    lock_file.write_text(json.dumps(res, indent=2))
    winners = [n for n in CANDIDATES if n in res and res[n]["passed"]]
    if winners:
        print(f"\nPicked by the simplicity rule: {LABELS[winners[0]]}")
        print("Next step: paper trade it for months before any real money.")
    else:
        print("\nNo candidate passed the final exam. Conclusion: no edge found. Do not retune on this data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
