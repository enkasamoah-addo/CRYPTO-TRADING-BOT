"""Backtester: replays history candle by candle through Charts -> Skeptic -> simulated orders.

Run from the project folder:   python -m bot.backtest

Realism rules (each one removes a way a backtest can flatter a strategy):
- A signal appears at a candle's CLOSE; the order fills at the NEXT candle's OPEN.
- Fees and slippage are charged on every entry and every exit.
- If one candle touches both the stop and the target, the STOP is assumed hit first.
- One position per symbol at a time; total positions are limited by the Skeptic.
- No leverage: orders are sized by risk, concentration and available cash.
It runs every period twice: with the Skeptic, and "naive" (every signal taken).
"""
import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .charts import all_proposals
from .config import load_config
from .data import csv_path, load_csv
from .decision_log import init_db, log_decision
from .sizing import position_size
from .skeptic import AccountState, Skeptic


@dataclass
class Result:
    label: str
    start_equity: float
    final_equity: float
    trades: List[dict]
    open_positions: List[dict]
    equity_curve: pd.Series
    decisions: List[dict]
    skipped_in_position: int
    fees_paid: float
    buy_and_hold_pct: float
    stats: dict = field(default_factory=dict)


def check_exit(o: float, h: float, l: float, stop: float, target: float) -> Optional[Tuple[float, str]]:
    """Did this candle close the trade? Returns (price, reason) or None.

    The stop is checked before the target, so a candle that touches both counts as a loss.
    Gaps through a level fill at the open, which is worse than the level for a stop.
    """
    if o <= stop:
        return o, "stop_gap"
    if l <= stop:
        return stop, "stop"
    if o >= target:
        return o, "target_gap"
    if h >= target:
        return target, "target"
    return None


def split_frames(frames: Dict[str, pd.DataFrame], n: int) -> List[Dict[str, pd.DataFrame]]:
    """Cut every symbol's history into n equal consecutive slices (same dates for all symbols)."""
    all_ts = sorted(set().union(*[set(df["timestamp"]) for df in frames.values()]))
    size = len(all_ts) // n
    slices = []
    for k in range(n):
        lo = all_ts[k * size]
        hi = all_ts[(k + 1) * size - 1] if k < n - 1 else all_ts[-1]
        slices.append({s: df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)].reset_index(drop=True)
                       for s, df in frames.items()})
    return slices


def _prepare(frames, cfg):
    swing_n = cfg["skeptic"]["swing_lookback_candles"]
    prepared = {}
    for symbol, df in frames.items():
        df = df.reset_index(drop=True)
        prepared[symbol] = {
            "ts": list(df["timestamp"]),
            "index": {t: i for i, t in enumerate(df["timestamp"])},
            "open": df["open"].to_numpy(), "high": df["high"].to_numpy(),
            "low": df["low"].to_numpy(), "close": df["close"].to_numpy(),
            "swing_low": df["low"].rolling(swing_n).min().to_numpy(),
            "props": dict(all_proposals(df, symbol, cfg)),
        }
    return prepared


def run_backtest(frames: Dict[str, pd.DataFrame], cfg: dict, use_skeptic: bool = True,
                 label: str = "", db_conn=None) -> Result:
    fee = cfg["fees"]["taker_pct"] / 100
    slip = cfg["fees"]["slippage_pct"] / 100
    max_dd = cfg["skeptic"]["max_drawdown_pct"]
    start_equity = float(cfg["backtest"]["starting_equity"])
    skeptic = Skeptic(cfg)
    prepared = _prepare(frames, cfg)
    timeline = sorted(set().union(*[set(d["ts"]) for d in prepared.values()]))

    cash = start_equity
    positions: List[dict] = []
    pending: List[tuple] = []  # approved (symbol, proposal) waiting for the next open
    trades: List[dict] = []
    decisions: List[dict] = []
    last_close: Dict[str, float] = {}
    equity_points = []
    peak = start_equity
    equity = start_equity
    halted = False
    day, day_start_equity = None, start_equity
    skipped = 0
    fees_paid = 0.0
    cap_divisor = cfg["skeptic"]["max_open_positions"] if use_skeptic else 1

    for t in timeline:
        # 1) Fill orders approved at the previous close, at this candle's open.
        for symbol, p in pending:
            d = prepared[symbol]
            i = d["index"].get(t)
            if i is None:
                continue
            open_ = float(d["open"][i])
            if not (p.stop < open_ < p.target):
                continue  # gapped past the stop or target before we could enter
            fill = open_ * (1 + slip)
            mark_equity = cash + sum(x["qty"] * last_close.get(x["symbol"], x["entry_fill"]) for x in positions)
            qty = position_size(fill, p.stop, mark_equity, cash, cfg, cap_divisor)
            cost = qty * fill
            if qty <= 0 or cost < 1.0:
                continue
            entry_fee = cost * fee
            cash -= cost + entry_fee
            fees_paid += entry_fee
            positions.append({"symbol": symbol, "qty": qty, "entry_fill": fill, "stop": p.stop,
                              "target": p.target, "entry_time": t, "entry_cost": cost + entry_fee})
        pending = []

        # 2) Did any open position hit its stop or target during this candle?
        for pos in list(positions):
            d = prepared[pos["symbol"]]
            i = d["index"].get(t)
            if i is None:
                continue  # this symbol has no candle at this time
            hit = check_exit(float(d["open"][i]), float(d["high"][i]), float(d["low"][i]),
                             pos["stop"], pos["target"])
            if hit is None:
                continue
            price, reason = hit
            exit_fill = price * (1 - slip)
            proceeds = pos["qty"] * exit_fill
            exit_fee = proceeds * fee
            cash += proceeds - exit_fee
            fees_paid += exit_fee
            pnl = proceeds - exit_fee - pos["entry_cost"]
            trades.append({"symbol": pos["symbol"], "entry_time": pos["entry_time"], "exit_time": t,
                           "entry": pos["entry_fill"], "exit": exit_fill, "qty": pos["qty"],
                           "pnl": pnl, "return_pct": pnl / pos["entry_cost"] * 100, "exit_reason": reason})
            positions.remove(pos)

        # 3) Mark everything to market at this candle's close.
        prev_equity = equity
        for symbol, d in prepared.items():
            i = d["index"].get(t)
            if i is not None:
                last_close[symbol] = float(d["close"][i])
        equity = cash + sum(x["qty"] * last_close[x["symbol"]] for x in positions)
        if t.date() != day:
            day = t.date()
            day_start_equity = prev_equity
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak * 100
        if use_skeptic and drawdown >= max_dd:
            halted = True  # latched: stays on until a human restarts the bot
        equity_points.append((t, equity))

        # 4) New signals on this candle's close. Judge each one; approved ones fill next open.
        for symbol, d in prepared.items():
            i = d["index"].get(t)
            if i is None or i not in d["props"]:
                continue
            p = d["props"][i]
            if any(x["symbol"] == symbol for x in positions) or any(s == symbol for s, _ in pending):
                skipped += 1
                continue
            if use_skeptic:
                state = AccountState(
                    equity=equity, open_positions=len(positions) + len(pending),
                    daily_pnl_pct=(equity - day_start_equity) / day_start_equity * 100,
                    drawdown_pct=drawdown, halted=halted)
                swing = d["swing_low"][i]
                verdict = skeptic.judge(p, state, None if pd.isna(swing) else float(swing))
                approved, reason, failed = verdict.approved, verdict.reason, verdict.failed
            else:
                approved, reason, failed = True, "no Skeptic (naive mode)", ()
            decisions.append({"time": p.created_at, "symbol": symbol, "approved": approved,
                              "failed": ",".join(failed), "reason": reason,
                              "entry": p.entry, "stop": p.stop, "target": p.target})
            if db_conn is not None:
                log_decision(db_conn, p, "approved" if approved else "blocked", reason)
            if approved:
                pending.append((symbol, p))

    open_positions = [{"symbol": x["symbol"], "qty": x["qty"], "entry_cost": x["entry_cost"],
                       "unrealized_pnl": x["qty"] * last_close[x["symbol"]] - x["entry_cost"]}
                      for x in positions]
    curve = pd.Series([e for _, e in equity_points], index=[t for t, _ in equity_points])
    holds = [prepared[s]["close"][-1] / prepared[s]["close"][0] - 1 for s in prepared if len(prepared[s]["close"])]
    result = Result(label, start_equity, equity, trades, open_positions, curve, decisions, skipped,
                    fees_paid, sum(holds) / len(holds) * 100 if holds else 0.0)
    result.stats = summarize(result)
    return result


def summarize(r: Result) -> dict:
    pnls = [t["pnl"] for t in r.trades]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    drawdown = ((r.equity_curve.cummax() - r.equity_curve) / r.equity_curve.cummax() * 100).max() \
        if len(r.equity_curve) else 0.0
    return {
        "return_pct": (r.final_equity / r.start_equity - 1) * 100,
        "trades": len(r.trades),
        "win_rate_pct": len(wins) / len(pnls) * 100 if pnls else 0.0,
        "profit_factor": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else None,
        "max_drawdown_pct": float(drawdown),
        "fees_paid": r.fees_paid,
        "signals_blocked": sum(1 for d in r.decisions if not d["approved"]),
        "signals_total": len(r.decisions),
    }


def _fmt(stats: dict) -> str:
    pf = "n/a" if stats["profit_factor"] is None else f"{stats['profit_factor']:.2f}"
    return (f"{stats['trades']:>4} trades  win {stats['win_rate_pct']:>4.0f}%  "
            f"return {stats['return_pct']:>+7.2f}%  maxDD {stats['max_drawdown_pct']:>5.2f}%  PF {pf:>5}  "
            f"fees {stats['fees_paid']:>7.2f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Backtest the Charts + Skeptic pipeline")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    d = cfg["data"]

    frames = {}
    for symbol in d["symbols"]:
        if not csv_path(d["dir"], symbol, d["timeframe"]).exists():
            print(f"Missing data for {symbol}. Run: python -m bot.download")
            return 1
        frames[symbol] = load_csv(d["dir"], symbol, d["timeframe"])

    n = cfg["backtest"]["periods"]
    periods = [("Full period", frames)] + [(f"Slice {k + 1}/{n}", f) for k, f in enumerate(split_frames(frames, n))]

    print(f"Starting equity {cfg['backtest']['starting_equity']} USDT. Fees {cfg['fees']['taker_pct']}% "
          f"+ slippage {cfg['fees']['slippage_pct']}% per side. Fills at next candle open.\n")
    full_skeptic = None
    for name, fr in periods:
        any_df = next(iter(fr.values()))
        span = f"{any_df['timestamp'].iloc[0].date()} to {any_df['timestamp'].iloc[-1].date()}"
        naive = run_backtest(fr, cfg, use_skeptic=False, label=name)
        guarded = run_backtest(fr, cfg, use_skeptic=True, label=name,
                               db_conn=init_db(":memory:") if name == "Full period" else None)
        if name == "Full period":
            full_skeptic, full_naive = guarded, naive
        print(f"{name} ({span}); buy-and-hold {naive.buy_and_hold_pct:+.1f}%")
        print(f"  naive   {_fmt(naive.stats)}")
        print(f"  skeptic {_fmt(guarded.stats)}\n")

    blocked = {}
    for dec in full_skeptic.decisions:
        if not dec["approved"]:
            for rule in dec["failed"].split(","):
                blocked[rule] = blocked.get(rule, 0) + 1
    print(f"Skeptic on the full period: {full_skeptic.stats['signals_blocked']} of "
          f"{full_skeptic.stats['signals_total']} signals blocked. Blocks by rule "
          f"(one signal can fail several):")
    for rule, count in sorted(blocked.items(), key=lambda kv: -kv[1]):
        print(f"  {rule}: {count}")

    out_dir = d["dir"]
    pd.DataFrame(full_skeptic.trades).to_csv(f"{out_dir}/backtest_trades_skeptic.csv", index=False)
    pd.DataFrame(full_skeptic.decisions).to_csv(f"{out_dir}/backtest_decisions_skeptic.csv", index=False)
    pd.DataFrame(full_naive.trades).to_csv(f"{out_dir}/backtest_trades_naive.csv", index=False)
    print(f"\nSaved trade and decision details to the {out_dir} folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
