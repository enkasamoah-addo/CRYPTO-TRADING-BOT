"""Dashboard: turns the paper bot's database and daily candles into ONE static HTML page.

    python -m bot.dashboard --config paper_trend.yaml        writes data/dashboard.html

Read-only: the database is opened in read-only mode, so this can never change a trade. The page
needs no server. Charts use the lightweight-charts library, loaded from a CDN, so viewing the
page needs internet (the numbers and tables show without it). Candles come from Binance's public
data when reachable, otherwise from the CSV files in data/ (the page says which, and how old).
"""
import argparse
import html
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import load_config
from .data import csv_path, load_csv
from .indicators import sma

CANDLES_SHOWN = 150
LIB_URL = "https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"

CSS = """
body{margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f5f1;color:#1d2420}
.wrap{box-sizing:border-box;max-width:1280px;margin:0 auto;padding:32px 40px 40px;display:flex;flex-direction:column;gap:24px}
h1{margin:0;font-size:26px;font-weight:700;letter-spacing:-0.01em}
.sub{font-size:14px;color:#5b6660}
.row{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
.pill{font-size:13px;color:#3f4944;background:#ecebe5;padding:7px 14px;border-radius:999px}
.pill.ok{color:#14573a;background:#dcefe4;font-weight:600}
.pill.bad{color:#7a1f10;background:#f7dfd8;font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}
.card{background:#fff;border:1px solid #e3e1d9;border-radius:14px;padding:20px;min-width:0}
.kpi .l{font-size:13px;color:#5b6660}.kpi .v{font-size:28px;font-weight:700;margin:6px 0}.kpi .n{font-size:13px;color:#5b6660}
.pos{color:#1f6b4a}.neg{color:#b5532f}
h2{margin:0 0 12px;font-size:16px;font-weight:600}
.cols{display:flex;flex-wrap:wrap;gap:16px;align-items:stretch}
.main{flex:999 1 620px;min-width:0}.side{flex:1 1 300px;min-width:0}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:#5b6660;margin-bottom:8px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
svg{width:100%;height:auto;display:block}
.act{display:flex;flex-direction:column;gap:12px;font-size:14px;line-height:1.45}
.act .d{color:#5b6660}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%}
th{font-weight:500;color:#5b6660;font-size:12px;text-align:left;padding:8px 12px;border-bottom:1px solid #e3e1d9;white-space:nowrap}
td{padding:12px;font-size:14px;border-bottom:1px solid #eeece6;white-space:nowrap}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
.muted{color:#5b6660;font-size:13px}
@media (max-width:600px){.wrap{padding:20px 16px}.charts{grid-template-columns:1fr}}
"""


# ------------------------------------------------------------------ data

def load_candles(cfg: dict, symbols, try_live: bool = True):
    """Return ({symbol: DataFrame}, source note)."""
    d = cfg["data"]
    frames, notes = {}, []
    if try_live:
        try:
            from .exchange import make_exchange
            from .feed import LiveFeed
            feed = LiveFeed(make_exchange(cfg), d["timeframe"],
                            retries=1)
            for s in symbols:
                frames[s] = feed.candles(s)
            notes.append("live Binance data")
        except Exception:
            frames = {}
    if not frames:
        for s in symbols:
            if csv_path(d["dir"], s, d["timeframe"]).exists():
                frames[s] = load_csv(d["dir"], s, d["timeframe"])
        if frames:
            last = max(f["timestamp"].iloc[-1] for f in frames.values()).date()
            notes.append(f"saved files, latest candle {last}")
    return frames, "; ".join(notes) or "no candle data available"


def _rows(conn, sql):
    cur = conn.execute(sql)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _unique_days(points):
    """lightweight-charts needs one point per day, ascending: keep the last value of each day."""
    by_day = {}
    for p in points:
        by_day[p["time"]] = p
    return [by_day[k] for k in sorted(by_day)]


def collect(conn: sqlite3.Connection, cfg: dict, frames: Dict[str, pd.DataFrame]) -> dict:
    """Everything the page shows, read from the database (read-only queries) and the candles."""
    def get(key, default=None):
        row = conn.execute("SELECT value FROM account WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    start = float(cfg["backtest"]["starting_equity"])
    positions = _rows(conn, "SELECT id, symbol, qty, entry_fill, stop, entry_time, entry_cost FROM positions ORDER BY id")
    trades = _rows(conn, "SELECT symbol, entry_time, exit_time, entry, exit, qty, pnl, return_pct, exit_reason "
                         "FROM trades ORDER BY id")
    last_close = get("last_close", {}) or {}
    cash = get("cash", start)
    invested = sum(p["qty"] * last_close.get(p["symbol"], p["entry_fill"]) for p in positions)
    equity = cash + invested
    curve = [(r[0], r[1]) for r in conn.execute("SELECT ts, equity FROM equity_log ORDER BY rowid")]
    peak, worst = start, 0.0
    for e in [start] + [v for _, v in curve]:
        peak = max(peak, e)
        worst = max(worst, (peak - e) / peak * 100)
    bh_start = get("bh_start")
    bh_ret = None
    if bh_start:
        bh_ret = (sum(last_close.get(s, px) / px for s, px in bh_start.items()) / len(bh_start) - 1) * 100

    bot_curve = _unique_days([{"time": _day(t), "value": round(v, 2)} for t, v in curve])
    hold_curve = []
    if bh_start and all(s in frames for s in bh_start):
        for pt in bot_curve:
            t = pd.Timestamp(pt["time"], tz="UTC") + pd.Timedelta(days=1)   # that day's close
            ratios = []
            for s, px0 in bh_start.items():
                f = frames[s][frames[s]["timestamp"] < t]
                ratios.append((float(f["close"].iloc[-1]) if len(f) else px0) / px0)
            hold_curve.append({"time": pt["time"], "value": round(start * sum(ratios) / len(ratios), 2)})

    ma_days = cfg["research"]["trend_days"]
    held = {p["symbol"]: p for p in positions}
    charts = {}
    for s, df in frames.items():
        if df.empty:
            continue
        shown = df.assign(ma=sma(df["close"], ma_days)).tail(CANDLES_SHOWN)
        candles = [{"time": _day(r.timestamp), "open": float(r.open), "high": float(r.high),
                    "low": float(r.low), "close": float(r.close)} for r in shown.itertuples()]
        days = [c["time"] for c in candles]
        first, last = days[0], days[-1]
        def place(ts):
            d = min(_day(ts), last)          # today's candle is not final yet: pin to the latest one
            return d if d >= first else None
        markers = []
        for p in positions:
            if p["symbol"] == s and place(p["entry_time"]):
                markers.append({"time": place(p["entry_time"]), "kind": "buy", "text": "bought"})
        for t in trades:
            if t["symbol"] != s:
                continue
            if place(t["entry_time"]):
                markers.append({"time": place(t["entry_time"]), "kind": "buy", "text": "bought"})
            if place(t["exit_time"]):
                markers.append({"time": place(t["exit_time"]), "kind": "sell",
                                "text": f"sold {t['pnl']:+.2f}"})
        markers.sort(key=lambda m: m["time"])
        pos = held.get(s)
        ma_last = shown["ma"].iloc[-1]
        charts[s] = {"candles": candles,
                     "ma": [{"time": _day(r.timestamp), "value": float(r.ma)} for r in shown.itertuples()
                            if pd.notna(r.ma)],
                     "markers": markers,
                     "entry": pos["entry_fill"] if pos else None, "stop": pos["stop"] if pos else None,
                     "last_close": float(df["close"].iloc[-1]),
                     "ma_last": float(ma_last) if pd.notna(ma_last) else None}
    return {"start": start, "equity": equity, "cash": cash, "invested": invested, "positions": positions,
            "trades": trades, "n_trades": len(trades), "total_pnl": sum(t["pnl"] for t in trades),
            "return_pct": (equity / start - 1) * 100, "drawdown_pct": worst,
            "halted": bool(get("halted", False)),
            "invested_pct": invested / equity * 100 if equity > 0 else 0.0,
            "bh_ret": bh_ret, "last_close": last_close,
            "last_cycle": curve[-1][0] if curve else None,
            "max_dd_limit": cfg["skeptic"]["max_drawdown_pct"], "trend_days": ma_days,
            "symbols": cfg["data"]["symbols"], "bot_curve": bot_curve, "hold_curve": hold_curve,
            "charts": charts, "source": ""}


# ------------------------------------------------------------------ pieces

def _e(x) -> str:
    return html.escape(str(x))


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    cls = "pos" if x > 0 else ("neg" if x < 0 else "")
    return f'<span class="{cls}">{x:+.1f}%</span>'


def activity(info: dict) -> List[str]:
    events = []
    for p in info["positions"]:
        events.append((p["entry_time"], f'<b>Bought {_e(p["symbol"])}</b> at {p["entry_fill"]:,.2f}, '
                                         f'stop {p["stop"]:,.2f}'))
    for t in info["trades"]:
        why = {"stop": "stop-loss hit", "stop_gap": "stop-loss hit (price gapped)",
               "trend_exit": "price closed below its average"}.get(t["exit_reason"], t["exit_reason"])
        events.append((t["exit_time"], f'<b>Sold {_e(t["symbol"])}</b> ({_e(why)}), '
                                       f'P&amp;L {t["pnl"]:+.2f} USDT'))
    events.sort(key=lambda e: e[0], reverse=True)
    return [f'<div><span class="d">{_e(ts[:10])}</span><br>{text}</div>' for ts, text in events[:8]]


# ------------------------------------------------------------------ page

def _json(obj) -> str:
    """JSON that is safe inside a <script> block: '</script>' in any text can never end it early."""
    return (json.dumps(obj).replace("</", "<\\/").replace("<!--", "<\\!--")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


SCRIPT = r"""
const D = __DATA__;
function mk(el){return LightweightCharts.createChart(el,{autoSize:true,layout:{background:{color:'#ffffff'},textColor:'#5b6660'},
  grid:{vertLines:{color:'#eeece6'},horzLines:{color:'#eeece6'}},rightPriceScale:{borderColor:'#e3e1d9'},timeScale:{borderColor:'#e3e1d9'}});}
if (typeof LightweightCharts === 'undefined') {
  document.querySelectorAll('.chartbox').forEach(function(b){b.innerHTML='<p class="muted">Charts need an internet connection to load. The numbers and tables above still work.</p>';});
} else {
  const eq = document.getElementById('equity-chart');
  if (eq && D.bot_curve.length > 1) {
    const c = mk(eq);
    if (D.hold_curve.length > 1) c.addLineSeries({color:'#9aa39e',lineWidth:2,priceLineVisible:false}).setData(D.hold_curve);
    c.addLineSeries({color:'#1f6b4a',lineWidth:3,priceLineVisible:false}).setData(D.bot_curve);
    c.timeScale().fitContent();
  }
  Object.keys(D.charts).forEach(function(sym){
    const el = document.getElementById('chart-' + sym.replace('/','-')); if(!el) return;
    const d = D.charts[sym], c = mk(el);
    const cs = c.addCandlestickSeries({upColor:'#ffffff',downColor:'#b5532f',borderUpColor:'#1f6b4a',borderDownColor:'#b5532f',
      wickUpColor:'#1f6b4a',wickDownColor:'#b5532f'});
    cs.setData(d.candles);
    c.addLineSeries({color:'#2f5bd0',lineWidth:2,priceLineVisible:false,lastValueVisible:false}).setData(d.ma);
    cs.setMarkers(d.markers.map(function(m){return {time:m.time,position:m.kind==='buy'?'belowBar':'aboveBar',
      color:m.kind==='buy'?'#1f6b4a':'#b5532f',shape:m.kind==='buy'?'arrowUp':'arrowDown',text:d.markers.length<8?m.text:''};}));
    if (d.entry) cs.createPriceLine({price:d.entry,color:'#2f5bd0',lineWidth:1,lineStyle:LightweightCharts.LineStyle.Dashed,title:'bought'});
    if (d.stop) cs.createPriceLine({price:d.stop,color:'#b5532f',lineWidth:1,lineStyle:LightweightCharts.LineStyle.Dashed,title:'stop'});
    c.timeScale().fitContent();
  });
}
"""


def render(info: dict, title: str = "Trend bot") -> str:
    ma_days = info["trend_days"]
    kill = ('<span class="pill bad">KILL SWITCH ON: new entries blocked</span>' if info["halted"]
            else '<span class="pill ok">Kill switch: off</span>')
    last = info["last_cycle"][:16].replace("T", " ") + " UTC" if info["last_cycle"] else "never"
    pos_rows = "".join(
        f'<tr><td><b>{_e(p["symbol"])}</b></td><td>{p["qty"]:.6f}</td><td>{p["entry_fill"]:,.2f}</td>'
        f'<td>{p["stop"]:,.2f}</td><td>{p["qty"] * (p["entry_fill"] - p["stop"]):,.2f} USDT</td>'
        f'<td>{p["qty"] * info["last_close"].get(p["symbol"], p["entry_fill"]):,.2f} USDT</td>'
        f'<td>Close below {ma_days}-day average</td></tr>' for p in info["positions"]
    ) or '<tr><td colspan="7" class="muted">No open positions. The bot is in cash and waiting.</td></tr>'
    trade_rows = "".join(
        f'<tr><td>{_e(t["symbol"])}</td><td>{_e(t["entry_time"][:10])}</td><td>{_e(t["exit_time"][:10])}</td>'
        f'<td>{t["entry"]:,.2f}</td><td>{t["exit"]:,.2f}</td>'
        f'<td class="{"pos" if t["pnl"] > 0 else "neg"}">{t["pnl"]:+.2f}</td><td>{_e(t["exit_reason"])}</td></tr>'
        for t in reversed(info["trades"][-15:])
    ) or '<tr><td colspan="7" class="muted">No closed trades yet.</td></tr>'
    notes = {}
    for sym, c in info["charts"].items():
        if c["ma_last"] is not None:
            up = c["last_close"] > c["ma_last"]
            notes[sym] = (f'Last close {c["last_close"]:,.2f}; {ma_days}-day average {c["ma_last"]:,.2f}. '
                          + ("Trend is up: the bot wants to hold." if up else "Trend is down: the bot wants to be out."))
    charts = "".join(
        f'<div class="card"><h2>{_e(sym)}</h2><div class="chartbox" id="chart-{_e(sym.replace("/", "-"))}" '
        f'style="height:320px"></div><p class="muted">{_e(notes.get(sym, ""))}</p></div>'
        for sym in info["charts"]) or '<div class="card muted">No candle data found.</div>'
    act = "".join(activity(info)) or '<div class="muted">Nothing yet. Most days the bot does nothing, and that is normal.</div>'
    equity_box = ('<div class="chartbox" id="equity-chart" style="height:280px"></div>' if len(info["bot_curve"]) > 1 else
                  '<p class="muted">The line appears after the second daily check. '
                  'Most days the bot does nothing, and that is normal.</p>')
    data = {"bot_curve": info["bot_curve"], "hold_curve": info["hold_curve"], "charts": info["charts"]}
    script = SCRIPT.replace("__DATA__", _json(data))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)} dashboard</title><style>{CSS}</style>
<script src="{LIB_URL}"></script></head><body><div class="wrap">
<div class="row"><div><h1>{_e(title)}</h1>
<div class="sub">Simulated trading: no real money, real prices. Daily trend filter on {_e(", ".join(info["symbols"]))}.</div></div>
<div class="row" style="gap:10px"><span class="pill">Last check: {_e(last)}</span>{kill}</div></div>
<div class="kpis">
<div class="card kpi"><div class="l">Equity</div><div class="v">{info["equity"]:,.2f} USDT</div><div class="n">Started with {info["start"]:,.0f}; cash {info["cash"]:,.2f}</div></div>
<div class="card kpi"><div class="l">This bot vs buy-and-hold</div><div class="v">{_pct(info["return_pct"])} / {_pct(info["bh_ret"])}</div><div class="n">Since the first check (hold excludes fees)</div></div>
<div class="card kpi"><div class="l">Worst drop so far</div><div class="v">{info["drawdown_pct"]:.1f}%</div><div class="n">Kill switch trips at {info["max_dd_limit"]}%</div></div>
<div class="card kpi"><div class="l">Money in the market</div><div class="v">{info["invested_pct"]:.0f}%</div><div class="n">The rest is cash, by design (1% risk per trade)</div></div>
</div>
<div class="cols"><div class="card main"><h2>Equity over time</h2>
<div class="legend"><span><span class="sw" style="background:#1f6b4a"></span>This bot</span><span><span class="sw" style="background:#9aa39e"></span>Buy and hold</span></div>
{equity_box}</div>
<div class="card side"><h2>Latest activity</h2><div class="act">{act}</div></div></div>
<div class="card"><h2>Open positions</h2><div class="tw"><table><thead><tr><th>Coin</th><th>Amount</th><th>Bought at</th><th>Stop</th><th>Risk if stop hits</th><th>Worth now</th><th>Exit rule</th></tr></thead><tbody>{pos_rows}</tbody></table></div></div>
<div class="charts">{charts}</div>
<div class="card"><h2>Closed trades</h2><div class="tw"><table><thead><tr><th>Coin</th><th>Bought</th><th>Sold</th><th>Buy price</th><th>Sell price</th><th>P&amp;L (USDT)</th><th>Why</th></tr></thead><tbody>{trade_rows}</tbody></table></div>
<p class="muted">{info["n_trades"]} closed so far, total {info["total_pnl"]:+.2f} USDT.</p></div>
<div class="muted">Candle data: {_e(info["source"])}. The blue line is the {ma_days}-day average; the bot holds while price is above it. Arrows mark simulated buys and sells.</div>
</div><script>{script}</script></body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write the paper-trading dashboard as one HTML file")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="output file (default: <data dir>/dashboard.html)")
    ap.add_argument("--offline", action="store_true", help="use saved CSV candles, do not contact Binance")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    db = Path(cfg["logging"]["db_path"])
    if not db.exists():
        print(f"No database at {db}. Run the bot once first: python -m bot.chief --config ... --once")
        return 1
    conn = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)   # read-only: cannot change anything
    frames, source = load_candles(cfg, cfg["data"]["symbols"], try_live=not args.offline)
    info = collect(conn, cfg, frames)
    info["source"] = source
    conn.close()
    out = Path(args.out or Path(cfg["data"]["dir"]) / "dashboard.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(info), encoding="utf-8")
    print(f"Dashboard written to {out.resolve()}  (open it in your browser)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
