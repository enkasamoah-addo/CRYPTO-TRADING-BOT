"""Scout agent: screens pairs by recent trading volume and volatility.

It does not predict anything. It only decides which pairs are active enough
to be worth passing to the Charts agent.
"""
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class Candidate:
    symbol: str
    quote_volume: float   # approx. traded value over the lookback window (price x volume)
    volatility_pct: float # standard deviation of candle returns, in percent


def screen(frames: Dict[str, pd.DataFrame], cfg: dict) -> List[Candidate]:
    """Return the most volatile pairs that pass the volume and volatility minimums."""
    s = cfg["scout"]
    n = s["lookback_candles"]
    kept = []
    for symbol, df in frames.items():
        window = df.tail(n)
        if len(window) < n:
            continue  # not enough history to judge
        quote_volume = float((window["close"] * window["volume"]).sum())
        volatility = float(window["close"].pct_change().dropna().std() * 100)
        if quote_volume >= s["min_quote_volume"] and volatility >= s["min_volatility_pct"]:
            kept.append(Candidate(symbol, quote_volume, volatility))
    kept.sort(key=lambda c: c.volatility_pct, reverse=True)
    return kept[: s["top_n"]]
