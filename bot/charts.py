"""Charts agent: moving average crossover, long only.

A signal fires on the CLOSE of the candle where the fast average crosses above the
slow one. The proposal's entry is that close price and its timestamp is the candle's
close time, so nothing here uses information from the future. A backtest should fill
the order at the NEXT candle's open, not at this entry price.
"""
from typing import List, Optional, Tuple

import pandas as pd

from .data import timeframe_ms
from .indicators import atr, sma
from .models import Proposal


def add_indicators(df: pd.DataFrame, strategy_cfg: dict) -> pd.DataFrame:
    out = df.copy()
    out["fast"] = sma(out["close"], strategy_cfg["fast_ma"])
    out["slow"] = sma(out["close"], strategy_cfg["slow_ma"])
    out["atr"] = atr(out, strategy_cfg["atr_period"])
    # Comparisons with NaN are False, so no signal fires before both averages exist.
    out["cross_up"] = (out["fast"].shift(1) <= out["slow"].shift(1)) & (out["fast"] > out["slow"])
    return out


def _make_proposal(row, symbol: str, tf_ms: int, strategy_cfg: dict) -> Optional[Proposal]:
    if pd.isna(row["atr"]) or row["atr"] <= 0:
        return None
    entry = float(row["close"])
    stop = entry - strategy_cfg["atr_stop_mult"] * float(row["atr"])
    if stop <= 0:
        return None
    target = entry + strategy_cfg["reward_risk"] * (entry - stop)
    closed_at = row["timestamp"] + pd.Timedelta(milliseconds=tf_ms)
    return Proposal(symbol, "long", entry, stop, target, closed_at.isoformat())


def all_proposals(df: pd.DataFrame, symbol: str, cfg: dict) -> List[Tuple[int, Proposal]]:
    """Every signal in the data, as (row index, Proposal). Used for backtesting."""
    strategy_cfg = cfg["strategy"]
    tf_ms = timeframe_ms(cfg["data"]["timeframe"])
    data = add_indicators(df, strategy_cfg)
    found = []
    for i in data.index[data["cross_up"]]:
        proposal = _make_proposal(data.loc[i], symbol, tf_ms, strategy_cfg)
        if proposal is not None:
            found.append((int(i), proposal))
    return found


def latest_proposal(df: pd.DataFrame, symbol: str, cfg: dict) -> Optional[Proposal]:
    """A proposal only if the most recent complete candle produced a signal."""
    if df.empty:
        return None
    strategy_cfg = cfg["strategy"]
    data = add_indicators(df, strategy_cfg)
    last = data.iloc[-1]
    if not last["cross_up"]:
        return None
    return _make_proposal(last, symbol, timeframe_ms(cfg["data"]["timeframe"]), strategy_cfg)
