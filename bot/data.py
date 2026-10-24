"""Historical price data: download, quality checks, storage, and synthetic data for tests."""
import time
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}


def timeframe_ms(timeframe: str) -> int:
    """'1h' -> 3_600_000. Supports m, h, d, w."""
    unit = timeframe[-1]
    if unit not in _UNIT_MS or not timeframe[:-1].isdigit():
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    return int(timeframe[:-1]) * _UNIT_MS[unit]


def to_frame(rows) -> pd.DataFrame:
    """Raw [ms, o, h, l, c, v] rows -> clean, sorted, de-duplicated DataFrame (UTC timestamps)."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_ohlcv_history(exchange, symbol, timeframe, since_ms, until_ms=None,
                        limit=1000, now_ms=None) -> pd.DataFrame:
    """Download candles from `since_ms` forward, page by page.

    `exchange` is any ccxt-style object with fetch_ohlcv(symbol, timeframe, since, limit).
    The still-forming current candle is dropped, because its values are not final.
    """
    tf = timeframe_ms(timeframe)
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    rows, cursor = [], since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = batch[-1][0] + tf
        if next_cursor <= cursor:  # no progress -> avoid an infinite loop
            break
        cursor = next_cursor
        if len(batch) < limit or (until_ms is not None and cursor >= until_ms):
            break
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = to_frame(rows)
    last_complete_open = now_ms - tf  # a candle is complete once open + tf <= now
    df = df[df["timestamp"] <= pd.to_datetime(last_complete_open, unit="ms", utc=True)]
    if until_ms is not None:
        df = df[df["timestamp"] < pd.to_datetime(until_ms, unit="ms", utc=True)]
    return df.reset_index(drop=True)


def check_quality(df: pd.DataFrame, timeframe: str) -> dict:
    """Report problems that would quietly corrupt a backtest."""
    tf = pd.Timedelta(milliseconds=timeframe_ms(timeframe))
    if df.empty:
        return {"rows": 0, "gaps": [], "bad_ohlc_rows": 0, "nan_rows": 0, "ok": False}
    gaps = df["timestamp"].diff().iloc[1:]
    missing = df["timestamp"].iloc[1:][gaps > tf]
    bad_ohlc = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    nan_rows = df[COLUMNS].isna().any(axis=1)
    report = {
        "rows": len(df),
        "start": df["timestamp"].iloc[0].isoformat(),
        "end": df["timestamp"].iloc[-1].isoformat(),
        "gaps": [t.isoformat() for t in missing],
        "bad_ohlc_rows": int(bad_ohlc.sum()),
        "nan_rows": int(nan_rows.sum()),
    }
    report["ok"] = not report["gaps"] and report["bad_ohlc_rows"] == 0 and report["nan_rows"] == 0
    return report


def csv_path(data_dir, symbol, timeframe) -> Path:
    return Path(data_dir) / f"{symbol.replace('/', '-')}_{timeframe}.csv"


def save_csv(df: pd.DataFrame, data_dir, symbol, timeframe) -> Path:
    path = csv_path(data_dir, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_csv(data_dir, symbol, timeframe) -> pd.DataFrame:
    df = pd.read_csv(csv_path(data_dir, symbol, timeframe))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def make_synthetic(n=2000, timeframe="1h", start="2024-01-01", start_price=100.0,
                   vol=0.01, seed=0) -> pd.DataFrame:
    """Random-walk candles for testing code paths. NOT real market behaviour."""
    rng = np.random.default_rng(seed)
    closes = start_price * np.exp(np.cumsum(rng.normal(0, vol, n)))
    opens = np.concatenate([[start_price], closes[:-1]])
    spread = np.abs(rng.normal(0, vol / 2, n)) * closes
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread
    volume = rng.uniform(10, 100, n)
    step = pd.Timedelta(milliseconds=timeframe_ms(timeframe))
    ts = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n, freq=step)
    return pd.DataFrame({"timestamp": ts, "open": opens, "high": highs,
                         "low": lows, "close": closes, "volume": volume})
