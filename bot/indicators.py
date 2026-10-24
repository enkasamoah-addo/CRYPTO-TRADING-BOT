"""Indicators. Every value at row i uses only rows up to i (no lookahead)."""
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range (Wilder smoothing): typical candle-to-candle price movement."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
