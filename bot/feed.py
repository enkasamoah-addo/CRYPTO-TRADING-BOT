"""Live market data for Chief. Public data only: no API key, and it cannot place orders."""
import time

import pandas as pd

from .data import timeframe_ms, to_frame


class LiveFeed:
    def __init__(self, exchange, timeframe: str, limit: int = 300, retries: int = 3):
        self.exchange = exchange
        self.timeframe = timeframe
        self.tf_ms = timeframe_ms(timeframe)
        self.limit = limit  # enough history for the slowest indicator to warm up
        self.retries = retries

    def _retry(self, fn):
        last = None
        for attempt in range(self.retries):
            try:
                return fn()
            except Exception as exc:  # network hiccups are normal; try again, then give up loudly
                last = exc
                time.sleep(2 * (attempt + 1))
        raise last

    def candles(self, symbol: str) -> pd.DataFrame:
        """Recent COMPLETE candles (the still-forming one is dropped)."""
        rows = self._retry(lambda: self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=self.limit))
        df = to_frame(rows)
        now = pd.Timestamp.now(tz="UTC")
        return df[df["timestamp"] + pd.Timedelta(milliseconds=self.tf_ms) <= now].reset_index(drop=True)

    def price(self, symbol: str) -> float:
        return float(self._retry(lambda: self.exchange.fetch_ticker(symbol))["last"])
