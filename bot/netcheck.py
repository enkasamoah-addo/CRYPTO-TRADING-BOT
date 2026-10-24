"""Can this machine get Binance's public price data? Used by the cloud job before each cycle.

    python -m bot.netcheck

Exit code 0 = yes, 1 = no. Binance refuses connections from some countries (HTTP 451), and
GitHub's free servers are in the US, so we check instead of assuming.
"""
import sys
import urllib.error
import urllib.request

ENDPOINTS = [
    ("Binance main API (what the bot uses)", "https://api.binance.com/api/v3/ping"),
    ("Binance public-data mirror", "https://data-api.binance.vision/api/v3/ping"),
]


def probe(url: str, opener=urllib.request.urlopen, timeout: int = 15) -> str:
    """Return 'ok' or a short plain-language reason."""
    try:
        with opener(url, timeout=timeout) as resp:
            return "ok" if resp.status == 200 else f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code == 451:
            return "HTTP 451: blocked for this location"
        return f"HTTP {exc.code}"
    except Exception as exc:
        return f"failed ({type(exc).__name__})"


def check_candles() -> str:
    """The real test: ask for a few daily candles exactly the way the bot does."""
    try:
        import ccxt
        rows = ccxt.binance({"enableRateLimit": True}).fetch_ohlcv("BTC/USDT", "1d", limit=5)
        return "ok" if len(rows) >= 3 else f"only {len(rows)} candles came back"
    except Exception as exc:
        return f"failed ({type(exc).__name__}: {str(exc)[:120]})"


def main(candles=check_candles, prober=probe) -> int:
    for name, url in ENDPOINTS:
        print(f"{name}: {prober(url)}")
    result = candles()
    print(f"Daily candles the way the bot fetches them: {result}")
    if result == "ok":
        print("RESULT: Binance is reachable from here.")
        return 0
    print("RESULT: the bot cannot get prices from here. It would need a different price source.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
