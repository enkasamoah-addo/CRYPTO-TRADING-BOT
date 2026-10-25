"""Build the exchange connection (public market data only; no keys are ever used)."""
import ccxt

MAIN_HOST = "https://api.binance.com"


def make_exchange(cfg: dict):
    """ccxt exchange object. If config sets exchange.api_base (for example Binance's public-data
    mirror https://data-api.binance.vision), every price request goes there instead. Needed on
    servers in places where Binance's main API is blocked (GitHub's free servers are in the US)."""
    ex = getattr(ccxt, cfg["exchange"]["name"])({"enableRateLimit": True})
    base = (cfg["exchange"].get("api_base") or "").rstrip("/")
    if base:
        for key, url in list(ex.urls["api"].items()):
            if isinstance(url, str) and url.startswith(MAIN_HOST):
                ex.urls["api"][key] = base + url[len(MAIN_HOST):]
    return ex
