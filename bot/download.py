"""Download historical candles for every symbol in config.yaml.

Run from the project folder:   python -m bot.download
Uses Binance's PUBLIC market data, so no API key is needed.
"""
import argparse
import sys
import time
from datetime import datetime, timezone

from .config import load_config
from .exchange import make_exchange
from .data import check_quality, fetch_ohlcv_history, save_csv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download historical OHLCV data")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    exchange = make_exchange(cfg)

    if data_cfg.get("history_start"):   # fixed start date (research config), e.g. 2017-08-17
        start = datetime.fromisoformat(str(data_cfg["history_start"])).replace(tzinfo=timezone.utc)
        since_ms = int(start.timestamp() * 1000)
    else:
        since_ms = int((time.time() - data_cfg["history_days"] * 86_400) * 1000)
    all_ok = True
    for symbol in data_cfg["symbols"]:
        print(f"Downloading {symbol} {data_cfg['timeframe']} ...", flush=True)
        df = fetch_ohlcv_history(exchange, symbol, data_cfg["timeframe"], since_ms)
        report = check_quality(df, data_cfg["timeframe"])
        path = save_csv(df, data_cfg["dir"], symbol, data_cfg["timeframe"])
        status = "OK" if report["ok"] else "PROBLEMS FOUND"
        print(f"  {report['rows']} candles -> {path}  [{status}]")
        if not report["ok"]:
            all_ok = False
            print(f"  gaps: {len(report['gaps'])}, bad OHLC rows: {report['bad_ohlc_rows']}, "
                  f"NaN rows: {report['nan_rows']}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
