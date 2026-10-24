"""Run Scout and Charts over the downloaded data and print what they find.

Run from the project folder:   python -m bot.preview
Needs the CSVs from `python -m bot.download`. This places no orders and judges
no trades; the Skeptic does not exist yet. It only shows what Scout and Charts see.
"""
import argparse
import sys

from .charts import all_proposals
from .config import load_config
from .data import check_quality, csv_path, load_csv
from .scout import screen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Preview Scout and Charts on downloaded data")
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

    print("SCOUT: pairs worth watching right now")
    for c in screen(frames, cfg):
        print(f"  {c.symbol}: volatility {c.volatility_pct:.3f}% per candle, "
              f"traded value {c.quote_volume:,.0f} over the last {cfg['scout']['lookback_candles']} candles")

    print("\nCHARTS: crossover signals in the downloaded history")
    for symbol, df in frames.items():
        report = check_quality(df, d["timeframe"])
        signals = all_proposals(df, symbol, cfg)
        print(f"  {symbol}: {len(df)} candles ({report['start'][:10]} to {report['end'][:10]}), "
              f"{len(signals)} signals")
        for _, p in signals[-3:]:
            print(f"    {p.created_at[:16]}  entry {p.entry:,.2f}  stop {p.stop:,.2f}  "
                  f"target {p.target:,.2f}  reward:risk {p.reward_risk:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
