"""Order execution. Only PAPER execution exists: it simulates fills and never contacts an
exchange for orders. A real/Testnet executor would implement the same two methods."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Fill:
    price: float   # average fill price, after slippage
    qty: float
    fee: float     # in USDT


class PaperExecutor:
    """Simulated fills at the given market price, with the configured slippage and fee."""

    def __init__(self, cfg: dict):
        self.fee = cfg["fees"]["taker_pct"] / 100
        self.slip = cfg["fees"]["slippage_pct"] / 100

    def expected_buy_price(self, price: float) -> float:
        return price * (1 + self.slip)

    def buy(self, symbol: str, qty: float, price: float) -> Fill:
        fill_price = self.expected_buy_price(price)
        return Fill(fill_price, qty, qty * fill_price * self.fee)

    def sell(self, symbol: str, qty: float, price: float) -> Fill:
        fill_price = price * (1 - self.slip)
        return Fill(fill_price, qty, qty * fill_price * self.fee)
