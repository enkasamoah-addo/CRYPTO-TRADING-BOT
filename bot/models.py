"""Shared data objects passed between agents."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Proposal:
    """A trade idea from the Charts agent. The Skeptic judges it; Chief executes it."""

    symbol: str
    side: str  # "long" only for now (Binance spot cannot short)
    entry: float
    stop: float
    target: float
    created_at: str  # ISO 8601 UTC timestamp

    def __post_init__(self):
        if self.side != "long":
            raise ValueError("only long proposals are supported")
        if not (self.stop < self.entry < self.target):
            raise ValueError("a long needs stop < entry < target")

    @property
    def risk(self) -> float:
        return self.entry - self.stop

    @property
    def reward(self) -> float:
        return self.target - self.entry

    @property
    def reward_risk(self) -> float:
        return self.reward / self.risk
