"""Skeptic agent: plain-code rules that can veto any trade proposal.

No AI is involved, so the Skeptic cannot be talked out of a decision. Every failed
rule is reported by name and in words, and all of them are collected (not just the first).
All thresholds come from config.yaml.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .models import Proposal


@dataclass
class AccountState:
    equity: float
    open_positions: int          # including approved orders waiting to fill
    daily_pnl_pct: float = 0.0   # change in equity since the start of the UTC day
    drawdown_pct: float = 0.0    # fall from the highest equity so far
    halted: bool = False         # kill switch latched; only a manual restart clears it
    data_ok: bool = True         # False if market data is stale or the API is erroring


@dataclass(frozen=True)
class Verdict:
    approved: bool
    reason: str
    failed: Tuple[str, ...] = ()


def round_number_distance_pct(price: float) -> float:
    """Distance (in % of price) from `price` to the nearest round number.

    'Round' means a multiple of one power of ten below the price's size:
    2,416 -> multiples of 100; 81,394 -> multiples of 1,000; 91.3 -> multiples of 1.
    """
    step = 10 ** (math.floor(math.log10(price)) - 1)
    nearest = round(price / step) * step
    return abs(price - nearest) / price * 100


class Skeptic:
    def __init__(self, cfg: dict):
        self.rules = cfg["skeptic"]
        self.fees = cfg["fees"]

    def judge(self, p: Proposal, state: AccountState, recent_low: Optional[float] = None) -> Verdict:
        r = self.rules
        failed = []  # (rule name, explanation)

        if not state.data_ok:
            failed.append(("bad_data", "market data is stale or the API is returning errors"))
        if state.halted:
            failed.append(("halted", "kill switch is active after the maximum drawdown"))
        if state.drawdown_pct >= r["max_drawdown_pct"]:
            failed.append(("drawdown_kill", f"drawdown {state.drawdown_pct:.1f}% reached the "
                                            f"{r['max_drawdown_pct']}% limit"))
        if state.daily_pnl_pct <= -r["max_daily_loss_pct"]:
            failed.append(("daily_loss", f"down {abs(state.daily_pnl_pct):.1f}% today, limit is "
                                         f"{r['max_daily_loss_pct']}%"))
        if state.open_positions >= r["max_open_positions"]:
            failed.append(("max_positions", f"already at {r['max_open_positions']} open positions"))
        # Tiny tolerance: 2.0 computed from prices can come out as 1.9999999999999998.
        if p.reward_risk < r["min_reward_risk"] - 1e-9:
            failed.append(("reward_risk", f"reward-to-risk {p.reward_risk:.2f} is below "
                                          f"{r['min_reward_risk']}"))

        round_cfg = r.get("reject_stop_on_round_number", False)
        if round_cfg:
            dist = round_number_distance_pct(p.stop)
            if dist <= r["round_number_tolerance_pct"]:
                failed.append(("round_number_stop", f"stop {p.stop:,.2f} sits within {dist:.2f}% of a "
                                                    f"round number, where other stops cluster"))

        if r.get("reject_stop_at_swing_low", False) and recent_low is not None and recent_low > 0:
            dist = abs(p.stop - recent_low) / recent_low * 100
            if dist <= r["swing_tolerance_pct"]:
                failed.append(("swing_low_stop", f"stop {p.stop:,.2f} sits within {dist:.2f}% of the "
                                                 f"recent low {recent_low:,.2f}"))

        # Round-trip trading costs (fee + slippage, entry and exit) vs. the profit if the target hits.
        cost = 2 * (self.fees["taker_pct"] + self.fees["slippage_pct"]) / 100 * p.entry
        fee_share = cost / p.reward * 100
        if fee_share > r["max_fee_share_of_target_pct"]:
            failed.append(("fees_vs_target", f"trading costs would eat {fee_share:.0f}% of the target "
                                             f"profit (limit {r['max_fee_share_of_target_pct']}%)"))

        # Rules that do not apply to a strategy (e.g. profit-target rules for a trend strategy
        # with no target) are switched off by name in config: skeptic.disabled_rules.
        off = set(r.get("disabled_rules") or [])
        failed = [(name, msg) for name, msg in failed if name not in off]

        if failed:
            return Verdict(False, "; ".join(msg for _, msg in failed), tuple(name for name, _ in failed))
        return Verdict(True, "all rules passed", ())
