"""Position sizing: how much to buy once a trade is approved."""


def position_size(entry: float, stop: float, equity: float, cash: float,
                  cfg: dict, cap_divisor: int) -> float:
    """Units to buy. The smallest of three limits:

    1. Risk limit: if the stop is hit, lose at most max_risk_per_trade_pct of equity.
    2. Concentration limit: one position may use at most equity / cap_divisor.
       (Spot trading has no leverage, so a tight stop could otherwise ask for more
       money than the account holds.)
    3. Cash limit: cannot spend money the account does not have.
    """
    risk_per_unit = entry - stop
    if risk_per_unit <= 0 or entry <= 0 or equity <= 0:
        return 0.0
    risk_frac = cfg["skeptic"]["max_risk_per_trade_pct"] / 100
    by_risk = equity * risk_frac / risk_per_unit
    by_concentration = equity / max(cap_divisor, 1) / entry
    cost_factor = 1 + cfg["fees"]["taker_pct"] / 100 + cfg["fees"]["slippage_pct"] / 100
    by_cash = max(cash, 0.0) / (entry * cost_factor)
    return max(0.0, min(by_risk, by_concentration, by_cash))
