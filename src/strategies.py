"""
Multi-leg options payoff engine.

Design:
  - Leg         : a single option position (frozen dataclass).
  - Strategy    : a collection of legs with payoff / analytics methods.
  - Factory fns : long_call, short_straddle, bull_call_spread, etc.

All P&L figures are per-unit (one options contract), not per lot.
Multiply by lot size when computing real-money P&L.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Leg
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Leg:
    """A single option leg."""

    option_type : Literal["call", "put"]
    strike      : float
    premium     : float                      # price paid / received per unit
    position    : Literal["long", "short"]
    quantity    : int = 1                    # number of contracts

    def intrinsic_at(self, spot: float) -> float:
        """Intrinsic value of the option at a given spot."""
        if self.option_type == "call":
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)

    def pnl_at(self, spot: float) -> float:
        """P&L per unit at expiry (intrinsic ± premium paid/received)."""
        sign = 1.0 if self.position == "long" else -1.0
        premium_flow = -self.premium if self.position == "long" else self.premium
        return sign * self.intrinsic_at(spot) * self.quantity + premium_flow * self.quantity


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """A collection of option legs forming a strategy."""

    name : str
    legs : list[Leg] = field(default_factory=list)

    def add_leg(self, leg: Leg) -> "Strategy":
        self.legs.append(leg)
        return self

    # ------------------------------------------------------------------
    # Core payoff
    # ------------------------------------------------------------------

    def pnl_at(self, spot: float) -> float:
        """Total P&L at a single spot price."""
        return sum(leg.pnl_at(spot) for leg in self.legs)

    def payoff(self, spots: np.ndarray) -> np.ndarray:
        """Vectorised payoff over an array of spot prices."""
        return np.array([self.pnl_at(float(s)) for s in spots])

    # ------------------------------------------------------------------
    # Summary analytics
    # ------------------------------------------------------------------

    def net_premium(self) -> float:
        """
        Net premium flow.
        Positive  = net debit  (you paid to enter).
        Negative  = net credit (you received premium).
        """
        total = 0.0
        for leg in self.legs:
            if leg.position == "long":
                total += leg.premium * leg.quantity
            else:
                total -= leg.premium * leg.quantity
        return total

    def _scan_range(self) -> np.ndarray:
        if not self.legs:
            raise ValueError("No legs defined.")
        min_k = min(leg.strike for leg in self.legs)
        max_k = max(leg.strike for leg in self.legs)
        centre = (min_k + max_k) / 2.0
        width  = max(centre * 0.40, max_k - min_k + 500)
        return np.linspace(max(centre - width, 0.0), centre + width, 10_000)

    def max_profit(self) -> float:
        pnl = self.payoff(self._scan_range())
        return float(np.max(pnl))

    def max_loss(self) -> float:
        pnl = self.payoff(self._scan_range())
        return float(np.min(pnl))

    def breakevens(self) -> list[float]:
        """Breakeven spot prices (sign changes in P&L curve)."""
        spots = self._scan_range()
        pnl   = self.payoff(spots)
        sign_changes = np.where(np.diff(np.sign(pnl)))[0]
        bps = []
        for i in sign_changes:
            x1, x2 = spots[i], spots[i + 1]
            y1, y2 = pnl[i], pnl[i + 1]
            if (y2 - y1) != 0:
                bp = x1 - y1 * (x2 - x1) / (y2 - y1)
                bps.append(round(bp, 2))
        return bps

    def summary(self) -> dict:
        mp  = self.max_profit()
        ml  = self.max_loss()
        rr  = abs(mp / ml) if ml != 0 else float("inf")
        return {
            "name"        : self.name,
            "legs"        : len(self.legs),
            "net_premium" : round(self.net_premium(), 2),
            "max_profit"  : round(mp, 2)  if mp < 1e9 else "unlimited",
            "max_loss"    : round(ml, 2)  if ml > -1e9 else "unlimited",
            "risk_reward" : round(rr, 2)  if rr < 1e9 else "unlimited",
            "breakevens"  : self.breakevens(),
        }

    def __repr__(self) -> str:
        s = self.summary()
        lines = [f"Strategy: {s['name']}"]
        for k, v in s.items():
            if k != "name":
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def long_call(strike: float, premium: float) -> Strategy:
    return Strategy("Long Call", [Leg("call", strike, premium, "long")])

def short_call(strike: float, premium: float) -> Strategy:
    return Strategy("Short Call", [Leg("call", strike, premium, "short")])

def long_put(strike: float, premium: float) -> Strategy:
    return Strategy("Long Put", [Leg("put", strike, premium, "long")])

def short_put(strike: float, premium: float) -> Strategy:
    return Strategy("Short Put", [Leg("put", strike, premium, "short")])

def bull_call_spread(
    low_strike: float, high_strike: float,
    low_premium: float, high_premium: float,
) -> Strategy:
    """Buy low strike call, sell high strike call."""
    return Strategy("Bull Call Spread", [
        Leg("call", low_strike,  low_premium,  "long"),
        Leg("call", high_strike, high_premium, "short"),
    ])

def bear_put_spread(
    high_strike: float, low_strike: float,
    high_premium: float, low_premium: float,
) -> Strategy:
    """Buy high strike put, sell low strike put."""
    return Strategy("Bear Put Spread", [
        Leg("put", high_strike, high_premium, "long"),
        Leg("put", low_strike,  low_premium,  "short"),
    ])

def long_straddle(strike: float, call_premium: float, put_premium: float) -> Strategy:
    """Buy ATM call and ATM put at the same strike."""
    return Strategy("Long Straddle", [
        Leg("call", strike, call_premium, "long"),
        Leg("put",  strike, put_premium,  "long"),
    ])

def short_straddle(strike: float, call_premium: float, put_premium: float) -> Strategy:
    """Sell ATM call and ATM put at the same strike."""
    return Strategy("Short Straddle", [
        Leg("call", strike, call_premium, "short"),
        Leg("put",  strike, put_premium,  "short"),
    ])

def long_strangle(
    call_strike: float, put_strike: float,
    call_premium: float, put_premium: float,
) -> Strategy:
    """Buy OTM call and OTM put."""
    return Strategy("Long Strangle", [
        Leg("call", call_strike, call_premium, "long"),
        Leg("put",  put_strike,  put_premium,  "long"),
    ])

def short_strangle(
    call_strike: float, put_strike: float,
    call_premium: float, put_premium: float,
) -> Strategy:
    """Sell OTM call and OTM put."""
    return Strategy("Short Strangle", [
        Leg("call", call_strike, call_premium, "short"),
        Leg("put",  put_strike,  put_premium,  "short"),
    ])

def iron_condor(
    put_long_strike : float, put_short_strike: float,
    call_short_strike: float, call_long_strike: float,
    p_long_prem: float, p_short_prem: float,
    c_short_prem: float, c_long_prem: float,
) -> Strategy:
    """
    Iron Condor:
      Long put (far OTM) / Short put / Short call / Long call (far OTM).
    """
    return Strategy("Iron Condor", [
        Leg("put",  put_long_strike,   p_long_prem,  "long"),
        Leg("put",  put_short_strike,  p_short_prem, "short"),
        Leg("call", call_short_strike, c_short_prem, "short"),
        Leg("call", call_long_strike,  c_long_prem,  "long"),
    ])


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Short straddle on NIFTY 19500
    strat = short_straddle(19500, call_premium=200, put_premium=190)
    print(strat)
    print()

    # Bull call spread
    spread = bull_call_spread(19400, 19600, low_premium=320, high_premium=160)
    print(spread)
