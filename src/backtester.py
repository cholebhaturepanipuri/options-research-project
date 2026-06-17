"""
Backtest engine for short-volatility options strategies.

Supports:
  - Short ATM straddle (sell ATM call + put at same strike).
  - Short strangle (sell OTM call + OTM put at fixed distance).

Entry  : first trading date of each expiry cycle.
Exit   : expiry date, stop-loss trigger, or premium-capture target.

IMPORTANT: All P&L figures are per index unit (not per lot).
           Multiply by lot_size when computing ₹ P&L.

NIFTY lot size: verify current value at nseindia.com before use.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

NIFTY_LOT_SIZE: int = 75          # verify current lot size from NSE
COST_PER_LOT_PER_LEG: float = 50  # ₹ approx. brokerage + STT + charges


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date          : pd.Timestamp
    expiry_date         : pd.Timestamp
    strategy            : str
    call_strike         : float
    put_strike          : float
    entry_call_premium  : float
    entry_put_premium   : float
    exit_call_premium   : float
    exit_put_premium    : float
    exit_reason         : str    # 'expiry' | 'stop_loss' | 'target'
    transaction_cost    : float  # ₹

    # ------------------------------------------------------------------
    # Derived P&L
    # ------------------------------------------------------------------

    @property
    def entry_total_premium(self) -> float:
        return self.entry_call_premium + self.entry_put_premium

    @property
    def exit_total_cost(self) -> float:
        return self.exit_call_premium + self.exit_put_premium

    @property
    def gross_pnl(self) -> float:
        """Premium received − cost to close (per unit, before lot-sizing)."""
        return self.entry_total_premium - self.exit_total_cost

    @property
    def net_pnl_rupees(self) -> float:
        """₹ P&L for one lot after transaction costs."""
        return self.gross_pnl * NIFTY_LOT_SIZE - self.transaction_cost

    def to_dict(self) -> dict:
        return {
            "entry_date"         : self.entry_date,
            "expiry_date"        : self.expiry_date,
            "strategy"           : self.strategy,
            "call_strike"        : self.call_strike,
            "put_strike"         : self.put_strike,
            "entry_total_premium": round(self.entry_total_premium, 2),
            "exit_total_cost"    : round(self.exit_total_cost, 2),
            "gross_pnl"          : round(self.gross_pnl, 2),
            "transaction_cost"   : round(self.transaction_cost, 2),
            "net_pnl_rupees"     : round(self.net_pnl_rupees, 2),
            "exit_reason"        : self.exit_reason,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _find_closest_strike(
    chain        : pd.DataFrame,
    spot         : float,
    option_type  : str,   # "CE" or "PE"
    target_strike: float,
    strike_col   : str = "strike",
    type_col     : str = "option_type",
    price_col    : str = "close",
) -> Optional[pd.Series]:
    """Return the row closest to target_strike for the given option type."""
    sub = chain[chain[type_col] == option_type]
    if sub.empty:
        return None
    dist = (sub[strike_col] - target_strike).abs()
    row  = sub.loc[dist.idxmin()]
    if pd.isna(row[price_col]) or row[price_col] <= 0:
        return None
    return row


def _get_day_prices(
    df         : pd.DataFrame,
    date       : pd.Timestamp,
    expiry     : pd.Timestamp,
    strike     : float,
    option_type: str,
    price_col  : str = "close",
) -> Optional[float]:
    """Fetch closing price for a specific (date, expiry, strike, type) row."""
    mask = (
        (df["date"]         == date)   &
        (df["expiry_date"]  == expiry) &
        (df["strike"]       == strike) &
        (df["option_type"]  == option_type)
    )
    subset = df[mask]
    if subset.empty:
        return None
    val = subset[price_col].iloc[0]
    return float(val) if not pd.isna(val) else None


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Backtest engine for short straddle and short strangle.

    Parameters
    ----------
    options_df : pd.DataFrame
        Clean options data with columns: date, expiry_date, strike,
        option_type (CE/PE), close, spot.
    stop_loss_multiplier : float
        Trigger stop-loss when current position cost >
        stop_loss_multiplier × entry premium received.
        E.g. 1.5 means max loss = 50% of entry premium (before costs).
    target_capture : float, optional
        Exit when (entry − current) / entry >= target_capture.
        E.g. 0.5 exits when 50% of premium has been captured.
        None = no target exit.
    otm_distance : float
        Strike offset (index points) for strangle legs.
    cost_per_lot_per_leg : float
        Transaction cost per lot per leg in ₹.
    lot_size : int
        Index options lot size.
    """

    def __init__(
        self,
        options_df           : pd.DataFrame,
        stop_loss_multiplier : float = 1.5,
        target_capture       : Optional[float] = None,
        otm_distance         : float = 100.0,
        cost_per_lot_per_leg : float = COST_PER_LOT_PER_LEG,
        lot_size             : int   = NIFTY_LOT_SIZE,
    ) -> None:
        self.df    = options_df.copy()
        self.sl    = stop_loss_multiplier
        self.tgt   = target_capture
        self.otm   = otm_distance
        self.cost  = cost_per_lot_per_leg
        self.lot   = lot_size
        self.trades: list[Trade] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tx_cost(self, n_legs: int = 2) -> float:
        return self.cost * n_legs * self.lot

    def _check_exit(
        self,
        entry_total  : float,
        current_total: float,
    ) -> Optional[Literal["stop_loss", "target"]]:
        """Return exit reason if a rule triggers, else None."""
        # Stop-loss: current cost to close has grown by sl × entry premium
        if current_total >= self.sl * entry_total:
            return "stop_loss"
        # Target: we've captured >= tgt fraction of premium
        if self.tgt is not None:
            captured = (entry_total - current_total) / entry_total
            if captured >= self.tgt:
                return "target"
        return None

    def _run_strategy(
        self,
        strategy_name: str,
        call_offset  : float,
        put_offset   : float,
    ) -> list[Trade]:
        """
        Generic loop over expiry cycles.

        call_offset / put_offset : how far OTM to go (0 = ATM straddle).
        """
        trades = []
        expiries = sorted(self.df["expiry_date"].unique())

        for expiry in expiries:
            cycle = self.df[self.df["expiry_date"] == expiry]
            dates = sorted(cycle["date"].unique())

            if len(dates) < 2:
                continue

            # ---------- Entry ----------
            entry_date  = dates[0]
            entry_chain = cycle[cycle["date"] == entry_date]

            if entry_chain.empty:
                continue

            spot = entry_chain["spot"].iloc[0]
            if pd.isna(spot):
                continue

            call_row = _find_closest_strike(
                entry_chain, spot, "CE", spot + call_offset
            )
            put_row  = _find_closest_strike(
                entry_chain, spot, "PE", spot - put_offset
            )

            if call_row is None or put_row is None:
                continue

            call_k        = float(call_row["strike"])
            put_k         = float(put_row["strike"])
            entry_call_p  = float(call_row["close"])
            entry_put_p   = float(put_row["close"])
            entry_total   = entry_call_p + entry_put_p

            if entry_total <= 0:
                continue

            # ---------- Holding period ----------
            exit_call_p = 0.0
            exit_put_p  = 0.0
            exit_reason = "expiry"

            for date in dates[1:]:
                curr_call = _get_day_prices(self.df, date, expiry, call_k, "CE")
                curr_put  = _get_day_prices(self.df, date, expiry, put_k,  "PE")

                # If prices unavailable, assume 0 at expiry
                if curr_call is None or curr_put is None:
                    if date == dates[-1]:
                        # Last day — use intrinsic from spot
                        day_data = cycle[cycle["date"] == date]
                        if not day_data.empty:
                            exp_spot = day_data["spot"].iloc[0]
                            curr_call = max(exp_spot - call_k, 0.0) if not pd.isna(exp_spot) else 0.0
                            curr_put  = max(put_k  - exp_spot, 0.0) if not pd.isna(exp_spot) else 0.0
                        else:
                            curr_call = curr_put = 0.0
                        exit_call_p = curr_call
                        exit_put_p  = curr_put
                        exit_reason = "expiry"
                    continue

                current_total = curr_call + curr_put

                reason = self._check_exit(entry_total, current_total)
                if reason is not None:
                    exit_call_p = curr_call
                    exit_put_p  = curr_put
                    exit_reason = reason
                    break

                if date == dates[-1]:
                    exit_call_p = curr_call
                    exit_put_p  = curr_put
                    exit_reason = "expiry"

            trades.append(Trade(
                entry_date         = entry_date,
                expiry_date        = expiry,
                strategy           = strategy_name,
                call_strike        = call_k,
                put_strike         = put_k,
                entry_call_premium = entry_call_p,
                entry_put_premium  = entry_put_p,
                exit_call_premium  = exit_call_p,
                exit_put_premium   = exit_put_p,
                exit_reason        = exit_reason,
                transaction_cost   = self._tx_cost(n_legs=2),
            ))

        return trades

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_short_straddle(self) -> "Backtester":
        """Enter ATM straddle (call and put at closest-to-spot strike)."""
        self.trades = self._run_strategy("short_straddle", 0.0, 0.0)
        return self

    def run_short_strangle(self) -> "Backtester":
        """Enter OTM strangle (call and put at otm_distance from spot)."""
        self.trades = self._run_strategy("short_strangle", self.otm, self.otm)
        return self

    def trade_log(self) -> pd.DataFrame:
        """Return all trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal synthetic dataset
    np.random.seed(0)
    dates    = pd.date_range("2023-01-02", "2023-03-31", freq="B")
    expiries = [pd.Timestamp("2023-01-26"), pd.Timestamp("2023-02-23"),
                pd.Timestamp("2023-03-30")]

    rows = []
    for expiry in expiries:
        for d in dates:
            if d > expiry:
                continue
            T   = (expiry - d).days / 365
            S   = 18000 + np.random.normal(0, 200)
            for strike in [S - 100, S, S + 100]:
                for ot, is_call in [("CE", True), ("PE", False)]:
                    from option_pricing import bs_price
                    prc = bs_price(S, round(strike, -2), 0.065, 0.14, max(T, 1e-4),
                                   "call" if is_call else "put")
                    rows.append({
                        "date"        : d,
                        "expiry_date" : expiry,
                        "strike"      : round(strike, -2),
                        "option_type" : ot,
                        "close"       : round(prc, 2),
                        "spot"        : round(S, 2),
                    })

    df = pd.DataFrame(rows)
    bt = Backtester(df, stop_loss_multiplier=1.5, target_capture=0.5)
    bt.run_short_straddle()
    log = bt.trade_log()
    print(log[["entry_date", "expiry_date", "entry_total_premium",
               "net_pnl_rupees", "exit_reason"]].to_string(index=False))
