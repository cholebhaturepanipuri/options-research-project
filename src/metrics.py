"""
Performance metrics for backtested strategies.

All metrics operate on the trade log DataFrame produced by Backtester.trade_log().
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    trade_log       : pd.DataFrame,
    pnl_col         : str = "net_pnl_rupees",
    risk_free_rate  : float = 0.065,
) -> dict:
    """
    Compute the ten required performance metrics from a trade log.

    Parameters
    ----------
    trade_log      : pd.DataFrame   Output of Backtester.trade_log().
    pnl_col        : str            Column name for per-trade P&L in ₹.
    risk_free_rate : float          Annual risk-free rate (for Sharpe).

    Returns
    -------
    dict with keys:
        total_return, avg_trade_pnl, win_rate, num_trades,
        max_drawdown, sharpe_ratio, profit_factor,
        best_trade, worst_trade, total_transaction_cost,
        exit_reasons
    """
    if trade_log.empty or pnl_col not in trade_log.columns:
        return {}

    pnl    = trade_log[pnl_col].dropna()
    equity = pnl.cumsum()

    total_return = pnl.sum()
    avg_trade    = pnl.mean()
    win_rate     = (pnl > 0).mean() * 100.0
    num_trades   = len(pnl)

    # Drawdown (on equity curve)
    running_max  = equity.cummax()
    drawdown     = equity - running_max
    max_drawdown = float(drawdown.min())

    # Sharpe (trade-level, annualised by sqrt of trades)
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(num_trades)) if pnl.std() > 0 else np.nan

    # Profit factor
    gains  = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    profit_factor = float(gains / losses) if losses > 0 else float("inf")

    best_trade  = float(pnl.max())
    worst_trade = float(pnl.min())

    total_cost = (
        float(trade_log["transaction_cost"].sum())
        if "transaction_cost" in trade_log.columns else 0.0
    )

    exit_reasons = (
        trade_log["exit_reason"].value_counts().to_dict()
        if "exit_reason" in trade_log.columns else {}
    )

    return {
        "total_return"          : round(total_return, 2),
        "avg_trade_pnl"         : round(avg_trade, 2),
        "win_rate_pct"          : round(win_rate, 1),
        "num_trades"            : num_trades,
        "max_drawdown"          : round(max_drawdown, 2),
        "sharpe_ratio"          : round(sharpe, 3) if not np.isnan(sharpe) else None,
        "profit_factor"         : round(profit_factor, 3),
        "best_trade"            : round(best_trade, 2),
        "worst_trade"           : round(worst_trade, 2),
        "total_transaction_cost": round(total_cost, 2),
        "exit_reasons"          : exit_reasons,
    }


# ---------------------------------------------------------------------------
# Time-series helpers (for charts)
# ---------------------------------------------------------------------------

def equity_curve(trade_log: pd.DataFrame, pnl_col: str = "net_pnl_rupees") -> pd.Series:
    """Cumulative ₹ P&L, indexed by entry_date."""
    return (
        trade_log.set_index("entry_date")[pnl_col]
        .cumsum()
        .rename("equity_curve")
    )


def drawdown_series(trade_log: pd.DataFrame, pnl_col: str = "net_pnl_rupees") -> pd.Series:
    """Drawdown from peak, indexed by entry_date (all values ≤ 0)."""
    eq = equity_curve(trade_log, pnl_col)
    return (eq - eq.cummax()).rename("drawdown")


def metrics_table(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Return metrics as a two-column DataFrame (Metric / Value)."""
    m = compute_metrics(trade_log)
    rows = [(k, v) for k, v in m.items() if k != "exit_reasons"]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(1)
    n = 30
    fake_log = pd.DataFrame({
        "entry_date"        : pd.date_range("2023-01-01", periods=n, freq="4W"),
        "net_pnl_rupees"    : np.random.normal(5000, 15000, n),
        "transaction_cost"  : [3750] * n,
        "exit_reason"       : np.random.choice(["expiry", "stop_loss", "target"], n),
    })
    m = compute_metrics(fake_log)
    print("\nPerformance Metrics")
    print("=" * 40)
    for k, v in m.items():
        print(f"  {k:<30}: {v}")
