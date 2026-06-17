"""
Plotting utilities using matplotlib.

All functions return a matplotlib Figure so they can be embedded in
notebooks (plt.show()) or in Streamlit (st.pyplot(fig)).
"""
from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from typing import Optional, Tuple

# Add src/ to path so imports work when run directly
sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Payoff diagram
# ---------------------------------------------------------------------------

def plot_payoff(
    strategy,
    spot_range : Optional[Tuple[float, float]] = None,
    title      : Optional[str] = None,
    figsize    : Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot P&L payoff diagram at expiry for a Strategy object.

    Parameters
    ----------
    strategy   : strategies.Strategy
    spot_range : (min_spot, max_spot). Auto-derived if None.
    title      : chart title.

    Returns
    -------
    matplotlib.Figure
    """
    if not strategy.legs:
        raise ValueError("Strategy has no legs.")

    strikes = [leg.strike for leg in strategy.legs]
    min_k, max_k = min(strikes), max(strikes)
    centre = (min_k + max_k) / 2.0

    if spot_range is None:
        width = max(centre * 0.30, max_k - min_k + 300)
        spot_range = (max(centre - width, 0.0), centre + width)

    spots = np.linspace(spot_range[0], spot_range[1], 1000)
    pnl   = strategy.payoff(spots)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(spots, pnl, color="steelblue", linewidth=2.5, label="P&L at expiry")
    ax.axhline(0, color="dimgray", linewidth=1.0, linestyle="--")

    ax.fill_between(spots, pnl, 0, where=(pnl >= 0), alpha=0.12, color="green", label="Profit zone")
    ax.fill_between(spots, pnl, 0, where=(pnl < 0),  alpha=0.12, color="red",   label="Loss zone")

    for bp in strategy.breakevens():
        ax.axvline(bp, color="darkorange", linestyle=":", linewidth=1.5)
        ax.annotate(
            f"BE {bp:,.0f}", xy=(bp, 0),
            xytext=(bp, (pnl.max() - pnl.min()) * 0.08),
            fontsize=9, color="darkorange", ha="center",
        )

    for leg in strategy.legs:
        ax.axvline(leg.strike, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Spot Price at Expiry", fontsize=11)
    ax.set_ylabel("P&L (₹)", fontsize=11)
    ax.set_title(title or strategy.name, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Greeks vs spot
# ---------------------------------------------------------------------------

def plot_greeks(
    K      : float,
    r      : float,
    sigma  : float,
    T      : float,
    S_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (14, 9),
) -> plt.Figure:
    """
    Plot all five Greeks vs spot for both call and put.

    Parameters
    ----------
    K, r, sigma, T : Black-Scholes parameters.
    S_range         : (min_spot, max_spot). Defaults to 70%–130% of K.
    """
    from greeks import delta, gamma, theta, vega, rho

    if S_range is None:
        S_range = (K * 0.70, K * 1.30)
    spots = np.linspace(S_range[0], S_range[1], 300)

    greek_fns = {
        "Delta" : lambda S, ot: delta(S, K, r, sigma, T, ot),
        "Gamma" : lambda S, _:  gamma(S, K, r, sigma, T),
        "Theta" : lambda S, ot: theta(S, K, r, sigma, T, ot),
        "Vega"  : lambda S, _:  vega(S, K, r, sigma, T),
        "Rho"   : lambda S, ot: rho(S, K, r, sigma, T, ot),
    }

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes_flat = axes.flatten()

    for i, (name, fn) in enumerate(greek_fns.items()):
        ax = axes_flat[i]
        ax.plot(spots, [fn(s, "call") for s in spots], color="steelblue", linewidth=2, label="Call")
        ax.plot(spots, [fn(s, "put")  for s in spots], color="tomato",    linewidth=2, label="Put")
        ax.axvline(K, color="gray", linestyle="--", linewidth=1, alpha=0.6, label=f"K={K:,.0f}")
        ax.axhline(0, color="dimgray", linewidth=0.7)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Spot", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    axes_flat[-1].set_visible(False)
    fig.suptitle(f"Greeks vs Spot  (K={K:,.0f}, σ={sigma:.0%}, T={T:.2f}yr, r={r:.1%})",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# IV vs RV
# ---------------------------------------------------------------------------

def plot_iv_rv(
    iv     : pd.Series,
    rv     : pd.Series,
    title  : str = "IV vs Realised Volatility",
    figsize: Tuple[int, int] = (12, 7),
) -> plt.Figure:
    """
    Two-panel chart: IV & RV lines (with spread shading) + IV−RV bar chart.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(iv.index, iv * 100, label="Implied Vol (IV)",    color="steelblue", linewidth=1.8)
    ax1.plot(rv.index, rv * 100, label="Realised Vol (RV)",   color="tomato",    linewidth=1.8)
    ax1.fill_between(iv.index, iv * 100, rv * 100,
                     where=(iv >= rv), alpha=0.15, color="green", label="IV > RV (vol carry)")
    ax1.fill_between(iv.index, iv * 100, rv * 100,
                     where=(iv <  rv), alpha=0.15, color="red",   label="RV > IV")
    ax1.set_ylabel("Volatility (%)", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    spread = (iv - rv) * 100
    colours = ["steelblue" if x >= 0 else "tomato" for x in spread]
    ax2.bar(spread.index, spread, color=colours, alpha=0.75, width=1.5)
    ax2.axhline(0, color="black", linewidth=0.9)
    ax2.set_ylabel("IV − RV (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Equity curve + drawdown
# ---------------------------------------------------------------------------

def plot_equity_curve(
    trade_log: pd.DataFrame,
    title    : str = "Strategy Equity Curve",
    figsize  : Tuple[int, int] = (12, 7),
) -> plt.Figure:
    """
    Two-panel chart: cumulative P&L (top) + drawdown (bottom).
    """
    from metrics import equity_curve, drawdown_series

    eq = equity_curve(trade_log)
    dd = drawdown_series(trade_log)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(eq.index, eq, color="steelblue", linewidth=2, label="Cumulative P&L")
    ax1.fill_between(eq.index, eq, 0, where=(eq >= 0), alpha=0.12, color="green")
    ax1.fill_between(eq.index, eq, 0, where=(eq <  0), alpha=0.12, color="red")
    ax1.set_ylabel("Cumulative P&L (₹)", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9)

    ax2.fill_between(dd.index, dd, 0, color="tomato", alpha=0.75)
    ax2.set_ylabel("Drawdown (₹)", fontsize=11)
    ax2.set_xlabel("Entry Date", fontsize=11)
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Delta hedging P&L breakdown
# ---------------------------------------------------------------------------

def plot_delta_hedge(hedge_df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot option P&L, hedge P&L, and total P&L from a delta-hedging simulation.

    Expects hedge_df to have columns: date, option_pnl, hedge_pnl, total_pnl.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(hedge_df["date"], hedge_df["option_pnl"], label="Option P&L",  color="steelblue", linewidth=1.8)
    ax.plot(hedge_df["date"], hedge_df["hedge_pnl"],  label="Hedge P&L",   color="darkorange", linewidth=1.8)
    ax.plot(hedge_df["date"], hedge_df["total_pnl"],  label="Total P&L",   color="green", linewidth=2.2, linestyle="--")
    ax.axhline(0, color="dimgray", linewidth=0.8)
    ax.set_title("Delta-Hedging Simulation P&L", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("P&L (₹)", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig
