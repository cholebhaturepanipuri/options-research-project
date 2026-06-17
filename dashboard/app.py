"""
Options Research Dashboard — Streamlit app.

Run with:
    streamlit run dashboard/app.py

Expects the processed options data at data/processed/nifty_options.csv
and spot data at data/processed/nifty_spot.csv.
Gracefully degrades to synthetic data if real data is not yet available.
"""
from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from option_pricing import bs_price, implied_volatility
from greeks          import all_greeks
from strategies      import (
    short_straddle, short_strangle, long_straddle, long_strangle,
    bull_call_spread, bear_put_spread, long_call, long_put,
)
from volatility      import close_to_close_rv, parkinson_rv, iv_rv_summary
from backtester      import Backtester
from metrics         import compute_metrics, equity_curve, drawdown_series, metrics_table
from plots           import (
    plot_payoff, plot_greeks, plot_iv_rv,
    plot_equity_curve, plot_delta_hedge,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Options Research Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 NIFTY Options Research Dashboard")
st.caption("Academic research project — not investment advice.")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = [
    "Payoff Visualiser",
    "Greeks Explorer",
    "IV vs Realised Volatility",
    "Strategy Backtest",
    "Data Overview",
]

page = st.sidebar.radio("Navigate to", PAGES)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Options Research  \n**Intern:** Pratham Hari")


# ---------------------------------------------------------------------------
# Shared parameters (sidebar)
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Model parameters")
spot   = st.sidebar.number_input("Spot (S)",      value=19500, step=50)
strike = st.sidebar.number_input("Strike (K)",     value=19500, step=50)
sigma  = st.sidebar.slider("Volatility σ (%)",    min_value=5,  max_value=80, value=15) / 100
r      = st.sidebar.slider("Risk-free rate r (%)", min_value=1, max_value=12, value=7) / 100
T_days = st.sidebar.slider("Days to expiry",       min_value=1, max_value=90,  value=30)
T      = T_days / 365.0


# ---------------------------------------------------------------------------
# Helper: load data (real or synthetic)
# ---------------------------------------------------------------------------

@st.cache_data
def load_options_data() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "nifty_options.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date", "expiry_date"])
        return df
    return pd.DataFrame()


@st.cache_data
def load_spot_data() -> pd.Series:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "nifty_spot.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"])
        return df.set_index("Date")["Close"].rename("spot")
    # Synthetic fallback
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="B")
    prices = 18000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, len(dates))))
    return pd.Series(prices, index=dates, name="spot")


# ===========================================================================
# PAGE 1 — Payoff Visualiser
# ===========================================================================

if page == "Payoff Visualiser":
    st.header("Option Strategy Payoff at Expiry")

    col1, col2 = st.columns([1, 2])

    with col1:
        strategy_name = st.selectbox("Strategy", [
            "Long Call", "Long Put", "Short Call", "Short Put",
            "Bull Call Spread", "Bear Put Spread",
            "Long Straddle", "Short Straddle",
            "Long Strangle", "Short Strangle",
        ])

        call_prem = st.number_input("Call premium (₹)", value=200, step=10)
        put_prem  = st.number_input("Put premium (₹)",  value=190, step=10)
        otm_dist  = st.number_input("OTM distance (strangle, ₹)", value=200, step=50)

        # Build strategy
        K2 = strike
        s_map = {
            "Long Call"        : lambda: long_call(K2, call_prem),
            "Long Put"         : lambda: long_put(K2, put_prem),
            "Short Call"       : lambda: long_call(K2, call_prem).__class__("Short Call",
                                    [__import__("strategies").Leg("call", K2, call_prem, "short")]),
            "Short Put"        : lambda: __import__("strategies").short_put(K2, put_prem),
            "Bull Call Spread" : lambda: bull_call_spread(K2, K2 + otm_dist, call_prem, call_prem * 0.5),
            "Bear Put Spread"  : lambda: bear_put_spread(K2, K2 - otm_dist, put_prem, put_prem * 0.5),
            "Long Straddle"    : lambda: long_straddle(K2, call_prem, put_prem),
            "Short Straddle"   : lambda: short_straddle(K2, call_prem, put_prem),
            "Long Strangle"    : lambda: long_strangle(K2 + otm_dist, K2 - otm_dist, call_prem * 0.5, put_prem * 0.5),
            "Short Strangle"   : lambda: short_strangle(K2 + otm_dist, K2 - otm_dist, call_prem * 0.5, put_prem * 0.5),
        }
        strat = s_map[strategy_name]()

        summary = strat.summary()
        st.markdown("### Strategy summary")
        st.metric("Net Premium",  f"₹{summary['net_premium']:,.0f}")
        st.metric("Max Profit",   str(summary["max_profit"]))
        st.metric("Max Loss",     str(summary["max_loss"]))
        bps = summary["breakevens"]
        st.write("**Breakevens:**", ", ".join(f"₹{b:,.0f}" for b in bps) if bps else "None")

    with col2:
        fig = plot_payoff(strat, title=f"{strategy_name} Payoff Diagram")
        st.pyplot(fig)
        plt.close(fig)


# ===========================================================================
# PAGE 2 — Greeks Explorer
# ===========================================================================

elif page == "Greeks Explorer":
    st.header("Black-Scholes Greeks Explorer")

    bs_call = bs_price(spot, strike, r, sigma, T, "call")
    bs_put  = bs_price(spot, strike, r, sigma, T, "put")
    g_call  = all_greeks(spot, strike, r, sigma, T, "call")
    g_put   = all_greeks(spot, strike, r, sigma, T, "put")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Call — ₹{bs_call:,.2f}")
        for k, v in g_call.items():
            st.metric(k.capitalize(), f"{v:.5f}")
    with col2:
        st.subheader(f"Put — ₹{bs_put:,.2f}")
        for k, v in g_put.items():
            st.metric(k.capitalize(), f"{v:.5f}")

    st.markdown("---")
    st.subheader("Greeks vs Spot")
    fig = plot_greeks(strike, r, sigma, T, S_range=(spot * 0.75, spot * 1.25))
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Implied Volatility solver")
    mkt_price = st.number_input("Market price to back-solve IV from", value=float(round(bs_call, 2)), step=5.0)
    ot_iv     = st.radio("Option type", ["call", "put"], horizontal=True)
    iv_solved = implied_volatility(mkt_price, spot, strike, r, T, ot_iv)
    if np.isnan(iv_solved):
        st.error("Could not solve IV — check inputs for arbitrage violations.")
    else:
        st.success(f"Implied Volatility: **{iv_solved:.2%}**")


# ===========================================================================
# PAGE 3 — IV vs RV
# ===========================================================================

elif page == "IV vs Realised Volatility":
    st.header("Implied Volatility vs Realised Volatility")

    spot_series = load_spot_data()

    if spot_series.empty:
        st.warning("No spot data found. Using synthetic data.")

    window = st.slider("RV window (trading days)", 5, 30, 20)
    rv_method = st.radio("RV estimator", ["Close-to-Close", "Parkinson"], horizontal=True)

    close = spot_series
    if rv_method == "Close-to-Close":
        rv = close_to_close_rv(close, window)
    else:
        # Parkinson needs high/low; approximate with ±0.5% noise
        high = close * 1.005
        low  = close * 0.995
        rv   = parkinson_rv(high, low, window)

    # Synthetic IV as proxy (India VIX / 100 shifted slightly)
    iv_proxy = rv.rolling(5).mean().shift(-3) * 1.2 + 0.02
    iv_proxy = iv_proxy.clip(0.05, 0.80)

    df_iv_rv = iv_rv_summary(iv_proxy.dropna(), rv.dropna()).dropna()

    if df_iv_rv.empty:
        st.warning("Not enough data for IV-RV analysis.")
    else:
        pct_iv_above = df_iv_rv["iv_above_rv"].mean() * 100
        avg_spread   = df_iv_rv["iv_rv_spread"].mean() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("IV > RV frequency", f"{pct_iv_above:.1f}%")
        c2.metric("Avg IV−RV spread",  f"{avg_spread:.1f}%")
        c3.metric("Data points",       len(df_iv_rv))

        fig = plot_iv_rv(df_iv_rv["iv"], df_iv_rv["rv"])
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "📌 **Note:** IV here is a synthetic proxy (smoothed RV × 1.2). "
            "Replace with actual ATM IV computed from bhavcopy data once the "
            "data pipeline is complete."
        )


# ===========================================================================
# PAGE 4 — Strategy Backtest
# ===========================================================================

elif page == "Strategy Backtest":
    st.header("Strategy Backtest Results")

    options_df = load_options_data()

    if options_df.empty:
        st.warning(
            "No processed options data found at `data/processed/nifty_options.csv`. "
            "Complete the data pipeline (notebook 03) first, then re-run the dashboard."
        )
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        strategy_choice = st.selectbox("Strategy", ["Short Straddle", "Short Strangle"])
    with col2:
        sl_mult = st.slider("Stop-loss multiplier", 1.0, 3.0, 1.5, 0.1)
    with col3:
        tgt = st.slider("Premium capture target (%)", 0, 100, 50)
        tgt = tgt / 100.0 if tgt > 0 else None

    otm_d = st.slider("OTM distance (strangle only, ₹)", 50, 300, 100, 25)

    if st.button("Run Backtest"):
        with st.spinner("Running backtest…"):
            bt = Backtester(
                options_df,
                stop_loss_multiplier = sl_mult,
                target_capture       = tgt,
                otm_distance         = otm_d,
            )
            if strategy_choice == "Short Straddle":
                bt.run_short_straddle()
            else:
                bt.run_short_strangle()

            log = bt.trade_log()

        if log.empty:
            st.error("No trades generated — check your data.")
        else:
            m = compute_metrics(log)

            st.subheader("Performance Metrics")
            mc = st.columns(5)
            mc[0].metric("Total Return (₹)",  f"₹{m['total_return']:,.0f}")
            mc[1].metric("Win Rate",           f"{m['win_rate_pct']:.1f}%")
            mc[2].metric("Sharpe Ratio",       f"{m.get('sharpe_ratio', 'N/A')}")
            mc[3].metric("Max Drawdown (₹)",   f"₹{m['max_drawdown']:,.0f}")
            mc[4].metric("Profit Factor",      f"{m['profit_factor']}")

            fig_eq = plot_equity_curve(log, title=f"{strategy_choice} Equity Curve")
            st.pyplot(fig_eq)
            plt.close(fig_eq)

            st.subheader("Trade Log")
            st.dataframe(log, use_container_width=True)

            st.subheader("Exit Reason Breakdown")
            exit_df = pd.Series(m["exit_reasons"]).rename("count").reset_index()
            exit_df.columns = ["Reason", "Count"]
            st.bar_chart(exit_df.set_index("Reason"))


# ===========================================================================
# PAGE 5 — Data Overview
# ===========================================================================

elif page == "Data Overview":
    st.header("Data Overview")

    options_df  = load_options_data()
    spot_series = load_spot_data()

    st.subheader("Spot / Index Data")
    if not spot_series.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Date range",   f"{spot_series.index.min().date()} → {spot_series.index.max().date()}")
        c2.metric("Observations", len(spot_series))
        c3.metric("Latest close", f"₹{spot_series.iloc[-1]:,.2f}")
        st.line_chart(spot_series)
    else:
        st.info("Spot data not loaded.")

    st.subheader("Options Data")
    if not options_df.empty:
        from data_loader import data_quality_report
        rpt = data_quality_report(options_df)
        for k, v in rpt.items():
            st.write(f"**{k}:** {v}")
        st.dataframe(options_df.head(50), use_container_width=True)
    else:
        st.info(
            "No processed options data found. "
            "Download NSE F&O bhavcopy files, run `data_loader.py`, "
            "and save the output to `data/processed/nifty_options.csv`."
        )
