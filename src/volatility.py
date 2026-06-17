"""
Volatility estimators and IV-RV analysis utilities.

Estimators implemented:
  - Close-to-close (standard)
  - Parkinson (1980)  — uses high/low, more efficient under GBM
  - Yang-Zhang (2000) — uses OHLC, most accurate standard estimator

All outputs are annualised volatility in decimal form (0.15 = 15% vol).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252   # annualisation factor


# ---------------------------------------------------------------------------
# Realised volatility estimators
# ---------------------------------------------------------------------------

def close_to_close_rv(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Standard close-to-close realised volatility.

    σ = std(log returns) * √252,  rolled over `window` days.

    Parameters
    ----------
    close  : pd.Series   Daily closing prices, date-indexed.
    window : int         Rolling window in trading days.

    Returns
    -------
    pd.Series   Annualised RV (decimal).
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)


def parkinson_rv(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson (1980) high-low range estimator.

    More efficient than close-to-close when there is no overnight gap.
    σ² ≈ (1 / 4ln2) * E[(ln H/L)²]

    Parameters
    ----------
    high, low : pd.Series   Daily high and low prices.
    window    : int         Rolling window.

    Returns
    -------
    pd.Series   Annualised RV (decimal).
    """
    log_hl = np.log(high / low)
    factor = 1.0 / (4.0 * np.log(2.0))
    var = factor * (log_hl ** 2).rolling(window).mean()
    return np.sqrt(var * TRADING_DAYS)


def yang_zhang_rv(
    open_: pd.Series,
    high : pd.Series,
    low  : pd.Series,
    close: pd.Series,
    window: int = 20,
    k: float = 0.34,
) -> pd.Series:
    """
    Yang-Zhang (2000) OHLC estimator.

    Combines three components:
      1. Overnight (close-to-open) variance.
      2. Open-to-close (intraday) variance.
      3. Rogers-Satchell variance.

    k controls the weight between components 1 and 2
    (default 0.34 is optimal under GBM).

    Parameters
    ----------
    open_, high, low, close : pd.Series   OHLC price series.
    window : int   Rolling window.
    k      : float Weight parameter.

    Returns
    -------
    pd.Series   Annualised RV (decimal).
    """
    n = window
    # Overnight return (close[t-1] → open[t])
    log_oc = np.log(open_ / close.shift(1))
    # Open-to-close return
    log_co = np.log(close / open_)
    # Rogers-Satchell term
    log_ho = np.log(high  / open_)
    log_lo = np.log(low   / open_)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    # Optimal weight (minimises variance of estimator)
    alpha = k / (k + (n + 1) / (n - 1))

    var_overnight = log_oc.rolling(n).var(ddof=1)
    var_oc        = log_co.rolling(n).var(ddof=1)
    var_rs        = rs.rolling(n).mean()

    sigma2 = var_overnight + alpha * var_oc + (1.0 - alpha) * var_rs
    return np.sqrt(sigma2 * TRADING_DAYS)


# ---------------------------------------------------------------------------
# ATM IV extraction from an options chain
# ---------------------------------------------------------------------------

def atm_implied_vol(
    chain_df   : pd.DataFrame,
    spot       : float,
    r          : float,
    T          : float,
    price_col  : str = "close",
    strike_col : str = "strike",
    type_col   : str = "option_type",
    ce_label   : str = "CE",
    pe_label   : str = "PE",
) -> float:
    """
    Extract ATM implied volatility from a single-day options chain.

    Averages the IV of the closest-to-ATM call and put.

    Parameters
    ----------
    chain_df   : pd.DataFrame   One day's option chain (already filtered for one expiry).
    spot       : float          Spot/index price that day.
    r          : float          Risk-free rate (decimal).
    T          : float          Time to expiry in years.
    price_col  : str            Column name for option prices.
    strike_col : str            Column name for strikes.
    type_col   : str            Column name for option type.
    ce_label, pe_label : str    Labels used in type_col for CE/PE.

    Returns
    -------
    float   Average ATM IV (decimal), or np.nan on failure.
    """
    from option_pricing import implied_volatility

    def closest_row(df: pd.DataFrame) -> pd.Series | None:
        if df.empty:
            return None
        df = df.copy()
        df["_dist"] = (df[strike_col] - spot).abs()
        return df.loc[df["_dist"].idxmin()]

    calls = chain_df[chain_df[type_col] == ce_label]
    puts  = chain_df[chain_df[type_col] == pe_label]

    call_row = closest_row(calls)
    put_row  = closest_row(puts)

    ivs = []
    for row, ot in [(call_row, "call"), (put_row, "put")]:
        if row is None:
            continue
        prc = row[price_col]
        k   = row[strike_col]
        if pd.isna(prc) or prc <= 0 or pd.isna(k):
            continue
        iv = implied_volatility(float(prc), float(spot), float(k), r, float(T), ot)
        if not np.isnan(iv):
            ivs.append(iv)

    return float(np.mean(ivs)) if ivs else np.nan


# ---------------------------------------------------------------------------
# IV-RV spread and summary
# ---------------------------------------------------------------------------

def iv_rv_spread(iv: pd.Series, rv: pd.Series) -> pd.Series:
    """IV minus RV — the volatility risk premium proxy."""
    return (iv - rv).rename("iv_rv_spread")


def iv_rv_summary(iv: pd.Series, rv: pd.Series) -> pd.DataFrame:
    """
    Combine IV, RV, spread, and a flag for IV > RV into one DataFrame.
    """
    spread = iv_rv_spread(iv, rv)
    return pd.DataFrame({
        "iv"          : iv,
        "rv"          : rv,
        "iv_rv_spread": spread,
        "iv_above_rv" : (spread > 0).astype(int),
    }).dropna()


def vix_percentile(vix: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling percentile rank of VIX (proxy for regime filter).
    High percentile → elevated vol regime → potentially worse for short-vol.
    """
    return vix.rolling(window).rank(pct=True)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = pd.Series(19000 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))), index=dates)
    high  = close * np.exp(np.abs(np.random.normal(0, 0.005, n)))
    low   = close * np.exp(-np.abs(np.random.normal(0, 0.005, n)))
    open_ = close.shift(1).fillna(close.iloc[0])

    ctc  = close_to_close_rv(close, 20)
    park = parkinson_rv(high, low, 20)
    yz   = yang_zhang_rv(open_, high, low, close, 20)

    print("Close-to-close RV (last 5):")
    print(ctc.tail())
    print("\nParkinson RV (last 5):")
    print(park.tail())
    print("\nYang-Zhang RV (last 5):")
    print(yz.tail())
