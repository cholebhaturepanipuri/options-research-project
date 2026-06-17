"""
Black-Scholes option pricing and implied volatility solver.

Formulas:
    d1 = [ln(S/K) + (r + σ²/2) * T] / (σ * √T)
    d2 = d1 - σ * √T
    Call = S*N(d1) - K*exp(-rT)*N(d2)
    Put  = K*exp(-rT)*N(-d2) - S*N(-d1)

IV solved via Brent's method (scipy.optimize.brentq).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from typing import Literal


# ---------------------------------------------------------------------------
# Core d1 / d2
# ---------------------------------------------------------------------------

def _d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
    return _d1(S, K, r, sigma, T) - sigma * np.sqrt(T)


# ---------------------------------------------------------------------------
# Black-Scholes price
# ---------------------------------------------------------------------------

def bs_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Black-Scholes European option price.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    r : float
        Risk-free rate, annualised decimal (e.g. 0.065 for 6.5%).
    sigma : float
        Volatility, annualised decimal (e.g. 0.20 for 20%).
    T : float
        Time to expiry in years.
    option_type : {"call", "put"}

    Returns
    -------
    float
        Theoretical option price.

    Examples
    --------
    >>> round(bs_price(19500, 19500, 0.065, 0.15, 0.25), 2)
    408.15
    """
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    if T <= 0:
        # At or past expiry — return intrinsic value
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(S, K, r, sigma, T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")


# ---------------------------------------------------------------------------
# Implied Volatility (Brent's method)
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: Literal["call", "put"] = "call",
    tol: float = 1e-6,
    sigma_low: float = 1e-4,
    sigma_high: float = 10.0,
) -> float:
    """
    Compute implied volatility using Brent's root-finding method.

    Returns np.nan if IV cannot be found (e.g. arbitrage violation or
    no root in [sigma_low, sigma_high]).

    Parameters
    ----------
    market_price : float
        Observed market (LTP) price of the option.
    S, K, r, T : float
        Spot, strike, rate, time to expiry (years).
    option_type : {"call", "put"}
    tol : float
        Convergence tolerance.
    sigma_low, sigma_high : float
        Search bracket for volatility.

    Returns
    -------
    float
        Implied volatility (annualised decimal), or np.nan on failure.

    Examples
    --------
    >>> round(implied_volatility(408.15, 19500, 19500, 0.065, 0.25), 4)
    0.15
    """
    if T <= 0:
        return np.nan

    # Arbitrage check: price must be >= intrinsic
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if market_price < intrinsic - 1e-3:
        return np.nan  # sub-intrinsic — likely bad data

    def objective(sigma: float) -> float:
        return bs_price(S, K, r, sigma, T, option_type) - market_price

    try:
        # brentq requires sign change at bracket endpoints
        if objective(sigma_low) * objective(sigma_high) > 0:
            return np.nan
        iv = brentq(objective, sigma_low, sigma_high, xtol=tol)
        return float(iv)
    except (ValueError, RuntimeError):
        return np.nan


# ---------------------------------------------------------------------------
# Vectorised helpers (for DataFrame operations)
# ---------------------------------------------------------------------------

def iv_series(
    prices: "pd.Series",  # noqa: F821
    S: "pd.Series",
    K: "pd.Series",
    r: float,
    T: "pd.Series",
    option_type: "pd.Series",
) -> "pd.Series":  # noqa: F821
    """Apply implied_volatility row-by-row on a DataFrame.

    option_type values should be 'call' or 'put' (lowercase).
    """
    import pandas as pd

    result = pd.Series(index=prices.index, dtype=float)
    for idx in prices.index:
        ot = str(option_type.loc[idx]).lower()
        if ot not in ("call", "put"):
            result.loc[idx] = np.nan
            continue
        result.loc[idx] = implied_volatility(
            market_price=prices.loc[idx],
            S=float(S.loc[idx]),
            K=float(K.loc[idx]),
            r=r,
            T=float(T.loc[idx]),
            option_type=ot,
        )
    return result


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S, K, r, sigma, T = 19500, 19500, 0.065, 0.15, 0.25
    call = bs_price(S, K, r, sigma, T, "call")
    put  = bs_price(S, K, r, sigma, T, "put")
    print(f"Call price : {call:.2f}")
    print(f"Put  price : {put:.2f}")

    # Put-call parity check
    pcp_lhs = call - put
    pcp_rhs = S - K * np.exp(-r * T)
    print(f"Put-call parity error: {abs(pcp_lhs - pcp_rhs):.8f}  (should be ~0)")

    # Round-trip IV check
    iv = implied_volatility(call, S, K, r, T, "call")
    print(f"Recovered IV: {iv:.4f}  (should be {sigma})")
