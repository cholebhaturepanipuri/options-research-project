"""
Black-Scholes Greeks: Delta, Gamma, Theta, Vega, Rho.

Conventions used throughout:
  - Theta returned as daily decay (divided by 365).
  - Vega returned per 1% move in vol (divided by 100).
  - Rho returned per 1% move in rate (divided by 100).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import Literal

from option_pricing import _d1, _d2


# ---------------------------------------------------------------------------
# Individual Greeks
# ---------------------------------------------------------------------------

def delta(
    S: float, K: float, r: float, sigma: float, T: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Delta — sensitivity of price to spot (dV/dS).

    Call delta : N(d1)         range [0, 1]
    Put delta  : N(d1) - 1     range [-1, 0]
    """
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else (0.5 if S == K else 0.0)
        return -1.0 if S < K else (-0.5 if S == K else 0.0)

    d1 = _d1(S, K, r, sigma, T)
    return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0


def gamma(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Gamma — rate of change of delta (d²V/dS²).

    Same for calls and puts.
    Gamma = n(d1) / (S * σ * √T)
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, r, sigma, T)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(
    S: float, K: float, r: float, sigma: float, T: float,
    option_type: Literal["call", "put"] = "call",
    annualised: bool = False,
) -> float:
    """
    Theta — rate of price decay over time (dV/dt).

    Negative for long options (time decay costs money).
    Returned as daily decay by default (divide by 365).
    """
    if T <= 0:
        return 0.0

    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(S, K, r, sigma, T)

    common = -(S * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))

    if option_type == "call":
        theta_ann = common - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta_ann = common + r * K * np.exp(-r * T) * norm.cdf(-d2)

    return theta_ann if annualised else theta_ann / 365.0


def vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Vega — sensitivity of price to volatility (dV/dσ).

    Same for calls and puts.
    Returned per 1% change in volatility (divide by 100).
    Vega_full = S * n(d1) * √T
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, r, sigma, T)
    return S * norm.pdf(d1) * np.sqrt(T) / 100.0


def rho(
    S: float, K: float, r: float, sigma: float, T: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Rho — sensitivity of price to risk-free rate (dV/dr).

    Returned per 1% change in rate (divide by 100).
    Call rho =  K*T*exp(-rT)*N(d2)
    Put  rho = -K*T*exp(-rT)*N(-d2)
    """
    if T <= 0:
        return 0.0
    d2 = _d2(S, K, r, sigma, T)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


# ---------------------------------------------------------------------------
# Convenience: all Greeks at once
# ---------------------------------------------------------------------------

def all_greeks(
    S: float, K: float, r: float, sigma: float, T: float,
    option_type: Literal["call", "put"] = "call",
) -> dict:
    """Return a dict of all five Greeks for a given option."""
    return {
        "delta": delta(S, K, r, sigma, T, option_type),
        "gamma": gamma(S, K, r, sigma, T),
        "theta": theta(S, K, r, sigma, T, option_type),
        "vega":  vega(S, K, r, sigma, T),
        "rho":   rho(S, K, r, sigma, T, option_type),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S, K, r, sigma, T = 19500, 19500, 0.065, 0.15, 0.25
    for ot in ("call", "put"):
        g = all_greeks(S, K, r, sigma, T, ot)
        print(f"\n--- {ot.upper()} ---")
        for name, val in g.items():
            print(f"  {name:<7}: {val:.6f}")

    # Put-call parity for delta:  delta_call - delta_put = 1  (approx, ignoring discounting)
    dc = delta(S, K, r, sigma, T, "call")
    dp = delta(S, K, r, sigma, T, "put")
    print(f"\nDelta(call) - Delta(put) = {dc - dp:.6f}  (should be ~1)")
