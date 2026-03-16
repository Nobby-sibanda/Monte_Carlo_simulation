"""
Monte Carlo Options Pricer — Core Module
=========================================
European Call & Put pricing via Geometric Brownian Motion simulation,
with Black-Scholes analytical comparison and full Greeks.

Fixes vs original:
  - bs_price now returns separate Greeks for calls AND puts
  - Edge-case guard when T <= 0 (at/past expiry)
  - monte_carlo uses numpy Generator (default_rng) instead of legacy random
  - Antithetic variates added for variance reduction
  - Reproducible runs via optional seed parameter
  - Division-by-zero guard in pct_diff utility
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ── Black-Scholes pricing + Greeks ────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    Black-Scholes analytical prices and Greeks for European options.

    Parameters
    ----------
    S     : current stock price
    K     : strike price
    T     : time to expiry in years (> 0)
    r     : continuous risk-free rate (decimal, e.g. 0.05 for 5%)
    sigma : annualised volatility (decimal, e.g. 0.20 for 20%)

    Returns
    -------
    dict with keys:
        call, put              – option prices
        d1, d2                 – intermediate values
        delta_call, delta_put  – price sensitivity to S
        gamma                  – delta sensitivity to S (same for call/put)
        theta_call, theta_put  – time decay per calendar day
        vega                   – price sensitivity to 1% move in sigma
        rho_call, rho_put      – price sensitivity to 1% move in r
    """
    # Guard: at or past expiry return intrinsic value, zero Greeks
    if T <= 0:
        call = float(max(S - K, 0.0))
        put  = float(max(K - S, 0.0))
        return dict(
            call=call, put=put, d1=0.0, d2=0.0,
            delta_call=1.0 if S > K else 0.0,
            delta_put=(-1.0 if S < K else 0.0),
            gamma=0.0, theta_call=0.0, theta_put=0.0,
            vega=0.0, rho_call=0.0, rho_put=0.0,
        )

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    disc = np.exp(-r * T)
    Nd1  = norm.cdf(d1)
    Nd2  = norm.cdf(d2)
    nd1  = norm.pdf(d1)

    call = S * Nd1 - K * disc * Nd2
    put  = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)

    delta_call = Nd1
    delta_put  = Nd1 - 1.0                                      # always negative
    gamma      = nd1 / (S * sigma * sqrt_T)

    # Theta: cost of one calendar day passing
    common_theta = -(S * nd1 * sigma) / (2.0 * sqrt_T)
    theta_call   = (common_theta - r * K * disc * Nd2)          / 365.0
    theta_put    = (common_theta + r * K * disc * norm.cdf(-d2)) / 365.0

    vega     = S * nd1 * sqrt_T * 0.01          # per 1% volatility
    rho_call = K * T * disc * Nd2 * 0.01        # per 1% rate
    rho_put  = -K * T * disc * norm.cdf(-d2) * 0.01

    return dict(
        call=call, put=put, d1=d1, d2=d2,
        delta_call=delta_call, delta_put=delta_put,
        gamma=gamma,
        theta_call=theta_call, theta_put=theta_put,
        vega=vega,
        rho_call=rho_call, rho_put=rho_put,
    )


# ── Monte Carlo simulation ─────────────────────────────────────────────────────

def monte_carlo(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_sims: int = 10_000,
    n_steps: int = 52,
    n_paths_plot: int = 20,
    antithetic: bool = True,
    seed: int | None = None,
) -> dict:
    """
    Simulate GBM paths and price European call/put options via Monte Carlo.

    Parameters
    ----------
    S, K, T, r, sigma : standard BSM parameters
    n_sims      : number of simulated paths (halved internally when antithetic=True)
    n_steps     : time steps per path (default 52 weekly steps)
    n_paths_plot: how many paths to return for visualisation
    antithetic  : use antithetic variates for variance reduction
    seed        : optional numpy random seed for reproducibility

    Returns
    -------
    dict with keys:
        mc_call, mc_put     – discounted mean payoff prices
        call_ci, put_ci     – 95% confidence interval half-widths
        terminal            – array of S_T values (n_sims,)
        sample_paths        – array of paths (n_paths_plot, n_steps+1)
        n_sims              – actual number of simulated paths
    """
    rng = np.random.default_rng(seed)

    dt       = T / n_steps
    discount = np.exp(-r * T)
    drift    = (r - 0.5 * sigma ** 2) * dt
    diff_std = sigma * np.sqrt(dt)

    if antithetic:
        half = n_sims // 2
        Z_half = rng.standard_normal((half, n_steps))
        Z = np.vstack([Z_half, -Z_half])          # antithetic pairs
    else:
        Z = rng.standard_normal((n_sims, n_steps))

    actual_sims = Z.shape[0]

    log_returns = np.cumsum(drift + diff_std * Z, axis=1)      # (n_sims, n_steps)
    path_prices = S * np.exp(log_returns)                       # (n_sims, n_steps)
    full_paths  = np.hstack([np.full((actual_sims, 1), S), path_prices])

    terminal = full_paths[:, -1]

    call_payoffs = np.maximum(terminal - K, 0.0)
    put_payoffs  = np.maximum(K - terminal, 0.0)

    mc_call = discount * call_payoffs.mean()
    mc_put  = discount * put_payoffs.mean()
    call_ci = 1.96 * discount * call_payoffs.std(ddof=1) / np.sqrt(actual_sims)
    put_ci  = 1.96 * discount * put_payoffs.std(ddof=1)  / np.sqrt(actual_sims)

    return dict(
        mc_call=mc_call,
        mc_put=mc_put,
        call_ci=call_ci,
        put_ci=put_ci,
        terminal=terminal,
        sample_paths=full_paths[:n_paths_plot, :],
        n_sims=actual_sims,
    )
