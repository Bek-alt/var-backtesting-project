import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats

# ── 1. Configuration ──────────────────────────────────────────────
TICKERS  = ['AAPL', 'JPM', 'MSFT', 'GS', 'XOM']   # 5 stocks
WEIGHTS  = np.array([0.25, 0.25, 0.20, 0.15, 0.15]) # must sum to 1
START    = '2018-01-01'
END      = '2024-01-01'
CONF     = 0.95        # 95% confidence level
HORIZON  = 1           # 1-day VaR
# ── 2. Download prices ────────────────────────────────────────────
print("Downloading data...")
raw   = yf.download(TICKERS, start=START, end=END)['Close']
raw   = raw.dropna()

# ── 3. Daily log returns ──────────────────────────────────────────
returns = np.log(raw / raw.shift(1)).dropna()

# ── 4. Weighted portfolio return (single series) ──────────────────
port_returns = returns.dot(WEIGHTS)

print(f"Data loaded: {len(port_returns)} trading days")
print(f"Date range : {port_returns.index[0].date()} → {port_returns.index[-1].date()}")
print(f"Mean daily return : {port_returns.mean():.4f}")
print(f"Daily volatility  : {port_returns.std():.4f}")

# METHOD 1 — HISTORICAL SIMULATION
# This method makes no assumptions about the distribution of returns.
# Idea: take the last 250 days of actual returns, sort them,
# and read off the 5th percentile. No assumptions about distribution.

WINDOW = 250  # 1 trading year

def historical_var(returns_series, window, confidence):
    """
    Rolls a window across the return series.
    At each day, sorts the past 'window' returns and takes
    the (1-confidence) percentile as the VaR estimate.
    Returns a Series aligned to the original index.
    """
    var_series = pd.Series(index=returns_series.index, dtype=float)

    for i in range(window, len(returns_series)):
        window_returns = returns_series.iloc[i - window:i]
        var_series.iloc[i] = np.percentile(window_returns, (1 - confidence) * 100)

    return var_series.dropna()

hs_var = historical_var(port_returns, WINDOW, CONF)

print("\n── Historical Simulation VaR ──")
print(f"Average VaR (95%): {hs_var.mean():.4f}")
print(f"Worst VaR day    : {hs_var.min():.4f}")
# ══════════════════════════════════════════════════════════════════
# METHOD 2 — VARIANCE-COVARIANCE (parametric)
# ══════════════════════════════════════════════════════════════════
# Idea: assume returns are normally distributed.
# VaR = z-score × portfolio standard deviation
# z-score for 95% = -1.645

def varcov_var(returns_df, weights, window, confidence):
    """
    At each day, compute the rolling covariance matrix of individual
    stock returns, then derive portfolio variance using w' Σ w.
    Portfolio std × z-score gives the parametric VaR.
    """
    z = stats.norm.ppf(1 - confidence)   
    var_series = pd.Series(index=returns_df.index, dtype=float)

    for i in range(window, len(returns_df)):
        window_ret = returns_df.iloc[i - window:i]
        cov_matrix = window_ret.cov().values          
        port_var   = weights @ cov_matrix @ weights   
        port_std   = np.sqrt(port_var)               
        var_series.iloc[i] = z * port_std             

    return var_series.dropna()

vc_var = varcov_var(returns, WEIGHTS, WINDOW, CONF)

print("\n── Variance-Covariance VaR ──")
print(f"Average VaR (95%): {vc_var.mean():.4f}")
print(f"Worst VaR day    : {vc_var.min():.4f}")

# 
# METHOD 3 — MONTE CARLO SIMULATION
# 
# Idea: simulate 10,000 possible tomorrows using correlated random
# paths, then take the 5th percentile of simulated P&L.
# Uses Cholesky decomposition to preserve stock correlations.

N_SIMULATIONS = 10_000

def monte_carlo_var(returns_df, weights, window, confidence, n_sims):
    """
    For each day, estimate mean & covariance from the past window.
    Cholesky-decompose the covariance matrix to generate correlated
    normal shocks. Compute portfolio return for each simulation.
    VaR = percentile of simulated portfolio returns.
    """
    var_series = pd.Series(index=returns_df.index, dtype=float)

    for i in range(window, len(returns_df)):
        window_ret = returns_df.iloc[i - window:i]
        mu         = window_ret.mean().values          # expected returns
        cov        = window_ret.cov().values           # covariance Σ

        # Cholesky: decompose Σ = L L' so we can generate correlated shocks
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            continue  # skip if matrix not positive definite

        # Generate uncorrelated standard normals, then correlate them
        Z            = np.random.standard_normal((n_sims, len(weights)))
        corr_shocks  = Z @ L.T                          # shape: (10000, 5)

        # Add mean drift to each simulation
        sim_returns  = mu + corr_shocks                 # (10000, 5)

        # Portfolio return for each simulation
        port_sim     = sim_returns @ weights             # (10000,)

        var_series.iloc[i] = np.percentile(port_sim, (1 - confidence) * 100)

    return var_series.dropna()

print("\nRunning Monte Carlo (this takes ~30 seconds)...")
mc_var = monte_carlo_var(returns, WEIGHTS, WINDOW, CONF, N_SIMULATIONS)

print("── Monte Carlo VaR ──")
print(f"Average VaR (95%): {mc_var.mean():.4f}")
print(f"Worst VaR day    : {mc_var.min():.4f}")

# ══════════════════════════════════════════════════════════════════
# COMPARISON PLOT
# ══════════════════════════════════════════════════════════════════

# Align all three series to the same date index
common_idx = hs_var.index.intersection(vc_var.index).intersection(mc_var.index)
actual     = port_returns.loc[common_idx]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# ── Top panel: VaR comparison ─────────────────────────────────────
ax1.plot(common_idx, hs_var.loc[common_idx], label='Historical Simulation',
         color='steelblue', linewidth=1.2)
ax1.plot(common_idx, vc_var.loc[common_idx], label='Variance-Covariance',
         color='darkorange', linewidth=1.2)
ax1.plot(common_idx, mc_var.loc[common_idx], label='Monte Carlo',
         color='green', linewidth=1.0, alpha=0.8, linestyle='--')
ax1.set_ylabel('VaR (log return)')
ax1.set_title('95% 1-Day VaR — Three Methods Compared')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── Bottom panel: actual returns vs VaR breaches ──────────────────
ax2.plot(common_idx, actual, color='gray', linewidth=0.8,
         alpha=0.7, label='Actual portfolio return')
ax2.plot(common_idx, hs_var.loc[common_idx], color='steelblue',
         linewidth=1.2, label='HS VaR (95%)')

# Highlight breach days (actual loss worse than VaR)
breaches = actual[actual < hs_var.loc[common_idx]]
ax2.scatter(breaches.index, breaches.values,
            color='red', s=15, zorder=5, label=f'VaR breaches ({len(breaches)} days)')

ax2.set_ylabel('Portfolio return')
ax2.set_title('Actual Returns vs Historical Simulation VaR — Breaches in Red')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('var_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as var_comparison.png")
# ── Cell 8: Kupiec Proportion of Failures (POF) test ─────────────
# H0: true exception rate = 1 - confidence = 5%
# H1: true exception rate != 5%
#
# Test statistic is a likelihood ratio:
# LR_POF = -2 × ln[ p0^x × (1-p0)^(T-x) / p_hat^x × (1-p_hat)^(T-x) ]
#
# where:
#   p0    = expected exception rate (0.05)
#   p_hat = observed exception rate (x/T)
#   x     = number of exceptions
#   T     = total days
#
# Under H0, LR_POF ~ chi-squared with 1 degree of freedom
# Critical value at 95% confidence = 3.841

from scipy import stats
import numpy as np

def kupiec_test(exceptions, T, confidence=0.95):
    """
    Kupiec proportion of failures test.
    Returns test statistic, p-value, and pass/fail.
    """
    p0    = 1 - confidence          # expected rate = 0.05
    x     = exceptions              # number of exceptions
    p_hat = x / T                   # observed rate

    # Edge case: if no exceptions or all exceptions, test breaks
    if x == 0 or x == T:
        return None, None, None

    # Log likelihood ratio statistic
    lr_stat = -2 * (
        x * np.log(p0 / p_hat) +
        (T - x) * np.log((1 - p0) / (1 - p_hat))
    )

    # p-value from chi-squared distribution with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    # Reject H0 if p-value < 0.05 (model fails the test)
    result  = "PASS" if p_value > 0.05 else "FAIL"

    return lr_stat, p_value, result

hs_lr, hs_pval, hs_kup = kupiec_test(hs_count, T)
vc_lr, vc_pval, vc_kup = kupiec_test(vc_count, T)
mc_lr, mc_pval, mc_kup = kupiec_test(mc_count, T)

print("── Kupiec Test (H0: exception rate = 5%) ──")
print(f"{'Model':<28} {'Stat':>8} {'p-value':>10} {'Result':>8}")
print("-" * 56)
print(f"{'Historical Simulation':<28} {hs_lr:>8.3f} {hs_pval:>10.4f} {hs_kup:>8}")
print(f"{'Variance-Covariance':<28} {vc_lr:>8.3f} {vc_pval:>10.4f} {vc_kup:>8}")
print(f"{'Monte Carlo':<28} {mc_lr:>8.3f} {mc_pval:>10.4f} {mc_kup:>8}")
print()
print("Interpretation:")
print("  p-value > 0.05  →  PASS  model exception rate is acceptable")
print("  p-value < 0.05  →  FAIL  model is mis-specified")