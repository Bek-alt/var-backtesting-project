# Value at Risk — Methodology and Validation Note

**Author:** Your Name
**Date:** April 2025
**Institution:** [Your University]
**Portfolio:** AAPL, JPM, MSFT, GS, XOM

---

## 1. Objective

This note documents the mathematical methodology and empirical
validation of three Value at Risk (VaR) models applied to a
five-asset equity portfolio over the period January 2018 to
January 2024, comprising 1,258 trading days.

VaR is defined as the maximum expected loss over a given horizon
at a specified confidence level. Formally, for confidence level
alpha and horizon h:

P(loss > VaR) = 1 - alpha

We implement VaR at alpha = 0.95 and h = 1 trading day.

---

## 2. Data

Daily closing prices were obtained from Yahoo Finance for five
equities spanning three sectors: technology (AAPL, MSFT),
banking (JPM, GS), and energy (XOM).

Log returns were computed as:

r_t = ln(P_t / P_{t-1})

Log returns are preferred over simple returns for three reasons:
time additivity, approximate symmetry, and closer approximation
to normality.

Portfolio returns were computed as the weighted sum:

r_portfolio,t = sum_i w_i × r_{i,t}

with weights w = [0.25, 0.25, 0.20, 0.15, 0.15].

---

## 3. Methodology

### 3.1 Historical Simulation

At each day t, the 250 preceding daily portfolio returns are
sorted and the 5th percentile is taken as the VaR estimate:

VaR_t = Percentile({r_{t-250}, ..., r_{t-1}}, 5)

This method makes no distributional assumption and captures
fat tails naturally. Its limitation is that it treats all
historical observations as equally likely regardless of
when they occurred.

### 3.2 Variance-Covariance

Under the assumption that returns are multivariate normally
distributed, the portfolio VaR is:

VaR_t = z_{0.05} × sigma_{portfolio,t}

where z_{0.05} = -1.645 is the 5th percentile of the standard
normal distribution and sigma_{portfolio,t} is the portfolio
standard deviation derived from:

sigma²_{portfolio,t} = w' Sigma_t w

Sigma_t is the 5x5 sample covariance matrix of individual stock
returns over the preceding 250 trading days. This method is
computationally efficient but assumes normality, which
underestimates tail risk during market stress.

### 3.3 Monte Carlo Simulation

At each day t, 10,000 return scenarios are simulated using:

r_sim = mu + epsilon,   epsilon = Z L'

where mu is the 250-day sample mean return vector, Z is a
matrix of independent standard normal shocks of shape
(10000 x 5), and L is the lower triangular Cholesky factor
of the sample covariance matrix satisfying L L' = Sigma_t.

The Cholesky decomposition ensures that the simulated shocks
have the correct covariance structure. This follows from:

Cov(epsilon) = Cov(Z L') = L Cov(Z) L' = L I L' = L L' = Sigma

VaR is the 5th percentile of the 10,000 simulated portfolio
returns. Monte Carlo is the most flexible method and can be
extended to non-normal distributions and nonlinear instruments.

---

## 4. Backtesting

### 4.1 Exception Counting

An exception occurs on day t when the realised portfolio return
is worse than the VaR estimate:

I_t = 1 if r_t < VaR_t,   0 otherwise

Under a correctly specified model at 95% confidence, exceptions
should occur with probability p0 = 0.05, giving an expected
count of T × 0.05 over T days.

### 4.2 Kupiec Proportion of Failures Test

The Kupiec test formally examines whether the observed exception
rate is consistent with p0 = 0.05. The likelihood ratio statistic is:

LR_POF = -2 × [x × ln(p0/p_hat) + (T-x) × ln((1-p0)/(1-p_hat))]

where x is the number of exceptions and p_hat = x/T is the
observed rate. Under the null hypothesis of correct specification,
LR_POF follows a chi-squared distribution with 1 degree of freedom.
The model is rejected at 95% confidence if LR_POF > 3.841.

### 4.3 Christoffersen Independence Test

A correctly specified model should produce exceptions that are
independent over time — not clustered. The Christoffersen test
examines the transition probabilities between exception and
non-exception days. Defining:

pi01 = P(exception today | no exception yesterday)
pi11 = P(exception today | exception yesterday)

Under independence, pi01 = pi11. The likelihood ratio statistic
tests this hypothesis and follows a chi-squared distribution
with 1 degree of freedom under the null.

### 4.4 Basel Traffic Light System

Under Basel III, VaR models are classified based on exception
counts over 250 trading days:

0-4  exceptions  →  GREEN   model acceptable
5-9  exceptions  →  YELLOW  model under scrutiny
10+  exceptions  →  RED     model rejected

---

## 5. Results and Interpretation

| Metric                | Hist Sim | Var-Cov | Monte Carlo |
|-----------------------|----------|---------|-------------|
| Actual exceptions     | 56       | 52      | 60          |
| Exception rate        | 4.45%    | 4.13%   | 4.77%       |
| Kupiec test           | PASS     | PASS    | PASS        |
| Christoffersen test   | PASS     | PASS    | PASS        |
| Basel zone            | RED      | RED     | RED         |

All three models pass both the Kupiec and Christoffersen tests,
confirming that exception rates are statistically consistent
with the 95% confidence level and that exceptions are not
clustered in time.

The Variance-Covariance model produces the fewest exceptions
(52, rate 4.13%), reflecting the tendency of the normality
assumption to produce more conservative VaR estimates during
low-volatility periods.

The RED Basel classification for all models is consistent with
the 95% confidence level used. Scaling 56 exceptions from 1258
days to a 250-day window yields approximately 11 exceptions,
exceeding the RED threshold of 10. This finding motivates the
regulatory convention of computing VaR at 99% confidence,
which would reduce expected exceptions from 62.9 to 12.6 over
1258 days and from 12.5 to 2.5 over 250 days — within the
GREEN zone.

---

## 6. Conclusion

This analysis demonstrates that all three VaR methodologies
produce statistically valid risk estimates at 95% confidence
over the 2018-2024 sample period. The models differ primarily
in their assumptions:

Historical Simulation makes no distributional assumption and
naturally captures fat tails but is computationally intensive
and treats all historical observations equally.

Variance-Covariance is fastest and most transparent but relies
on the normality assumption which fails during market stress.

Monte Carlo is the most flexible and extensible method, capable
of incorporating non-normal distributions, stochastic volatility,
and nonlinear instruments, at the cost of computational intensity.

For regulatory purposes, moving to 99% confidence would bring
all models into the Basel GREEN zone. For internal risk
management at 95% confidence, Historical Simulation and Monte
Carlo are preferred given their ability to capture tail risk
without parametric assumptions.