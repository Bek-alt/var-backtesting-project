# Value at Risk — Implementation and Backtesting

A quantitative finance project implementing and validating three
industry-standard Value at Risk (VaR) methodologies on a
multi-asset equity portfolio.

## Portfolio

| Stock | Company          | Sector     | Weight |
|-------|-----------------|------------|--------|
| AAPL  | Apple           | Technology | 25%    |
| JPM   | JPMorgan Chase  | Banking    | 25%    |
| MSFT  | Microsoft       | Technology | 20%    |
| GS    | Goldman Sachs   | Banking    | 15%    |
| XOM   | ExxonMobil      | Energy     | 15%    |

## Methods Implemented

**1. Historical Simulation**
- Non-parametric approach using 250-day rolling window
- No distributional assumptions
- Captures fat tails naturally

**2. Variance-Covariance (Parametric)**
- Assumes normally distributed returns
- Portfolio variance derived from w'Σw
- Uses rolling covariance matrix

**3. Monte Carlo Simulation**
- 10,000 simulated return paths per day
- Cholesky decomposition for correlated shocks
- Based on Geometric Brownian Motion

## Backtesting Framework

| Test                  | Purpose                                      |
|-----------------------|----------------------------------------------|
| Exception counting    | Count days where loss exceeded VaR           |
| Kupiec POF test       | Test if exception rate is statistically valid |
| Christoffersen test   | Test if exceptions are independent over time  |
| Basel traffic light   | Regulatory classification (Green/Yellow/Red) |

## Results

| Metric                | Hist Sim | Var-Cov | Monte Carlo |
|-----------------------|----------|---------|-------------|
| Total days tested     | 1258     | 1258    | 1258        |
| Expected exceptions   | 62.9     | 62.9    | 62.9        |
| Actual exceptions     | 56       | 52      | 60          |
| Exception rate        | 4.45%    | 4.13%   | 4.77%       |
| Kupiec test           | PASS     | PASS    | PASS        |
| Christoffersen test   | PASS     | PASS    | PASS        |
| Basel zone            | RED      | RED     | RED         |

All three models pass both statistical tests, confirming
exception rates are consistent with the 95% confidence level.
The RED Basel classification reflects the strict regulatory
threshold and motivates the use of 99% confidence in
regulatory VaR reporting.

## Key Finding

The Variance-Covariance model produces the fewest exceptions
(52) due to the normality assumption causing it to overestimate
volatility during calm periods. Historical Simulation and Monte
Carlo produce exception rates closer to the theoretical 5%,
with no evidence of exception clustering in any model.

## Requirements

Install dependencies:
pip install yfinance numpy pandas matplotlib scipy jupyter

## Run

jupyter notebook var_project.ipynb