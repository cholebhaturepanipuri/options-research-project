# NIFTY Options Research Project — Final Report

**Intern:** Pratham Hari
**Mentor:** Mahavir A. Bhattacharya
**Organisation:** QuantInsti
**GitHub:** https://github.com/cholebhaturepanipuri/options-research-project

## 6. Backtest Results (Real NSE Data — Jan to Jun 2024)

### 6.1 Short Straddle (1.5x SL, 50% target)

| Metric | Value |
|--------|-------|
| Total return | Rs 5,01,315 |
| Win rate | 66.7% |
| Average trade P&L | Rs 11,936 |
| Max drawdown | -Rs 1,44,499 |
| Sharpe ratio | 1.725 |
| Profit factor | 2.462 |
| Exit breakdown | 52.4% expiry, 47.6% target |

### 6.2 Short Strangle (150pt OTM, same rules)

| Metric | Value |
|--------|-------|
| Total return | Rs 4,82,505 |
| Win rate | 69.0% |
| Max drawdown | -Rs 1,42,534 |
| Sharpe ratio | 1.674 |
| Profit factor | 2.391 |

### 6.3 Stop-loss sensitivity

| SL Multiplier | Total Return | Win Rate | Sharpe |
|--------------|-------------|---------|--------|
| 1.0x | -Rs 2,09,906 | 16.7% | -2.560 |
| 1.5x | Rs 5,01,315 | 66.7% | 1.725 |
| 2.0x | Rs 5,01,315 | 66.7% | 1.725 |
| 3.0x | Rs 5,01,315 | 66.7% | 1.725 |

1.5x is the optimal stop-loss level. Tighter (1.0x) destroys returns completely.
