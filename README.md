# NIFTY Options Research Project

A complete end-to-end options research pipeline built in Python.
Covers option payoffs, Black-Scholes pricing, Greeks, implied volatility,
realised volatility, strategy backtesting, delta hedging, and a Streamlit
research dashboard.

**Intern:** Pratham Hari  
**Mentor:** Mahavir A. Bhattacharya  
**Duration:** 30 days

---

## Project structure

```
options-research-project/
├── data/
│   ├── raw/              # Downloaded NSE bhavcopy CSVs (not committed)
│   ├── processed/        # Cleaned, joined datasets
│   └── sample/           # Small sample for quick testing
│
├── notebooks/
│   ├── 01_options_basics.ipynb
│   ├── 02_black_scholes_greeks_iv.ipynb
│   ├── 03_iv_vs_rv_analysis.ipynb
│   ├── 04_strategy_backtest.ipynb
│   └── 05_delta_hedging.ipynb
│
├── src/
│   ├── option_pricing.py   # BSM pricing + IV solver (Brent's method)
│   ├── greeks.py           # Delta, Gamma, Theta, Vega, Rho
│   ├── strategies.py       # Multi-leg payoff engine
│   ├── volatility.py       # RV estimators + IV-RV analysis
│   ├── data_loader.py      # NSE bhavcopy loader and cleaner
│   ├── backtester.py       # Short straddle / strangle backtest engine
│   ├── metrics.py          # Performance metrics
│   └── plots.py            # Matplotlib chart utilities
│
├── dashboard/
│   └── app.py              # Streamlit research dashboard
│
├── reports/
│   ├── final_report.md
│   └── presentation_notes.md
│
├── requirements.txt
├── config.yaml
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/options-research-project.git
cd options-research-project

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Data

Options data comes from the **NSE F&O Bhavcopy** (free, no login required).

1. Go to https://www.nseindia.com/market-data/future-and-options-bhavcopy-report
2. Download daily CSV files and place them in `data/raw/`.
3. Run the data pipeline notebook: `notebooks/03_...` or call `data_loader.py` directly.

Spot data can be downloaded via `yfinance`:

```python
import yfinance as yf
nifty = yf.download("^NSEI", start="2022-01-01", end="2024-01-01")
nifty["Close"].to_csv("data/processed/nifty_spot.csv")
```

---

## Running the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard degrades gracefully if processed data is not yet available —
the Payoff Visualiser and Greeks Explorer work with no data files.

---

## Running notebooks

```bash
jupyter lab
```

Open notebooks in order: 01 → 02 → 03 → 04 → 05.

---

## Module overview

| Module            | What it does                                              |
|-------------------|-----------------------------------------------------------|
| `option_pricing`  | `bs_price`, `implied_volatility` (Brent's method)        |
| `greeks`          | `delta`, `gamma`, `theta`, `vega`, `rho`, `all_greeks`   |
| `strategies`      | `Leg`, `Strategy`, factory functions for 10+ strategies  |
| `volatility`      | Close-to-close, Parkinson, Yang-Zhang RV; IV-RV spread   |
| `data_loader`     | Load/clean NSE bhavcopy; add T, moneyness, spot join     |
| `backtester`      | `Backtester` with short straddle and strangle runs       |
| `metrics`         | 10 performance metrics; equity curve; drawdown series    |
| `plots`           | Payoff, Greeks, IV-RV, equity curve, delta hedge charts  |

---

## Known limitations

- Backtester uses end-of-day closing prices (no intraday stop-loss).
- Transaction costs are approximate (₹50 per lot per leg).
- NIFTY lot size is hardcoded — verify current value from NSE.
- NSE bhavcopy column names have changed across years; `data_loader.py`
  handles both legacy and new formats but edge cases may exist.
- No margin or capital requirement modelling.

---

## Stretch goals (after core project)

- [ ] Volatility smile / skew analysis
- [ ] Walk-forward backtest
- [ ] Margin approximation
- [ ] Unit tests (pytest)
- [ ] Dockerfile
- [ ] Deploy dashboard to Streamlit Community Cloud
