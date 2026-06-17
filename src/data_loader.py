"""
NSE F&O bhavcopy data loader.

Downloads and cleans NSE options data from the standard F&O bhavcopy CSV
format published daily by NSE India.

NSE bhavcopy URL pattern (as of 2024):
  https://nsearchives.nseindia.com/content/fo/
  BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv

Column mapping handles the two common naming conventions NSE has used.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column mappings (NSE has used different names across years)
# ---------------------------------------------------------------------------

# Older format (pre-2023 approximately)
_LEGACY_MAP = {
    "INSTRUMENT"  : "instrument",
    "SYMBOL"      : "symbol",
    "EXPIRY_DT"   : "expiry_date",
    "STRIKE_PR"   : "strike",
    "OPTION_TYP"  : "option_type",
    "OPEN"        : "open",
    "HIGH"        : "high",
    "LOW"         : "low",
    "CLOSE"       : "close",
    "SETTLE_PR"   : "settle_price",
    "CONTRACTS"   : "contracts",
    "VAL_INLAKH"  : "value_lakh",
    "OPEN_INT"    : "open_interest",
    "CHG_IN_OI"   : "oi_change",
    "TIMESTAMP"   : "date",
}

# Newer format (2023+)
_NEW_MAP = {
    "TradDt"      : "date",
    "BizDt"       : "date",
    "Sgmt"        : "segment",
    "Src"         : "src",
    "FinInstrmTp" : "instrument",
    "FinInstrmId" : "symbol",
    "XpryDt"      : "expiry_date",
    "StrkPric"    : "strike",
    "OptnTp"      : "option_type",
    "OpnPric"     : "open",
    "HghPric"     : "high",
    "LwPric"      : "low",
    "ClsPric"     : "close",
    "SttlmPric"   : "settle_price",
    "TtlTradgVol" : "contracts",
    "TtlTrfVal"   : "value_lakh",
    "OpnIntrst"   : "open_interest",
    "ChngInOpnIntrst": "oi_change",
}

_REQUIRED_COLS = {"instrument", "symbol", "expiry_date", "strike", "option_type",
                  "open", "high", "low", "close", "date"}


# ---------------------------------------------------------------------------
# Load a single bhavcopy file
# ---------------------------------------------------------------------------

def load_bhavcopy(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load and clean one NSE F&O bhavcopy CSV file.

    Handles both legacy and new column naming conventions.
    Parses dates, strips whitespace, drops nulls in critical columns.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    pd.DataFrame with standardised column names.
    """
    filepath = Path(filepath)
    raw = pd.read_csv(filepath, low_memory=False)

    # Normalise column names: strip whitespace, try both maps
    raw.columns = raw.columns.str.strip()

    upper_cols = {c: c.upper() for c in raw.columns}
    raw = raw.rename(columns=upper_cols)

    # Detect format and apply mapping
    if "INSTRUMENT" in raw.columns:
        df = raw.rename(columns=_LEGACY_MAP)
    elif "FININSTRMTP" in raw.columns or "FININSTRMID" in raw.columns:
        # New format — column names may be mixed case in the CSV
        new_map_upper = {k.upper(): v for k, v in _NEW_MAP.items()}
        df = raw.rename(columns=new_map_upper)
    else:
        # Unknown format — try generic strip and pass through
        warnings.warn(
            f"Unrecognised bhavcopy format in {filepath.name}. "
            "Attempting best-effort load.",
            UserWarning,
        )
        df = raw.copy()

    # Drop duplicate/redundant columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    # Parse date columns
    for col in ("date", "expiry_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # Strip whitespace from string columns
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    # Numeric coercion for price/volume columns
    for col in ("strike", "open", "high", "low", "close", "settle_price",
                "contracts", "open_interest"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing critical fields
    critical = [c for c in ("date", "expiry_date", "strike", "close") if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        warnings.warn(f"{dropped} rows dropped due to missing critical fields in {filepath.name}.")

    return df


def load_bhavcopy_folder(folder: Union[str, Path]) -> pd.DataFrame:
    """
    Load and concatenate all bhavcopy CSVs from a folder, sorted by date.

    Parameters
    ----------
    folder : str or Path   Directory containing bhavcopy CSVs.

    Returns
    -------
    pd.DataFrame   Combined, date-sorted DataFrame.
    """
    folder = Path(folder)
    files  = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    dfs = []
    for f in files:
        try:
            dfs.append(load_bhavcopy(f))
        except Exception as exc:
            warnings.warn(f"Could not load {f.name}: {exc}")

    if not dfs:
        raise ValueError("No files were loaded successfully.")

    combined = pd.concat(dfs, ignore_index=True)
    if "date" in combined.columns:
        combined = combined.sort_values("date").reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Filtering and enrichment
# ---------------------------------------------------------------------------

def filter_nifty_options(df: pd.DataFrame, symbol: str = "NIFTY") -> pd.DataFrame:
    """
    Keep only NIFTY index options (instrument = OPTIDX, symbol = NIFTY).

    Also normalises option_type to uppercase CE / PE,
    and drops rows with unknown option types.
    """
    df = df.copy()

    if "instrument" in df.columns:
        df = df[df["instrument"].str.upper().str.strip() == "OPTIDX"]
    if "symbol" in df.columns:
        df = df[df["symbol"].str.upper().str.strip() == symbol.upper()]

    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].str.upper().str.strip()
        df = df[df["option_type"].isin(["CE", "PE"])]

    return df.reset_index(drop=True)


def add_time_to_expiry(df: pd.DataFrame) -> pd.DataFrame:
    """Add `days_to_expiry` (int) and `T` (float, years) columns."""
    df = df.copy()
    df["days_to_expiry"] = (df["expiry_date"] - df["date"]).dt.days
    df["T"] = df["days_to_expiry"] / 365.0
    # Drop rows where expiry is in the past (data error)
    df = df[df["days_to_expiry"] >= 0].reset_index(drop=True)
    return df


def add_moneyness(df: pd.DataFrame, spot_col: str = "spot") -> pd.DataFrame:
    """
    Add moneyness columns: S/K ratio, log-moneyness, and ATM flag.

    ATM defined as |S/K − 1| ≤ 0.5%.
    Requires `spot_col` to already be present in the DataFrame.
    """
    df = df.copy()
    df["moneyness"]     = df[spot_col] / df["strike"]
    df["log_moneyness"] = np.log(df["moneyness"])
    df["is_atm"]        = (np.abs(df["moneyness"] - 1.0) <= 0.005)
    return df


def load_spot_prices(
    filepath : Union[str, Path],
    date_col : str = "Date",
    close_col: str = "Close",
) -> pd.Series:
    """
    Load NIFTY spot / index prices from a CSV (e.g., yfinance download).

    Returns
    -------
    pd.Series   Daily spot prices, date-indexed, named "spot".
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)
    return df[close_col].rename("spot")


def join_spot(
    options_df  : pd.DataFrame,
    spot_series : pd.Series,
    date_col    : str = "date",
) -> pd.DataFrame:
    """
    Left-join spot prices onto the options DataFrame by date.

    Warns if any rows are missing a spot match.
    """
    df = options_df.copy()
    df = df.join(spot_series, on=date_col, how="left")
    missing = df["spot"].isna().sum()
    if missing > 0:
        warnings.warn(f"{missing} rows have no matching spot price — check date alignment.")
    return df


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------

def data_quality_report(df: pd.DataFrame) -> dict:
    """Print a short data quality summary."""
    report = {
        "total_rows"      : len(df),
        "date_range"      : (df["date"].min(), df["date"].max()) if "date" in df.columns else None,
        "unique_dates"    : df["date"].nunique() if "date" in df.columns else None,
        "unique_expiries" : df["expiry_date"].nunique() if "expiry_date" in df.columns else None,
        "unique_strikes"  : df["strike"].nunique() if "strike" in df.columns else None,
        "option_types"    : df["option_type"].value_counts().to_dict() if "option_type" in df.columns else None,
        "missing_close"   : int(df["close"].isna().sum()) if "close" in df.columns else None,
        "zero_price_rows" : int((df["close"] == 0).sum()) if "close" in df.columns else None,
    }
    return report


# ---------------------------------------------------------------------------
# Quick self-test (runs only when executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, datetime

    sample_csv = """INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP
OPTIDX,NIFTY,25-JAN-2024,19500,CE,210,225,195,218,218,5000,10000,50000,2000,02-JAN-2024
OPTIDX,NIFTY,25-JAN-2024,19500,PE,185,200,175,192,192,4800,9500,48000,1800,02-JAN-2024
OPTIDX,NIFTY,25-JAN-2024,19600,CE,150,165,140,158,158,3000,6000,30000,1000,02-JAN-2024
FUTSTK,RELIANCE,25-JAN-2024,0,XX,2500,2550,2480,2520,2520,2000,5000,20000,500,02-JAN-2024
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_csv)
        tmp_path = f.name

    df_raw = load_bhavcopy(tmp_path)
    df_nifty = filter_nifty_options(df_raw)
    df_nifty = add_time_to_expiry(df_nifty)

    # Fake spot
    spot = pd.Series({"spot": 19520}, name="spot")
    spot.index = [pd.Timestamp("2024-01-02")]
    spot_series = spot.rename_axis("date")

    print("Loaded rows:", len(df_raw))
    print("NIFTY options:", len(df_nifty))
    print(df_nifty[["date", "expiry_date", "strike", "option_type", "close", "T"]].to_string(index=False))
    print("\nData quality report:")
    for k, v in data_quality_report(df_nifty).items():
        print(f"  {k}: {v}")
