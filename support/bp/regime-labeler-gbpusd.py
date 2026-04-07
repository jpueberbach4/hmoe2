import json
import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime, timezone
from typing import List, Dict, Any

# ==============================================================================
# MANUAL REGIME CONFIGURATION GBPUSD
# Edit this array to mark your exact Bull and Bear market ranges.
# ==============================================================================
DEFAULT_REGIMES = [
    {"from": "2017-01-01", "to": "2018-04-09", "type": "bull"},
    {"from": "2018-04-10", "to": "2019-08-15", "type": "bear"},
    {"from": "2019-08-16", "to": "2019-12-19", "type": "bull"},
    {"from": "2019-12-20", "to": "2020-05-20", "type": "bear"},
    {"from": "2020-05-21", "to": "2021-05-26", "type": "bull"},
    {"from": "2021-05-27", "to": "2022-09-26", "type": "bear"},
    {"from": "2022-09-27", "to": "2023-07-13", "type": "bull"},
    {"from": "2023-07-14", "to": "2023-10-19", "type": "bear"},
    {"from": "2023-10-20", "to": "2024-09-20", "type": "bull"},
    {"from": "2024-09-21", "to": "2025-01-14", "type": "bear"},
    {"from": "2025-01-15", "to": "2025-07-04", "type": "bull"},
    {"from": "2025-07-05", "to": "2025-11-04", "type": "bear"},
    {"from": "2025-11-05", "to": "2026-01-27", "type": "bull"},
    {"from": "2026-01-28", "to": "2026-03-31", "type": "bear"}
]

def description() -> str:
    return (
        "Supervised Regime Labeler. Outputs 1 for Bull regimes, -1 for Bear regimes, "
        "and 0 for unmarked ranging/choppy markets based on explicit date ranges."
    )

def meta() -> Dict:
    return {
        "author": "Google Gemini",
        "version": 1.0,
        "panel": 1,          # Put this on a lower pane so it doesn't overlay the price
        "verified": 1,
        "talib-validated": 0, 
        "polars": 1
    }

def warmup_count(options: Dict[str, Any]) -> int:
    """No lookback required. This is an absolute temporal mapping."""
    return 0 

def position_args(args: List[str]) -> Dict[str, Any]:
    # Allows passing a custom JSON string via URL if your engine supports it
    if len(args) > 0:
        return {"regimes": args[0]}
    return {}

def calculate_polars(indicator_str: str, options: Dict[str, Any]) -> pl.Expr:
    """
    Polars-native conditional mapping using a chained when/then statement.
    """
    # 1. Load Regimes
    regimes_input = options.get('regimes', DEFAULT_REGIMES)
    if isinstance(regimes_input, str):
        try:
            regimes = json.loads(regimes_input)
        except Exception:
            regimes = DEFAULT_REGIMES
    else:
        regimes = regimes_input

    # 2. Identify the time column. Adjust this if your engine uses "time" or "date" instead.
    time_col = pl.col("time_ms")
    
    # Base state is 0 (ranging)
    expr = pl.lit(0)

    # 3. Chain the conditions
    for r in regimes:
        start_str = r.get("from")
        end_str = r.get("to")
        rtype = r.get("type", "").lower()
        
        val = 1 if rtype == "bull" else 0

        # Convert strings to UTC Milliseconds (matches your fetch_raw_api_data schema)
        start_ms = int(datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

        # Build the condition: If candle is between Start and End...
        condition = (time_col >= start_ms) & (time_col <= end_ms)
        
        # ...Then assign the regime value, otherwise keep the previous state
        expr = pl.when(condition).then(val).otherwise(expr)

    return expr.alias(indicator_str)

def calculate(df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
    """
    Legacy Pandas fallback using the DatetimeIndex.
    """
    regimes_input = options.get('regimes', DEFAULT_REGIMES)
    if isinstance(regimes_input, str):
        try:
            regimes = json.loads(regimes_input)
        except Exception:
            regimes = DEFAULT_REGIMES
    else:
        regimes = regimes_input

    # Initialize everything as 0
    result = pd.Series(0, index=df.index, name='regime')

    for r in regimes:
        start_str = r.get("from")
        end_str = r.get("to")
        rtype = r.get("type", "").lower()
        
        val = 1 if rtype == "bull" else (-1 if rtype == "bear" else 0)
        
        # Pandas allows direct string slicing on DatetimeIndexes
        try:
            result.loc[start_str:end_str] = val
        except KeyError:
            # If dates are completely out of bounds for the current slice, ignore
            pass

    return pd.DataFrame({ 'regime': result })