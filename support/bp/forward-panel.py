import polars as pl
from typing import List, Dict, Any

def description() -> str:
    return (
        "Generic MTF Panel Forwarder: Fetches any indicator from a target timeframe "
        "and binds it to the current timeframe's timeline. \n"
        "Uses backward as-of joining and is-open filtering to guarantee "
        "zero lookahead bias when merging Higher Timeframes to Lower Timeframes.\n"
        "Includes an optional 'smoothing' parameter to apply an SMA to the forwarded "
        "data on the base timeline, removing 'staircase' artifacts for TCN backends."
    )

def meta() -> Dict:
    return {
        "author": "Google Gemini",
        "version": 1.1, 
        "panel": 1,
        "verified": 1,
        "polars": 0,
        "polars_input": 1,
        "nocheck": 1
    }

def warmup_count(options: Dict[str, Any]) -> int:
    # Warmup is handled internally by extending the fetch window
    return 0

def position_args(args: List[str]) -> Dict[str, Any]:
    # Example incoming call: forward_1d_macd_12,26,9_6
    # args = ["1d", "macd", "12,26,9", "6"]
    return {
        "target_tf": args[0] if len(args) > 0 else "1d",
        "target_ind": args[1] if len(args) > 1 else "rsi",
        "target_params": args[2] if len(args) > 2 else "14",
        "smoothing": args[3] if len(args) > 3 else "0"  # NEW: Smoothing parameter
    }

def calculate(df: pl.DataFrame, options: Dict[str, Any]) -> pl.DataFrame:
    from util.api import get_data
    import polars as pl

    symbol = df['symbol'].item(0)
    
    # Extract MTF configuration
    target_tf = options.get("target_tf", "1d")
    target_ind = options.get("target_ind", "rsi")
    target_params = options.get("target_params", "")
    smoothing_period = int(options.get("smoothing", 0))
    
    # Reconstruct the target indicator string (e.g. macd_12_26_9)
    if target_params:
        params_formatted = str(target_params).replace(",", "_")
        full_indicator = f"{target_ind}_{params_formatted}"
    else:
        full_indicator = target_ind

    # Get the exact time bounds of the current lower-timeframe dataset
    time_min = df["time_ms"][0]
    time_max = df["time_ms"][-1]

    # Force API to return Polars DataFrames
    api_opts = {**options, "return_polars": True}

    # Extend the start time backwards to cover the "ramp up" period of HTF indicators.
    warmup_ms = 86400000 * 365 

    # Fetch the target higher-timeframe data
    data = get_data(
        symbol=symbol,
        timeframe=target_tf,
        after_ms=time_min - warmup_ms,
        until_ms=time_max + 1,
        indicators=[full_indicator, "is-open"],
        limit=1000000,
        options=api_opts
    )

    if data.is_empty():
        return pl.DataFrame()

    # Automatically detect which columns the backend generated for the indicator
    exclude_cols = {"time_ms", "is-open", "symbol", "timeframe", "open", "high", "low", "close", "volume"}
    ind_cols = [col for col in data.columns if col not in exclude_cols]

    # Prefix the generated columns with the target timeframe
    rename_mapping = {col: f"{target_tf}_{col}" for col in ind_cols}

    # Clean the higher timeframe data
    lazy_target = (
        data.lazy()
        .filter(pl.col("is-open") == 0)      # Drop open candles completely
        .select(["time_ms"] + ind_cols)      # Keep only time and indicator data
        .rename(rename_mapping)              # Apply the 1d_ prefix
        .sort("time_ms")
    )

    # Make a flat timeline of the current timeframe to join into
    timeline = df.select([pl.col("time_ms").cast(pl.UInt64)]).lazy()
    
    # Join the target indicator onto the base timeline (Backward As-Of)
    lazy_result = timeline.join_asof(lazy_target, on="time_ms", strategy="backward")

    # --- APPLY SMOOTHING POST-MERGE ---
    if smoothing_period > 1:
        smoothed_exprs = [
            # min_periods=1 prevents creating NaNs at the start of the dataset
            pl.col(col_name).rolling_mean(window_size=smoothing_period, min_periods=1).alias(col_name)
            for col_name in rename_mapping.values()
        ]
        lazy_result = lazy_result.with_columns(smoothed_exprs)

    # Collect final frame
    result_ldf = (
        lazy_result
        .select(list(rename_mapping.values())) 
        .collect(engine="streaming")
    )

    return result_ldf