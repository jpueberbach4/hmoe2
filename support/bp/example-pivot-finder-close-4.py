import polars as pl
from typing import List, Dict, Any

def description() -> str:
    return (
        "Gaussian Proximity Pivot Identifier. "
        "Bottoms = -1.0 to 0.0 fade. Tops = 1.0 to 0.0 fade. "
        "Maintains original block widths but centers the peak."
    )

def meta() -> Dict:
    return {
        "author": "Quant",
        "version": 8.2,
        "panel": 1,
        "verified": 1,
        "polars_input": 1
    }

def warmup_count(options: Dict[str, Any]) -> int:
    window = int(options.get('window', 80))
    sma_len = int(options.get('sma_len', 5))
    return window + sma_len

def position_args(args: List[str]) -> Dict[str, Any]:
    return {
        "window": args[0] if len(args) > 0 else "80",
        "what": args[1] if len(args) > 1 else "bottoms",
        "threshold": args[2] if len(args) > 2 else "0.01",
        "sma_len": args[3] if len(args) > 3 else "5"
    }

def calculate(df: pl.DataFrame, options: Dict[str, Any]) -> pl.DataFrame:
    n = int(options.get('window', 80))
    what = str(options.get('what', 'bottoms')).strip().lower()
    thresh = float(options.get('threshold', 0.01))
    sma_len = int(options.get('sma_len', 5))

    # Add row index so we can measure distance to the peak
    df_lazy = df.lazy().with_columns([
        pl.int_range(0, pl.len()).alias("idx"),
        pl.col("close").rolling_max(window_size=n*2+1, center=True).alias("local_max"),
        pl.col("close").rolling_min(window_size=n*2+1, center=True).alias("local_min"),
        pl.col("close").rolling_mean(window_size=sma_len).alias("smooth_close")
    ])

    strict_bottom = pl.when(pl.col("close") == pl.col("local_min")).then(-1.0).otherwise(0.0)
    strict_top = pl.when(pl.col("close") == pl.col("local_max")).then(1.0).otherwise(0.0)

    in_bottom_zone = (pl.col("close") <= pl.col("local_min") * (1.0 + thresh)) | \
                     (pl.col("smooth_close") <= pl.col("local_min") * (1.0 + thresh))
                     
    in_top_zone = (pl.col("close") >= pl.col("local_max") * (1.0 - thresh)) | \
                  (pl.col("smooth_close") >= pl.col("local_max") * (1.0 - thresh))

    if what == "bottoms":
        in_zone = in_bottom_zone
        strict_pivot = strict_bottom
    elif what == "tops":
        in_zone = in_top_zone
        strict_pivot = strict_top
    else:
        in_zone = in_bottom_zone | in_top_zone
        strict_pivot = pl.when(strict_bottom == -1.0).then(-1.0)\
                         .when(strict_top == 1.0).then(1.0)\
                         .otherwise(0.0)

    df_lazy = df_lazy.with_columns([
        in_zone.alias("in_zone"),
        strict_pivot.alias("strict_pivot")
    ]).with_columns([
        (pl.col("in_zone") != pl.col("in_zone").shift().fill_null(pl.col("in_zone"))).cum_sum().alias("run_id")
    ])

    # Find the exact center of the pivot within the run
    df_lazy = df_lazy.with_columns([
        pl.col("strict_pivot").max().over("run_id").alias("run_max"),
        pl.col("strict_pivot").min().over("run_id").alias("run_min"),
        pl.when(pl.col("strict_pivot") != 0.0).then(pl.col("idx")).otherwise(None)
          .mean().over("run_id").alias("center_idx")
    ]).with_columns([
        pl.when(pl.col("run_max") == 1.0).then(1.0)
          .when(pl.col("run_min") == -1.0).then(-1.0)
          .otherwise(0.0).alias("run_pivot_val"),
        # Distance from the exact center
        (pl.col("idx") - pl.col("center_idx")).abs().alias("dist")
    ]).with_columns([
        # Get the maximum distance from center for this specific block
        pl.col("dist").max().over("run_id").alias("max_dist")
    ])

    # Calculate the Gaussian fade. Sigma scales with the block width.
    sigma = pl.col("max_dist") / 2.0
    gaussian_weight = pl.when(pl.col("max_dist") == 0.0).then(1.0)\
                        .otherwise(( -0.5 * (pl.col("dist") / sigma).pow(2) ).exp())

    final_expr = pl.when(pl.col("in_zone") & (pl.col("run_pivot_val") != 0.0))\
                   .then(pl.col("run_pivot_val") * gaussian_weight)\
                   .otherwise(0.0)

    return (
        df_lazy
        .select([
            final_expr.alias("major_pivot")
        ])
        .collect(engine="streaming")
    )