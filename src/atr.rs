//! Average True Range (ATR)
//!
//! ATR is a volatility indicator that measures market volatility by
//! decomposing the entire range of an asset price for a period.

use anyhow::Result;
use polars::prelude::*;

/// Calculate Average True Range (ATR)
///
/// True Range = max(high - low, |high - prev_close|, |low - prev_close|)
/// ATR = EMA of True Range
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `period` - ATR period (standard: 14)
///
/// # Returns
///
/// A new Series containing ATR values
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::atr;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let atr_14 = atr(&high, &low, &close, 14).unwrap();
/// ```
pub fn atr(high: &Series, low: &Series, close: &Series, period: usize) -> Result<Series> {
    let ewm_opts = EWMOptions {
        alpha: 1.0 / period as f64,
        adjust: true,
        bias: false,
        min_periods: period,
        ignore_nulls: true,
    };
    
    let df = DataFrame::new(vec![
        high.clone(),
        low.clone(),
        close.clone(),
    ])?;
    
    let result = df
        .lazy()
        .with_columns(vec![
            (col("high") - col("low")).alias("tr1"),
            (col("high") - col("close").shift(lit(1))).abs().alias("tr2"),
            (col("low") - col("close").shift(lit(1))).abs().alias("tr3"),
        ])
        .with_column(
            when(col("tr1").gt_eq(col("tr2")).and(col("tr1").gt_eq(col("tr3"))))
                .then(col("tr1"))
                .when(col("tr2").gt_eq(col("tr3")))
                .then(col("tr2"))
                .otherwise(col("tr3"))
                .alias("tr"),
        )
        .with_column(col("tr").ewm_mean(ewm_opts).alias("atr"))
        .collect()?;
    
    Ok(result.column("atr")?.clone())
}

/// Calculate ATR with the standard 14-period setting
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
///
/// # Returns
///
/// A new Series containing ATR values
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::atr_14;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let atr = atr_14(&high, &low, &close).unwrap();
/// ```
pub fn atr_14(high: &Series, low: &Series, close: &Series) -> Result<Series> {
    atr(high, low, close, 14)
}
