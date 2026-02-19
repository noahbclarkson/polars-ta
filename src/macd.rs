//! Moving Average Convergence Divergence (MACD)
//!
//! MACD is a trend-following momentum indicator that shows the
//! relationship between two exponential moving averages of a security's price.

use anyhow::Result;
use polars::prelude::*;

/// Result of MACD calculation containing MACD line, signal line, and histogram
#[derive(Debug)]
pub struct MacdResult {
    /// The MACD line (fast EMA - slow EMA)
    pub macd: Series,
    /// The signal line (EMA of MACD)
    pub signal: Series,
    /// The histogram (MACD - signal)
    pub histogram: Series,
}

/// Calculate MACD with custom parameters
///
/// # Arguments
///
/// * `series` - Input price series
/// * `fast` - Fast EMA period (default: 12)
/// * `slow` - Slow EMA period (default: 26)
/// * `signal` - Signal line EMA period (default: 9)
///
/// # Returns
///
/// A `MacdResult` containing the MACD line, signal line, and histogram
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::macd;
///
/// let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let result = macd(&close, 12, 26, 9).unwrap();
/// println!("MACD: {:?}", result.macd);
/// ```
pub fn macd(series: &Series, fast: usize, slow: usize, signal: usize) -> Result<MacdResult> {
    let col_name = series.name();
    
    let ewm_fast = EWMOptions {
        alpha: 1.0 / fast as f64,
        adjust: true,
        bias: false,
        min_periods: fast,
        ignore_nulls: true,
    };
    
    let ewm_slow = EWMOptions {
        alpha: 1.0 / slow as f64,
        adjust: true,
        bias: false,
        min_periods: slow,
        ignore_nulls: true,
    };
    
    let ewm_signal = EWMOptions {
        alpha: 1.0 / signal as f64,
        adjust: true,
        bias: false,
        min_periods: signal,
        ignore_nulls: true,
    };
    
    let df = DataFrame::new(vec![series.clone()])?;
    
    let result = df
        .lazy()
        .with_column(col(col_name).ewm_mean(ewm_fast).alias("ema_fast"))
        .with_column(col(col_name).ewm_mean(ewm_slow).alias("ema_slow"))
        .with_column((col("ema_fast") - col("ema_slow")).alias("macd_line"))
        .with_column(col("macd_line").ewm_mean(ewm_signal).alias("signal_line"))
        .with_column((col("macd_line") - col("signal_line")).alias("histogram"))
        .collect()?;
    
    Ok(MacdResult {
        macd: result.column("macd_line")?.clone(),
        signal: result.column("signal_line")?.clone(),
        histogram: result.column("histogram")?.clone(),
    })
}

/// Calculate MACD with default parameters (12/26/9)
///
/// # Arguments
///
/// * `series` - Input price series
///
/// # Returns
///
/// A `MacdResult` containing the MACD line, signal line, and histogram
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::macd_default;
///
/// let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let result = macd_default(&close).unwrap();
/// ```
pub fn macd_default(series: &Series) -> Result<MacdResult> {
    macd(series, 12, 26, 9)
}
