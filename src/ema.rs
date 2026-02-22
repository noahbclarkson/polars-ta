//! Exponential Moving Average (EMA)
//!
//! The EMA gives more weight to recent prices, making it more responsive
//! to new information compared to a simple moving average.

use anyhow::Result;
use polars::prelude::*;

/// Calculate Exponential Moving Average
///
/// Uses the formula: EMA_t = α * price_t + (1 - α) * EMA_{t-1}
///
/// **Note on Alpha Convention:** This implementation uses `α = 1 / period`.
/// This differs from the standard Wilder/TA-Lib convention of `α = 2 / (period + 1)`.
/// Users migrating from other libraries may see slightly different numerical values.
///
/// # Arguments
///
/// * `series` - Input price series
/// * `period` - EMA period (e.g., 20, 50, 200)
///
/// # Returns
///
/// A new Series containing the EMA values
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::ema;
///
/// let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let ema_20 = ema(&close, 20).unwrap();
/// ```
pub fn ema(series: &Series, period: usize) -> Result<Series> {
    let col_name = series.name();
    let alpha = 1.0 / period as f64;
    
    let ewm_opts = EWMOptions {
        alpha,
        adjust: true,
        bias: false,
        min_periods: period,
        ignore_nulls: true,
    };
    
    let df = DataFrame::new(vec![series.clone()])?;
    
    let result = df
        .lazy()
        .select([col(col_name).ewm_mean(ewm_opts).alias("ema")])
        .collect()?;
    
    Ok(result.column("ema")?.clone())
}
