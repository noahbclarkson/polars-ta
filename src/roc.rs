//! Rate of Change (ROC)
//!
//! ROC is a momentum oscillator that measures the percentage change in price
//! between the current price and the price n periods ago.

use anyhow::Result;
use polars::prelude::*;

/// Calculate Rate of Change (ROC)
///
/// ROC = ((current_price - price_n_periods_ago) / price_n_periods_ago) × 100
///
/// # Arguments
///
/// * `series` - Input price series
/// * `period` - Number of periods to look back
///
/// # Returns
///
/// A new Series containing ROC values (as percentage)
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::roc;
///
/// let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let roc_10 = roc(&close, 10).unwrap();
/// ```
pub fn roc(series: &Series, period: usize) -> Result<Series> {
    let col_name = series.name();
    
    let df = DataFrame::new(vec![series.clone()])?;
    
    let result = df
        .lazy()
        .with_column(
            ((col(col_name) - col(col_name).shift(lit(period as i64))) 
                / col(col_name).shift(lit(period as i64)) 
                * lit(100.0))
                .alias("roc"),
        )
        .collect()?;
    
    Ok(result.column("roc")?.clone())
}
