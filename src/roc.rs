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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roc_length() {
        let series = Series::new("close", (1..=20).map(|x| x as f64).collect::<Vec<_>>());
        let result = roc(&series, 10).unwrap();
        assert_eq!(result.len(), series.len());
    }

    #[test]
    fn test_roc_positive_in_uptrend() {
        let prices: Vec<f64> = (1..=20).map(|x| x as f64 * 2.0).collect();
        let series = Series::new("close", prices);
        let result = roc(&series, 5).unwrap();
        let vals = result.f64().unwrap();

        for i in 5..vals.len() {
            let v = vals.get(i).unwrap();
            assert!(v > 0.0, "ROC should be positive in uptrend at index {}, got {}", i, v);
        }
    }

    #[test]
    fn test_roc_zero_for_constant() {
        let series = Series::new("close", vec![100.0; 20]);
        let result = roc(&series, 5).unwrap();
        let vals = result.f64().unwrap();

        for i in 5..vals.len() {
            let v = vals.get(i).unwrap();
            assert!(v.abs() < 0.001, "ROC should be 0 for constant series, got {}", v);
        }
    }
}
