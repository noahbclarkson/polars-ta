//! Williams %R
//!
//! Williams %R is a momentum indicator that measures overbought and oversold levels,
//! ranging from -100 to 0.

use anyhow::Result;
use polars::prelude::*;

/// Calculate Williams %R
///
/// %R = (Highest High - Close) / (Highest High - Lowest Low) × -100
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `period` - Lookback period (standard: 14)
///
/// # Returns
///
/// A new Series containing Williams %R values (range: -100 to 0)
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::williams_r;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let wr = williams_r(&high, &low, &close, 14).unwrap();
/// ```
pub fn williams_r(high: &Series, low: &Series, close: &Series, period: usize) -> Result<Series> {
    let rolling_opts = RollingOptions {
        window_size: Duration::new(period as i64),
        min_periods: period,
        weights: None,
        center: false,
        by: None,
        closed_window: None,
        warn_if_unsorted: true,
        fn_params: None,
    };

    let df = DataFrame::new(vec![
        high.clone(),
        low.clone(),
        close.clone(),
    ])?;

    let result = df
        .lazy()
        .with_column(col("high").rolling_max(rolling_opts.clone()).alias("highest_high"))
        .with_column(col("low").rolling_min(rolling_opts).alias("lowest_low"))
        .with_column(
            ((col("highest_high") - col("close")) 
                / (col("highest_high") - col("lowest_low")) 
                * lit(-100.0))
                .alias("williams_r"),
        )
        .collect()?;

    Ok(result.column("williams_r")?.clone())
}

/// Calculate Williams %R with the standard 14-period setting
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
///
/// # Returns
///
/// A new Series containing Williams %R values (range: -100 to 0)
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::williams_r_14;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let wr = williams_r_14(&high, &low, &close).unwrap();
/// ```
pub fn williams_r_14(high: &Series, low: &Series, close: &Series) -> Result<Series> {
    williams_r(high, low, close, 14)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_williams_r_returns_correct_length() {
        let high = Series::new("high", (0..20).map(|i| 105.0 + i as f64).collect::<Vec<_>>());
        let low = Series::new("low", (0..20).map(|i| 100.0 + i as f64).collect::<Vec<_>>());
        let close = Series::new("close", (0..20).map(|i| 103.0 + i as f64).collect::<Vec<_>>());
        
        let result = williams_r_14(&high, &low, &close).unwrap();
        
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_williams_r_values_in_range() {
        let high = Series::new("high", (0..20).map(|i| 105.0 + i as f64).collect::<Vec<_>>());
        let low = Series::new("low", (0..20).map(|i| 100.0 + i as f64).collect::<Vec<_>>());
        let close = Series::new("close", (0..20).map(|i| 103.0 + i as f64).collect::<Vec<_>>());
        
        let result = williams_r_14(&high, &low, &close).unwrap();
        let wr_ca = result.f64().unwrap();
        
        for i in 0..wr_ca.len() {
            if let Some(val) = wr_ca.get(i) {
                if !val.is_nan() {
                    assert!((-100.0..=0.0).contains(&val), 
                        "Williams %R value {} at index {} should be between -100 and 0", val, i);
                }
            }
        }
    }

    #[test]
    fn test_williams_r_at_high_is_zero() {
        // When close equals highest high, %R should be 0
        let high = Series::new("high", &[110.0; 20]);
        let low = Series::new("low", &[100.0; 20]);
        let close = Series::new("close", &[110.0; 20]); // At highest high
        
        let result = williams_r(&high, &low, &close, 14).unwrap();
        let wr_ca = result.f64().unwrap();
        
        if let Some(val) = wr_ca.get(19) {
            if !val.is_nan() {
                assert!((val - 0.0).abs() < 0.001, 
                    "Williams %R should be 0 when close is at highest high, got {}", val);
            }
        }
    }

    #[test]
    fn test_williams_r_at_low_is_minus_100() {
        // When close equals lowest low, %R should be -100
        let high = Series::new("high", &[110.0; 20]);
        let low = Series::new("low", &[100.0; 20]);
        let close = Series::new("close", &[100.0; 20]); // At lowest low
        
        let result = williams_r(&high, &low, &close, 14).unwrap();
        let wr_ca = result.f64().unwrap();
        
        if let Some(val) = wr_ca.get(19) {
            if !val.is_nan() {
                assert!((val - (-100.0)).abs() < 0.001, 
                    "Williams %R should be -100 when close is at lowest low, got {}", val);
            }
        }
    }
}
