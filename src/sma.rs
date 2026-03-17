//! Simple Moving Average (SMA)
//!
//! The SMA is an unweighted mean of the previous n data points.

use anyhow::Result;
use polars::prelude::*;

/// Calculate Simple Moving Average
///
/// Uses a rolling window with min_periods = period
///
/// # Arguments
///
/// * `series` - Input price series
/// * `period` - SMA period (e.g., 20, 50, 200)
///
/// # Returns
///
/// A new Series containing the SMA values
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::sma;
///
/// let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let sma_20 = sma(&close, 20).unwrap();
/// ```
pub fn sma(series: &Series, period: usize) -> Result<Series> {
    let col_name = series.name();
    
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
    
    let df = DataFrame::new(vec![series.clone()])?;
    
    let result = df
        .lazy()
        .select([col(col_name).rolling_mean(rolling_opts).alias("sma")])
        .collect()?;
    
    Ok(result.column("sma")?.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_length() {
        let series = Series::new("close", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let result = sma(&series, 3).unwrap();
        assert_eq!(result.len(), series.len());
    }

    #[test]
    fn test_sma_simple_average() {
        let series = Series::new("close", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = sma(&series, 3).unwrap();
        let vals = result.f64().unwrap();

        // SMA(3) at index 2 = (1+2+3)/3 = 2.0
        let val = vals.get(2).unwrap();
        assert!((val - 2.0).abs() < 0.01, "SMA(3) of [1,2,3] should be 2.0, got {}", val);

        // SMA(3) at index 4 = (3+4+5)/3 = 4.0
        let val = vals.get(4).unwrap();
        assert!((val - 4.0).abs() < 0.01, "SMA(3) of [3,4,5] should be 4.0, got {}", val);
    }

    #[test]
    fn test_sma_constant_series() {
        let series = Series::new("close", vec![7.0; 10]);
        let result = sma(&series, 5).unwrap();
        let vals = result.f64().unwrap();

        for i in 4..10 {
            let val = vals.get(i).unwrap();
            assert!((val - 7.0).abs() < 0.01, "SMA of constant should equal constant");
        }
    }

    #[test]
    fn test_sma_uptrend_increasing() {
        let prices: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let series = Series::new("close", prices);
        let result = sma(&series, 5).unwrap();
        let vals = result.f64().unwrap();

        // SMA should increase in an uptrend
        let early = vals.get(5).unwrap();
        let late = vals.get(18).unwrap();
        assert!(late > early, "SMA should increase in uptrend");
    }
}
