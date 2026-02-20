//! Stochastic Oscillator
//!
//! The Stochastic Oscillator is a momentum indicator comparing a particular closing price
//! of a security to a range of its prices over a certain period of time.

use anyhow::Result;
use polars::prelude::*;

/// Result of Stochastic Oscillator calculation
#[derive(Debug)]
pub struct StochasticResult {
    /// %K line (fast stochastic) - the raw oscillator
    pub k: Series,
    /// %D line (slow stochastic) - SMA of %K
    pub d: Series,
}

/// Calculate Stochastic Oscillator
///
/// %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) × 100
/// %D = SMA(%K, smooth_period)
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `k_period` - Period for %K calculation (standard: 14)
/// * `d_period` - Period for %D smoothing (standard: 3)
///
/// # Returns
///
/// A `StochasticResult` containing the %K and %D lines
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::stochastic;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let stoch = stochastic(&high, &low, &close, 14, 3).unwrap();
/// println!("%K: {:?}", stoch.k);
/// println!("%D: {:?}", stoch.d);
/// ```
pub fn stochastic(
    high: &Series,
    low: &Series,
    close: &Series,
    k_period: usize,
    d_period: usize,
) -> Result<StochasticResult> {
    let rolling_opts = RollingOptions {
        window_size: Duration::new(k_period as i64),
        min_periods: k_period,
        weights: None,
        center: false,
        by: None,
        closed_window: None,
        warn_if_unsorted: true,
        fn_params: None,
    };

    let rolling_opts_d = RollingOptions {
        window_size: Duration::new(d_period as i64),
        min_periods: d_period,
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
            ((col("close") - col("lowest_low")) 
                / (col("highest_high") - col("lowest_low")) 
                * lit(100.0))
                .alias("k_raw"),
        )
        .with_column(col("k_raw").rolling_mean(rolling_opts_d).alias("d"))
        .with_column(col("k_raw").alias("k"))
        .collect()?;

    Ok(StochasticResult {
        k: result.column("k")?.clone(),
        d: result.column("d")?.clone(),
    })
}

/// Calculate Stochastic Oscillator with standard 14/3 settings
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
///
/// # Returns
///
/// A `StochasticResult` containing the %K and %D lines
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::stochastic_14_3;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let stoch = stochastic_14_3(&high, &low, &close).unwrap();
/// ```
pub fn stochastic_14_3(high: &Series, low: &Series, close: &Series) -> Result<StochasticResult> {
    stochastic(high, low, close, 14, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stochastic_returns_correct_length() {
        let high = Series::new("high".into(), &[105.0, 106.0, 107.0, 106.5, 108.0, 109.0, 108.5, 110.0, 111.0, 110.5, 112.0, 113.0, 112.5, 114.0, 115.0]);
        let low = Series::new("low".into(), &[100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0, 108.0, 107.5, 109.0, 110.0]);
        let close = Series::new("close".into(), &[103.0, 104.0, 105.0, 104.5, 106.0, 107.0, 106.5, 108.0, 109.0, 108.5, 110.0, 111.0, 110.5, 112.0, 113.0]);
        
        let result = stochastic_14_3(&high, &low, &close).unwrap();
        
        assert_eq!(result.k.len(), close.len());
        assert_eq!(result.d.len(), close.len());
    }

    #[test]
    fn test_stochastic_values_between_0_and_100() {
        let high = Series::new("high".into(), &[105.0, 106.0, 107.0, 106.5, 108.0, 109.0, 108.5, 110.0, 111.0, 110.5, 112.0, 113.0, 112.5, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0]);
        let low = Series::new("low".into(), &[100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0, 108.0, 107.5, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0]);
        let close = Series::new("close".into(), &[103.0, 104.0, 105.0, 104.5, 106.0, 107.0, 106.5, 108.0, 109.0, 108.5, 110.0, 111.0, 110.5, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0]);
        
        let result = stochastic_14_3(&high, &low, &close).unwrap();
        
        for i in 0..result.k.len() {
            if let Some(val) = result.k.f64().unwrap().get(i) {
                if !val.is_nan() {
                    assert!(val >= 0.0 && val <= 100.0, "K value {} at index {} out of range", val, i);
                }
            }
        }
    }
}
