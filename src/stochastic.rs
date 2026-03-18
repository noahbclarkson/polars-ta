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

    fn make_data(n: usize) -> (Series, Series, Series) {
        let high = Series::new("high", (0..n).map(|i| 105.0 + i as f64).collect::<Vec<_>>());
        let low = Series::new("low", (0..n).map(|i| 100.0 + i as f64).collect::<Vec<_>>());
        let close = Series::new("close", (0..n).map(|i| 103.0 + i as f64).collect::<Vec<_>>());
        (high, low, close)
    }

    #[test]
    fn test_stochastic_returns_correct_length() {
        let (high, low, close) = make_data(15);
        let result = stochastic_14_3(&high, &low, &close).unwrap();
        assert_eq!(result.k.len(), 15);
        assert_eq!(result.d.len(), 15);
    }

    #[test]
    fn test_stochastic_values_between_0_and_100() {
        let (high, low, close) = make_data(20);
        let result = stochastic_14_3(&high, &low, &close).unwrap();

        for i in 0..result.k.len() {
            if let Some(val) = result.k.f64().unwrap().get(i) {
                if !val.is_nan() {
                    assert!((0.0..=100.0).contains(&val), "K value {} at index {} out of range", val, i);
                }
            }
        }
    }

    #[test]
    fn test_stochastic_custom_params() {
        let (high, low, close) = make_data(20);
        let result = stochastic(&high, &low, &close, 5, 3).unwrap();
        assert_eq!(result.k.len(), 20);
        assert_eq!(result.d.len(), 20);
    }

    #[test]
    fn test_stochastic_close_at_high_gives_100() {
        // When close == high == rolling_high, %K should be 100
        let n = 20;
        let high = Series::new("high", vec![100.0_f64; n]);
        let low = Series::new("low", vec![90.0_f64; n]);
        let close = Series::new("close", vec![100.0_f64; n]); // at the high

        let result = stochastic(&high, &low, &close, 5, 3).unwrap();
        let k_ca = result.k.f64().unwrap();

        // Last value should be 100 (close == high of the range)
        if let Some(val) = k_ca.get(n - 1) {
            if !val.is_nan() {
                assert!((val - 100.0).abs() < 1e-9, "K should be 100 when close == high, got {}", val);
            }
        }
    }

    #[test]
    fn test_stochastic_close_at_low_gives_zero() {
        // When close == low == rolling_low, %K should be 0
        let n = 20;
        let high = Series::new("high", vec![110.0_f64; n]);
        let low = Series::new("low", vec![100.0_f64; n]);
        let close = Series::new("close", vec![100.0_f64; n]); // at the low

        let result = stochastic(&high, &low, &close, 5, 3).unwrap();
        let k_ca = result.k.f64().unwrap();

        if let Some(val) = k_ca.get(n - 1) {
            if !val.is_nan() {
                assert!(val.abs() < 1e-9, "K should be 0 when close == low, got {}", val);
            }
        }
    }

    #[test]
    fn test_stochastic_d_is_smoothed_k() {
        // D should have fewer non-NaN values than K (due to extra smoothing)
        let (high, low, close) = make_data(20);
        let result = stochastic_14_3(&high, &low, &close).unwrap();

        let k_non_null = result.k.f64().unwrap().into_iter().filter(|v| v.map(|x| !x.is_nan()).unwrap_or(false)).count();
        let d_non_null = result.d.f64().unwrap().into_iter().filter(|v| v.map(|x| !x.is_nan()).unwrap_or(false)).count();

        // D needs more warmup than K (k_period + d_period - 1 total)
        assert!(d_non_null <= k_non_null, "D should have fewer valid values than K");
    }
}
