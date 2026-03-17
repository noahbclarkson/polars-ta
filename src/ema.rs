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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_returns_correct_length() {
        let series = Series::new("close", &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]);
        let result = ema(&series, 5).unwrap();
        assert_eq!(result.len(), series.len());
    }

    #[test]
    fn test_ema_values_in_uptrend() {
        let prices: Vec<f64> = (1..=20).map(|x| x as f64 * 10.0).collect();
        let series = Series::new("close", prices);
        let result = ema(&series, 5).unwrap();
        let vals = result.f64().unwrap();

        // In a strict uptrend, later EMA values should be higher
        let last = vals.get(vals.len() - 1).unwrap();
        let mid = vals.get(10).unwrap();
        assert!(last > mid, "EMA should increase in uptrend");
    }

    #[test]
    fn test_ema_different_periods() {
        let prices: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let series = Series::new("close", prices);

        let ema5 = ema(&series, 5).unwrap();
        let ema20 = ema(&series, 20).unwrap();

        let ema5_vals = ema5.f64().unwrap();
        let ema20_vals = ema20.f64().unwrap();

        // In uptrend, faster EMA (shorter period) should be higher
        let last_5 = ema5_vals.get(ema5_vals.len() - 1).unwrap();
        let last_20 = ema20_vals.get(ema20_vals.len() - 1).unwrap();
        assert!(last_5 > last_20, "Faster EMA should be above slower in uptrend");
    }

    #[test]
    fn test_ema_constant_series() {
        let series = Series::new("close", vec![50.0; 20]);
        let result = ema(&series, 10).unwrap();
        let vals = result.f64().unwrap();

        for i in 10..20 {
            let val = vals.get(i).unwrap();
            assert!((val - 50.0).abs() < 0.01, "EMA of constant series should equal constant");
        }
    }
}
