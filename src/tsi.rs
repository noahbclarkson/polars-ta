//! True Strength Index (TSI)
//!
//! TSI is a momentum oscillator that applies double exponential smoothing
//! to price changes and their absolute values, revealing trend direction
//! and magnitude while reducing noise.
//!
//! Formula:
//! 1. PC = close - close.shift(1)   (price change)
//! 2. double_smoothed_PC  = EMA(EMA(PC,  slow_period), fast_period)
//! 3. double_smoothed_APC = EMA(EMA(|PC|, slow_period), fast_period)
//! 4. TSI = 100 × double_smoothed_PC / double_smoothed_APC
//!
//! Values range roughly from -100 to +100.
//! Common periods: slow=25, fast=13. Signal line = EMA(TSI, signal_period).

use anyhow::Result;
use polars::prelude::*;

use crate::ema::ema;

/// Calculate True Strength Index (TSI)
///
/// # Arguments
///
/// * `df` - DataFrame containing a `close` column
/// * `slow_period` - Slow EMA period (standard: 25)
/// * `fast_period` - Fast EMA period (standard: 13)
///
/// # Returns
///
/// A Series named "tsi". The first `slow_period + fast_period - 1` values
/// will typically be null.
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::tsi;
///
/// let close = Series::new("close", &[100.0_f64; 80]);
/// let df = DataFrame::new(vec![close]).unwrap();
/// let tsi_vals = tsi(&df, 25, 13).unwrap();
/// ```
pub fn tsi(df: &DataFrame, slow_period: usize, fast_period: usize) -> Result<Series> {
    let close = df.column("close")?;
    let close_ca = close.f64()?;
    let n = close.len();

    // Build PC[i] = close[i] - close[i-1] and |PC[i]|
    let mut pc_values:     Vec<Option<f64>> = vec![None; n];
    let mut abs_pc_values: Vec<Option<f64>> = vec![None; n];

    for i in 1..n {
        if let (Some(curr), Some(prev)) = (close_ca.get(i), close_ca.get(i - 1)) {
            let delta = curr - prev;
            pc_values[i]     = Some(delta);
            abs_pc_values[i] = Some(delta.abs());
        }
    }

    // Series with "close" name so ema() can pick it up
    let pc_series     = Series::new("close", pc_values);
    let abs_pc_series = Series::new("close", abs_pc_values);

    // Double-smooth PC and |PC|
    let ema1_pc = ema(&pc_series, slow_period)?;
    let mut ema1_pc_named = ema1_pc;
    ema1_pc_named.rename("close");
    let double_pc = ema(&ema1_pc_named, fast_period)?;

    let ema1_abs_pc = ema(&abs_pc_series, slow_period)?;
    let mut ema1_abs_pc_named = ema1_abs_pc;
    ema1_abs_pc_named.rename("close");
    let double_abs_pc = ema(&ema1_abs_pc_named, fast_period)?;

    // TSI = 100 × double_pc / double_abs_pc
    let dp_ca  = double_pc.f64()?;
    let dap_ca = double_abs_pc.f64()?;

    let mut tsi_out: Vec<Option<f64>> = vec![None; n];

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        if let (Some(dp), Some(dap)) = (dp_ca.get(i), dap_ca.get(i)) {
            if dap != 0.0 {
                tsi_out[i] = Some(100.0 * dp / dap);
            }
        }
    }

    Ok(Series::new("tsi", tsi_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_df(prices: Vec<f64>) -> DataFrame {
        let close = Series::new("close", prices);
        DataFrame::new(vec![close]).unwrap()
    }

    #[test]
    fn test_tsi_output_length() {
        let df = make_df((0..80).map(|i| 100.0 + i as f64 * 0.3).collect());
        let result = tsi(&df, 25, 13).unwrap();
        assert_eq!(result.len(), 80);
    }

    #[test]
    fn test_tsi_initial_nulls() {
        let df = make_df((0..80).map(|i| 100.0 + i as f64 * 0.3).collect());
        let slow = 25_usize;
        let result = tsi(&df, slow, 13).unwrap();
        let f64_ca = result.f64().unwrap();
        // The first slow_period values should be null at minimum
        for i in 0..slow {
            assert!(f64_ca.get(i).is_none(), "Expected null at index {}", i);
        }
    }

    #[test]
    fn test_tsi_uptrend_positive() {
        // Consistent uptrend → positive price changes → TSI should be positive
        let df = make_df((0..100).map(|i| 50.0 + i as f64 * 1.0).collect());
        let result = tsi(&df, 25, 13).unwrap();
        let f64_ca = result.f64().unwrap();
        let mut found_value = false;
        for i in 50..100 {
            if let Some(v) = f64_ca.get(i) {
                assert!(v > 0.0, "TSI should be positive in uptrend, got {}", v);
                found_value = true;
            }
        }
        assert!(found_value, "Expected non-null TSI values in second half");
    }

    #[test]
    fn test_tsi_constant_is_zero() {
        // Constant prices → price change = 0 → TSI = 0 (or null if both smoothed = 0)
        let df = make_df(vec![100.0; 100]);
        let result = tsi(&df, 10, 5).unwrap();
        let f64_ca = result.f64().unwrap();
        for i in 20..100 {
            // Either null (0/0) or exactly 0
            if let Some(v) = f64_ca.get(i) {
                assert!(v.abs() < 1e-8, "Expected TSI~0 for constant prices, got {}", v);
            }
        }
    }
}
