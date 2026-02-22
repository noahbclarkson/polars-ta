//! TRIX — Triple Smoothed EMA Rate of Change
//!
//! TRIX is the 1-period percentage rate of change of a triple-smoothed EMA.
//! It filters out insignificant price movements ("triple-filtering").
//!
//! Formula:
//! 1. EMA1 = EMA(close, period)
//! 2. EMA2 = EMA(EMA1, period)
//! 3. EMA3 = EMA(EMA2, period)
//! 4. TRIX = (EMA3 - EMA3.shift(1)) / EMA3.shift(1) × 100

use anyhow::Result;
use polars::prelude::*;

use crate::ema::ema;

/// Calculate TRIX indicator
///
/// # Arguments
///
/// * `df` - DataFrame containing a `close` column
/// * `period` - EMA period (standard: 14 or 18)
///
/// # Returns
///
/// A Series named "trix". Roughly the first `3 * period` values will be null
/// (3 layers of EMA warm-up + 1 for the ROC).
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::trix;
///
/// let close = Series::new("close", &[100.0_f64; 50]);
/// let df = DataFrame::new(vec![close]).unwrap();
/// let trix_vals = trix(&df, 14).unwrap();
/// ```
pub fn trix(df: &DataFrame, period: usize) -> Result<Series> {
    let close = df.column("close")?;

    // Apply EMA three times, renaming to "close" each time (required by ema())
    let ema1 = ema(close, period)?;
    let mut ema1_for_ema2 = ema1.clone();
    ema1_for_ema2.rename("close");
    let ema2 = ema(&ema1_for_ema2, period)?;
    let mut ema2_for_ema3 = ema2.clone();
    ema2_for_ema3.rename("close");
    let ema3 = ema(&ema2_for_ema3, period)?;

    let n = ema3.len();
    let ema3_ca = ema3.f64()?;

    // TRIX[i] = (EMA3[i] - EMA3[i-1]) / EMA3[i-1] * 100
    let mut trix_out: Vec<Option<f64>> = vec![None; n];

    for i in 1..n {
        if let (Some(curr), Some(prev)) = (ema3_ca.get(i), ema3_ca.get(i - 1)) {
            if prev != 0.0 {
                trix_out[i] = Some((curr - prev) / prev * 100.0);
            }
        }
    }

    Ok(Series::new("trix", trix_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_df(prices: Vec<f64>) -> DataFrame {
        let close = Series::new("close", prices);
        DataFrame::new(vec![close]).unwrap()
    }

    #[test]
    fn test_trix_output_length() {
        let df = make_df((0..60).map(|i| 100.0 + i as f64 * 0.5).collect());
        let result = trix(&df, 10).unwrap();
        assert_eq!(result.len(), 60);
    }

    #[test]
    fn test_trix_initial_nulls() {
        let df = make_df((0..60).map(|i| 100.0 + i as f64 * 0.5).collect());
        let period = 10_usize;
        let result = trix(&df, period).unwrap();
        let f64_ca = result.f64().unwrap();
        // First period values must be null (at least period values from triple EMA)
        for i in 0..period {
            assert!(f64_ca.get(i).is_none(), "Expected null at {}", i);
        }
    }

    #[test]
    fn test_trix_constant_prices() {
        // Constant prices → EMA = price → no change → TRIX ≈ 0
        let df = make_df(vec![100.0; 60]);
        let result = trix(&df, 10).unwrap();
        let f64_ca = result.f64().unwrap();
        for i in 30..60 {
            if let Some(v) = f64_ca.get(i) {
                assert!(v.abs() < 1e-8, "TRIX should be ~0 for constant prices, got {}", v);
            }
        }
    }

    #[test]
    fn test_trix_trending_up() {
        // Strong uptrend → triple EMA rises → TRIX should be positive
        let df = make_df((0..80).map(|i| 50.0 + i as f64 * 2.0).collect());
        let result = trix(&df, 10).unwrap();
        let f64_ca = result.f64().unwrap();
        let mut found_positive = false;
        for i in 40..80 {
            if let Some(v) = f64_ca.get(i) {
                assert!(v >= 0.0, "TRIX should be non-negative in uptrend, got {}", v);
                found_positive = true;
            }
        }
        assert!(found_positive, "Expected at least some non-null TRIX values");
    }
}
