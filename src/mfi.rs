//! Money Flow Index (MFI)
//!
//! MFI is a volume-weighted RSI that uses both price and volume to measure
//! buying and selling pressure. Often called the "volume-weighted RSI".
//!
//! Formula:
//! 1. Typical Price (TP) = (high + low + close) / 3
//! 2. Raw Money Flow = TP × volume
//! 3. PMF = sum of positive RMF in period (when TP > previous TP)
//! 4. NMF = sum of negative RMF in period (when TP < previous TP)
//! 5. MFR = PMF / NMF
//! 6. MFI = 100 - 100 / (1 + MFR)
//!
//! Values range 0-100. Above 80 = overbought, below 20 = oversold.

use anyhow::Result;
use polars::prelude::*;

/// Calculate Money Flow Index
///
/// # Arguments
///
/// * `df` - DataFrame containing `high`, `low`, `close`, `volume` columns
/// * `period` - Lookback period (standard: 14)
///
/// # Returns
///
/// A Series named "mfi" with values in the range [0, 100].
/// The first `period` values are null.
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::mfi;
///
/// let high   = Series::new("high",   &[105.0, 106.0, 107.0, 108.0, 107.5]);
/// let low    = Series::new("low",    &[100.0, 101.0, 102.0, 103.0, 102.5]);
/// let close  = Series::new("close",  &[103.0, 104.0, 105.0, 106.0, 105.5]);
/// let volume = Series::new("volume", &[1000.0, 1200.0, 900.0, 1500.0, 800.0]);
/// let df = DataFrame::new(vec![high, low, close, volume]).unwrap();
/// let mfi_values = mfi(&df, 14).unwrap();
/// ```
pub fn mfi(df: &DataFrame, period: usize) -> Result<Series> {
    let high   = df.column("high")?;
    let low    = df.column("low")?;
    let close  = df.column("close")?;
    let volume = df.column("volume")?;

    let high_ca   = high.f64()?;
    let low_ca    = low.f64()?;
    let close_ca  = close.f64()?;
    let volume_ca = volume.f64()?;

    let n = high.len();

    // Step 1: Calculate typical prices and raw money flows
    let mut tp: Vec<f64>  = Vec::with_capacity(n);
    let mut rmf: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let h = high_ca.get(i).unwrap_or(f64::NAN);
        let l = low_ca.get(i).unwrap_or(f64::NAN);
        let c = close_ca.get(i).unwrap_or(f64::NAN);
        let v = volume_ca.get(i).unwrap_or(0.0);
        let t = (h + l + c) / 3.0;
        tp.push(t);
        rmf.push(t * v);
    }

    // Step 2: Compute MFI for each bar (from `period` onward)
    let mut mfi_out: Vec<Option<f64>> = vec![None; n];

    #[allow(clippy::needless_range_loop)]
    for i in period..n {
        let mut pos_mf = 0.0_f64;
        let mut neg_mf = 0.0_f64;

        // j ranges from (i - period + 1) to i; compare tp[j] with tp[j-1]
        // j starts at 1 to ensure j-1 >= 0
        let start = if i + 1 >= period { i + 1 - period } else { 1 };
        let start = start.max(1);

        for j in start..=i {
            if tp[j] > tp[j - 1] {
                pos_mf += rmf[j];
            } else if tp[j] < tp[j - 1] {
                neg_mf += rmf[j];
            }
        }

        let val = if neg_mf == 0.0 {
            100.0
        } else {
            let mfr = pos_mf / neg_mf;
            100.0 - 100.0 / (1.0 + mfr)
        };

        mfi_out[i] = Some(val);
    }

    Ok(Series::new("mfi", mfi_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_df(n: usize) -> DataFrame {
        // Prices trending up: tp increases monotonically → all positive money flow
        let high:   Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 1.5).collect();
        let low:    Vec<f64> = (0..n).map(|i| 98.0 + i as f64 * 1.5).collect();
        let close:  Vec<f64> = (0..n).map(|i| 99.0 + i as f64 * 1.5).collect();
        let volume: Vec<f64> = vec![1000.0; n];
        DataFrame::new(vec![
            Series::new("high",   high),
            Series::new("low",    low),
            Series::new("close",  close),
            Series::new("volume", volume),
        ])
        .unwrap()
    }

    #[test]
    fn test_mfi_output_length() {
        let df = make_df(30);
        let result = mfi(&df, 14).unwrap();
        assert_eq!(result.len(), 30);
    }

    #[test]
    fn test_mfi_initial_nulls() {
        let df = make_df(20);
        let result = mfi(&df, 14).unwrap();
        let f64_ca = result.f64().unwrap();
        // Positions 0..13 (indices 0 to 13) should be null
        for i in 0..14 {
            assert!(f64_ca.get(i).is_none(), "Expected null at index {}", i);
        }
    }

    #[test]
    fn test_mfi_all_up_trend() {
        // All prices rising → all positive money flow → MFI should be 100
        let df = make_df(20);
        let result = mfi(&df, 14).unwrap();
        let f64_ca = result.f64().unwrap();
        for i in 14..20 {
            let v = f64_ca.get(i).expect("Expected non-null value");
            assert!((v - 100.0).abs() < 0.001, "Expected ~100, got {}", v);
        }
    }

    #[test]
    fn test_mfi_range() {
        // MFI should always be in [0, 100]
        let df = make_df(50);
        let result = mfi(&df, 10).unwrap();
        let f64_ca = result.f64().unwrap();
        for i in 10..50 {
            if let Some(v) = f64_ca.get(i) {
                assert!(v >= 0.0 && v <= 100.0, "MFI out of range: {}", v);
            }
        }
    }
}
