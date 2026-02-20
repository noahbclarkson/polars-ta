//! Chaikin Money Flow (CMF)
//!
//! CMF is a volume-based indicator that measures buying and selling pressure
//! over a specified period, ranging from -1 to 1.

use anyhow::Result;
use polars::prelude::*;

/// Calculate Chaikin Money Flow (CMF)
///
/// Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
/// Money Flow Volume = MFM × Volume
/// CMF = Sum(MFV, period) / Sum(Volume, period)
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `volume` - Volume series
/// * `period` - Period for calculations (standard: 20)
///
/// # Returns
///
/// A new Series containing CMF values (range: -1 to 1)
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::cmf;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let volume = Series::new("volume", &[1000.0, 1200.0, 1100.0, 900.0, 1500.0]);
/// let cmf_values = cmf(&high, &low, &close, &volume, 20).unwrap();
/// ```
pub fn cmf(
    high: &Series,
    low: &Series,
    close: &Series,
    volume: &Series,
    period: usize,
) -> Result<Series> {
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
        volume.clone(),
    ])?;

    let result = df
        .lazy()
        // Calculate Money Flow Multiplier
        // MFM = ((Close - Low) - (High - Close)) / (High - Low)
        .with_column(
            (((col("close") - col("low")) - (col("high") - col("close")))
                / (col("high") - col("low")))
                .alias("mfm"),
        )
        // Calculate Money Flow Volume
        .with_column(
            (col("mfm") * col("volume")).alias("mfv"),
        )
        // Calculate rolling sums
        .with_column(
            col("mfv").rolling_sum(rolling_opts.clone()).alias("mfv_sum"),
        )
        .with_column(
            col("volume").rolling_sum(rolling_opts).alias("vol_sum"),
        )
        // Calculate CMF
        .with_column(
            (col("mfv_sum") / col("vol_sum")).alias("cmf"),
        )
        .collect()?;

    Ok(result.column("cmf")?.clone())
}

/// Calculate CMF with the standard 20-period setting
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `volume` - Volume series
///
/// # Returns
///
/// A new Series containing CMF values (range: -1 to 1)
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::cmf_20;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let volume = Series::new("volume", &[1000.0, 1200.0, 1100.0, 900.0, 1500.0]);
/// let cmf = cmf_20(&high, &low, &close, &volume).unwrap();
/// ```
pub fn cmf_20(
    high: &Series,
    low: &Series,
    close: &Series,
    volume: &Series,
) -> Result<Series> {
    cmf(high, low, close, volume, 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmf_returns_correct_length() {
        let high = Series::new("high".into(), (0..25).map(|i| 105.0 + i as f64).collect::<Vec<_>>());
        let low = Series::new("low".into(), (0..25).map(|i| 100.0 + i as f64).collect::<Vec<_>>());
        let close = Series::new("close".into(), (0..25).map(|i| 103.0 + i as f64).collect::<Vec<_>>());
        let volume = Series::new("volume".into(), (0..25).map(|i| 1000.0 + i as f64 * 10.0).collect::<Vec<_>>());
        
        let result = cmf_20(&high, &low, &close, &volume).unwrap();
        
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_cmf_values_in_range() {
        let high = Series::new("high".into(), (0..25).map(|i| 105.0 + i as f64).collect::<Vec<_>>());
        let low = Series::new("low".into(), (0..25).map(|i| 100.0 + i as f64).collect::<Vec<_>>());
        let close = Series::new("close".into(), (0..25).map(|i| 103.0 + i as f64).collect::<Vec<_>>());
        let volume = Series::new("volume".into(), vec![1000.0; 25]);
        
        let result = cmf_20(&high, &low, &close, &volume).unwrap();
        let cmf_ca = result.f64().unwrap();
        
        for i in 0..cmf_ca.len() {
            if let Some(val) = cmf_ca.get(i) {
                if !val.is_nan() {
                    assert!(val >= -1.0 && val <= 1.0, 
                        "CMF value {} at index {} should be between -1 and 1", val, i);
                }
            }
        }
    }

    #[test]
    fn test_cmf_at_high_positive() {
        // When close is at high (strong buying), CMF should be positive
        let high = Series::new("high".into(), &[110.0; 25]);
        let low = Series::new("low".into(), &[100.0; 25]);
        let close = Series::new("close".into(), &[110.0; 25]); // At high
        let volume = Series::new("volume".into(), vec![1000.0; 25]);
        
        let result = cmf(&high, &low, &close, &volume, 20).unwrap();
        let cmf_ca = result.f64().unwrap();
        
        if let Some(val) = cmf_ca.get(24) {
            if !val.is_nan() {
                assert!((val - 1.0).abs() < 0.001, 
                    "CMF should be 1.0 when close is always at high, got {}", val);
            }
        }
    }

    #[test]
    fn test_cmf_at_low_negative() {
        // When close is at low (strong selling), CMF should be negative
        let high = Series::new("high".into(), &[110.0; 25]);
        let low = Series::new("low".into(), &[100.0; 25]);
        let close = Series::new("close".into(), &[100.0; 25]); // At low
        let volume = Series::new("volume".into(), vec![1000.0; 25]);
        
        let result = cmf(&high, &low, &close, &volume, 20).unwrap();
        let cmf_ca = result.f64().unwrap();
        
        if let Some(val) = cmf_ca.get(24) {
            if !val.is_nan() {
                assert!((val - (-1.0)).abs() < 0.001, 
                    "CMF should be -1.0 when close is always at low, got {}", val);
            }
        }
    }
}
