//! Bollinger Bands
//!
//! Bollinger Bands are volatility bands placed above and below a moving average.
//! They expand and contract based on market volatility.

use anyhow::Result;
use polars::prelude::*;

/// Result of Bollinger Bands calculation
#[derive(Debug)]
pub struct BollingerResult {
    /// Upper band (middle + std_dev * rolling_std)
    pub upper: Series,
    /// Middle band (SMA)
    pub middle: Series,
    /// Lower band (middle - std_dev * rolling_std)
    pub lower: Series,
}

/// Calculate Bollinger Bands
///
/// Middle = SMA(period)
/// Upper = Middle + (std_dev × Rolling Std)
/// Lower = Middle - (std_dev × Rolling Std)
///
/// # Arguments
///
/// * `series` - Input price series
/// * `period` - Period for SMA and rolling std (standard: 20)
/// * `std_dev` - Number of standard deviations (standard: 2.0)
///
/// # Returns
///
/// A `BollingerResult` containing upper, middle, and lower bands
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::bollinger;
///
/// let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let bands = bollinger(&close, 20, 2.0).unwrap();
/// println!("Upper: {:?}", bands.upper);
/// println!("Middle: {:?}", bands.middle);
/// println!("Lower: {:?}", bands.lower);
/// ```
pub fn bollinger(series: &Series, period: usize, std_dev: f64) -> Result<BollingerResult> {
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
        .with_column(col(col_name).rolling_mean(rolling_opts.clone()).alias("middle"))
        .with_column(col(col_name).rolling_std(rolling_opts).alias("std"))
        .with_column((col("middle") + lit(std_dev) * col("std")).alias("upper"))
        .with_column((col("middle") - lit(std_dev) * col("std")).alias("lower"))
        .collect()?;
    
    Ok(BollingerResult {
        upper: result.column("upper")?.clone(),
        middle: result.column("middle")?.clone(),
        lower: result.column("lower")?.clone(),
    })
}

/// Calculate Bollinger Bands with standard 20-period, 2 std dev settings
///
/// # Arguments
///
/// * `series` - Input price series
///
/// # Returns
///
/// A `BollingerResult` containing upper, middle, and lower bands
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::bollinger_20_2;
///
/// let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let bands = bollinger_20_2(&close).unwrap();
/// ```
pub fn bollinger_20_2(series: &Series) -> Result<BollingerResult> {
    bollinger(series, 20, 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bollinger_length() {
        let prices: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let series = Series::new("close", prices);
        let result = bollinger(&series, 20, 2.0).unwrap();
        assert_eq!(result.upper.len(), series.len());
        assert_eq!(result.middle.len(), series.len());
        assert_eq!(result.lower.len(), series.len());
    }

    #[test]
    fn test_bollinger_upper_above_lower() {
        let prices: Vec<f64> = (1..=30).map(|x| (x as f64 * 0.8).sin() * 5.0 + 100.0).collect();
        let series = Series::new("close", prices);
        let result = bollinger_20_2(&series).unwrap();

        let upper = result.upper.f64().unwrap();
        let lower = result.lower.f64().unwrap();

        for i in 20..upper.len() {
            let u = upper.get(i).unwrap();
            let l = lower.get(i).unwrap();
            assert!(u > l, "Upper band must be above lower band at index {}", i);
        }
    }

    #[test]
    fn test_bollinger_middle_is_sma() {
        let prices = vec![100.0; 25]; // constant
        let series = Series::new("close", prices);
        let result = bollinger_20_2(&series).unwrap();

        let middle = result.middle.f64().unwrap();
        let val = middle.get(24).unwrap();
        assert!((val - 100.0).abs() < 0.01, "Middle band should equal SMA for constant series");
    }

    #[test]
    fn test_bollinger_constant_series_bands_equal() {
        let prices = vec![50.0; 25];
        let series = Series::new("close", prices);
        let result = bollinger_20_2(&series).unwrap();

        let upper = result.upper.f64().unwrap();
        let lower = result.lower.f64().unwrap();
        let middle = result.middle.f64().unwrap();

        // Constant price: std dev = 0, so all three bands equal
        let u = upper.get(24).unwrap();
        let l = lower.get(24).unwrap();
        let m = middle.get(24).unwrap();
        assert!((u - m).abs() < 0.01, "Upper and middle should be equal for constant series");
        assert!((l - m).abs() < 0.01, "Lower and middle should be equal for constant series");
    }
}
