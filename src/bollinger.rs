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
