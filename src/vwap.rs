//! Volume Weighted Average Price (VWAP)
//!
//! VWAP is a trading benchmark used to gauge the average price at which a security
//! has traded throughout the day, based on both volume and price.

use anyhow::Result;
use polars::prelude::*;

/// Calculate Volume Weighted Average Price (VWAP)
///
/// VWAP = Σ(Price × Volume) / Σ(Volume)
///
/// Typically uses typical price = (high + low + close) / 3
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
/// A new Series containing VWAP values (cumulative from start)
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::vwap;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let volume = Series::new("volume", &[1000.0, 1200.0, 800.0, 1500.0, 900.0]);
/// let vwap_values = vwap(&high, &low, &close, &volume).unwrap();
/// ```
pub fn vwap(high: &Series, low: &Series, close: &Series, volume: &Series) -> Result<Series> {
    let df = DataFrame::new(vec![
        high.clone(),
        low.clone(),
        close.clone(),
        volume.clone(),
    ])?;

    let result = df
        .lazy()
        .with_column(
            ((col("high") + col("low") + col("close")) / lit(3.0))
                .alias("typical_price"),
        )
        .with_column((col("typical_price") * col("volume")).alias("tp_vol"))
        .with_column(col("tp_vol").cum_sum(false).alias("cum_tp_vol"))
        .with_column(col("volume").cum_sum(false).alias("cum_vol"))
        .with_column((col("cum_tp_vol") / col("cum_vol")).alias("vwap"))
        .collect()?;

    Ok(result.column("vwap")?.clone())
}

/// Calculate Rolling VWAP over a specified window
///
/// This calculates VWAP over a rolling window instead of cumulative.
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `volume` - Volume series
/// * `period` - Rolling window period
///
/// # Returns
///
/// A new Series containing rolling VWAP values
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::rolling_vwap;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let volume = Series::new("volume", &[1000.0, 1200.0, 800.0, 1500.0, 900.0]);
/// let vwap_values = rolling_vwap(&high, &low, &close, &volume, 20).unwrap();
/// ```
pub fn rolling_vwap(
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
        .with_column(
            ((col("high") + col("low") + col("close")) / lit(3.0))
                .alias("typical_price"),
        )
        .with_column((col("typical_price") * col("volume")).alias("tp_vol"))
        .with_column(col("tp_vol").rolling_sum(rolling_opts.clone()).alias("rolling_tp_vol"))
        .with_column(col("volume").rolling_sum(rolling_opts).alias("rolling_vol"))
        .with_column((col("rolling_tp_vol") / col("rolling_vol")).alias("vwap"))
        .collect()?;

    Ok(result.column("vwap")?.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vwap_returns_correct_length() {
        let high = Series::new("high".into(), &[105.0, 106.0, 107.0, 106.5, 108.0]);
        let low = Series::new("low".into(), &[100.0, 101.0, 102.0, 101.5, 103.0]);
        let close = Series::new("close".into(), &[103.0, 104.0, 105.0, 104.5, 106.0]);
        let volume = Series::new("volume".into(), &[1000.0, 1200.0, 800.0, 1500.0, 900.0]);
        
        let result = vwap(&high, &low, &close, &volume).unwrap();
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_vwap_first_value_equals_typical_price() {
        let high = Series::new("high".into(), &[105.0]);
        let low = Series::new("low".into(), &[100.0]);
        let close = Series::new("close".into(), &[103.0]);
        let volume = Series::new("volume".into(), &[1000.0]);
        
        let typical_price = (105.0 + 100.0 + 103.0) / 3.0;
        let result = vwap(&high, &low, &close, &volume).unwrap();
        
        let vwap_val = result.f64().unwrap().get(0).unwrap();
        assert!((vwap_val - typical_price).abs() < 0.0001);
    }

    #[test]
    fn test_rolling_vwap_returns_correct_length() {
        let high = Series::new("high".into(), &[105.0, 106.0, 107.0, 106.5, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0]);
        let low = Series::new("low".into(), &[100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0]);
        let close = Series::new("close".into(), &[103.0, 104.0, 105.0, 104.5, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0]);
        let volume = Series::new("volume".into(), &[1000.0; 22]);
        
        let result = rolling_vwap(&high, &low, &close, &volume, 20).unwrap();
        assert_eq!(result.len(), close.len());
    }
}
