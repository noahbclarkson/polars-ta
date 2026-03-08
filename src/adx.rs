//! Average Directional Index (ADX)
//!
//! ADX is a technical indicator used to measure the strength of a trend,
//! not the direction. Values range from 0 to 100.

use anyhow::Result;
use polars::prelude::*;

/// Result of ADX calculation
#[derive(Debug)]
pub struct AdxResult {
    /// ADX line - trend strength (0-100)
    pub adx: Series,
    /// +DI line - positive directional indicator
    pub plus_di: Series,
    /// -DI line - negative directional indicator
    pub minus_di: Series,
}

/// Calculate Average Directional Index (ADX)
///
/// ADX measures trend strength on a scale of 0-100:
/// - 0-25: Weak or no trend
/// - 25-50: Strong trend
/// - 50-75: Very strong trend
/// - 75-100: Extremely strong trend
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `period` - Period for calculations (standard: 14)
///
/// # Returns
///
/// An `AdxResult` containing ADX, +DI, and -DI lines
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::adx;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let result = adx(&high, &low, &close, 14).unwrap();
/// println!("ADX: {:?}", result.adx);
/// ```
pub fn adx(high: &Series, low: &Series, close: &Series, period: usize) -> Result<AdxResult> {
    let ewm_opts = EWMOptions {
        alpha: 1.0 / period as f64,
        adjust: true,
        bias: false,
        min_periods: period,
        ignore_nulls: true,
    };

    let df = DataFrame::new(vec![
        high.clone(),
        low.clone(),
        close.clone(),
    ])?;

    let result = df
        .lazy()
        // Calculate +DM and -DM
        .with_column(col("high").diff(1, Default::default()).alias("high_diff"))
        .with_column(col("low").diff(1, Default::default()).alias("low_diff"))
        .with_column(
            when(col("high_diff").gt(0.0).and(col("high_diff").gt(col("low_diff").abs())))
                .then(col("high_diff"))
                .otherwise(lit(0.0))
                .alias("plus_dm"),
        )
        .with_column(
            when(col("low_diff").lt(0.0).and(col("low_diff").abs().gt(col("high_diff"))))
                .then(col("low_diff").abs())
                .otherwise(lit(0.0))
                .alias("minus_dm"),
        )
        // Calculate True Range
        .with_columns(vec![
            (col("high") - col("low")).alias("tr1"),
            (col("high") - col("close").shift(lit(1))).abs().alias("tr2"),
            (col("low") - col("close").shift(lit(1))).abs().alias("tr3"),
        ])
        .with_column(
            when(col("tr1").gt_eq(col("tr2")).and(col("tr1").gt_eq(col("tr3"))))
                .then(col("tr1"))
                .when(col("tr2").gt_eq(col("tr3")))
                .then(col("tr2"))
                .otherwise(col("tr3"))
                .alias("tr"),
        )
        // Smooth the values using EWM
        .with_column(col("tr").ewm_mean(ewm_opts).alias("smoothed_tr"))
        .with_column(col("plus_dm").ewm_mean(ewm_opts).alias("smoothed_plus_dm"))
        .with_column(col("minus_dm").ewm_mean(ewm_opts).alias("smoothed_minus_dm"))
        // Calculate DI
        .with_column(
            (col("smoothed_plus_dm") / col("smoothed_tr") * lit(100.0))
                .alias("plus_di"),
        )
        .with_column(
            (col("smoothed_minus_dm") / col("smoothed_tr") * lit(100.0))
                .alias("minus_di"),
        )
        // Calculate DX
        .with_column(
            ((col("plus_di") - col("minus_di")).abs() 
                / (col("plus_di") + col("minus_di")) 
                * lit(100.0))
                .alias("dx"),
        )
        // Smooth DX to get ADX
        .with_column(
            when(col("dx").is_not_nan())
                .then(col("dx").ewm_mean(EWMOptions {
                    alpha: 1.0 / period as f64,
                    adjust: true,
                    bias: false,
                    min_periods: period,
                    ignore_nulls: true,
                }))
                .otherwise(lit(f64::NAN))
                .alias("adx"),
        )
        .collect()?;

    Ok(AdxResult {
        adx: result.column("adx")?.clone(),
        plus_di: result.column("plus_di")?.clone(),
        minus_di: result.column("minus_di")?.clone(),
    })
}

/// Calculate ADX with standard 14-period setting
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
///
/// # Returns
///
/// An `AdxResult` containing ADX, +DI, and -DI lines
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::adx_14;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let result = adx_14(&high, &low, &close).unwrap();
/// ```
pub fn adx_14(high: &Series, low: &Series, close: &Series) -> Result<AdxResult> {
    adx(high, low, close, 14)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adx_returns_correct_length() {
        let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0, 109.0, 108.5, 110.0, 111.0, 110.5, 112.0, 113.0, 112.5, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0]);
        let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0, 108.0, 107.5, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0]);
        let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0, 107.0, 106.5, 108.0, 109.0, 108.5, 110.0, 111.0, 110.5, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0]);
        
        let result = adx_14(&high, &low, &close).unwrap();
        
        assert_eq!(result.adx.len(), close.len());
        assert_eq!(result.plus_di.len(), close.len());
        assert_eq!(result.minus_di.len(), close.len());
    }

    #[test]
    fn test_adx_values_in_valid_range() {
        let high = Series::new("high", (0..30).map(|i| 100.0 + i as f64).collect::<Vec<_>>());
        let low = Series::new("low", (0..30).map(|i| 95.0 + i as f64).collect::<Vec<_>>());
        let close = Series::new("close", (0..30).map(|i| 98.0 + i as f64).collect::<Vec<_>>());
        
        let result = adx_14(&high, &low, &close).unwrap();
        
        for i in 0..result.adx.len() {
            if let Some(val) = result.adx.f64().unwrap().get(i) {
                if !val.is_nan() {
                    assert!((0.0..=100.0).contains(&val), "ADX value {} at index {} out of range", val, i);
                }
            }
        }
    }
}
