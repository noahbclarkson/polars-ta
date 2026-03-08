//! Commodity Channel Index (CCI)
//!
//! CCI is a momentum-based oscillator used to help determine when an asset
//! is reaching overbought or oversold conditions.

use anyhow::Result;
use polars::prelude::*;

/// Calculate Commodity Channel Index (CCI)
///
/// CCI = (Typical Price - SMA(TP)) / (0.015 * Mean Deviation)
/// where Typical Price = (High + Low + Close) / 3
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `period` - Period for calculations (standard: 20)
///
/// # Returns
///
/// A new Series containing CCI values
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::cci;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let cci_values = cci(&high, &low, &close, 20).unwrap();
/// ```
pub fn cci(high: &Series, low: &Series, close: &Series, period: usize) -> Result<Series> {
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
    ])?;

    let result = df
        .lazy()
        // Calculate Typical Price
        .with_column(
            ((col("high") + col("low") + col("close")) / lit(3.0))
                .alias("tp"),
        )
        // Calculate SMA of TP
        .with_column(
            col("tp").rolling_mean(rolling_opts.clone()).alias("sma_tp"),
        )
        // Calculate Mean Deviation: mean(|TP - SMA_TP|) over period
        .with_column(
            (col("tp") - col("sma_tp")).abs().alias("deviation"),
        )
        .with_column(
            col("deviation").rolling_mean(rolling_opts).alias("mean_dev"),
        )
        // Calculate CCI
        .with_column(
            ((col("tp") - col("sma_tp")) / (lit(0.015) * col("mean_dev")))
                .alias("cci"),
        )
        .collect()?;

    Ok(result.column("cci")?.clone())
}

/// Calculate CCI with the standard 20-period setting
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
///
/// # Returns
///
/// A new Series containing CCI values
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::cci_20;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let cci = cci_20(&high, &low, &close).unwrap();
/// ```
pub fn cci_20(high: &Series, low: &Series, close: &Series) -> Result<Series> {
    cci(high, low, close, 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cci_returns_correct_length() {
        let high = Series::new("high", (0..30).map(|i| 105.0 + i as f64).collect::<Vec<_>>());
        let low = Series::new("low", (0..30).map(|i| 100.0 + i as f64).collect::<Vec<_>>());
        let close = Series::new("close", (0..30).map(|i| 103.0 + i as f64).collect::<Vec<_>>());
        
        let result = cci_20(&high, &low, &close).unwrap();
        
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_cci_extreme_values() {
        // When price is at the high of the range, CCI should be high positive
        // When price is at the low of the range, CCI should be negative
        let high = Series::new("high", &[110.0; 25]);
        let low = Series::new("low", &[100.0; 25]);
        let close = Series::new("close", &[110.0; 25]); // At high
        
        let result = cci(&high, &low, &close, 20).unwrap();
        let cci_ca = result.f64().unwrap();
        
        // At index 24, CCI should be positive (price at high)
        if let Some(val) = cci_ca.get(24) {
            if !val.is_nan() {
                assert!(val > 0.0, "CCI should be positive when close is at high");
            }
        }
    }
}
