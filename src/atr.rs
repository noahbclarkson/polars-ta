//! Average True Range (ATR)
//!
//! ATR is a volatility indicator that measures market volatility by
//! decomposing the entire range of an asset price for a period.

use anyhow::Result;
use polars::prelude::*;

/// Calculate Average True Range (ATR)
///
/// True Range = max(high - low, |high - prev_close|, |low - prev_close|)
/// ATR = EMA of True Range
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `period` - ATR period (standard: 14)
///
/// # Returns
///
/// A new Series containing ATR values
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::atr;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let atr_14 = atr(&high, &low, &close, 14).unwrap();
/// ```
pub fn atr(high: &Series, low: &Series, close: &Series, period: usize) -> Result<Series> {
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
        .with_column(col("tr").ewm_mean(ewm_opts).alias("atr"))
        .collect()?;
    
    Ok(result.column("atr")?.clone())
}

/// Calculate ATR with the standard 14-period setting
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
///
/// # Returns
///
/// A new Series containing ATR values
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::atr_14;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let atr = atr_14(&high, &low, &close).unwrap();
/// ```
pub fn atr_14(high: &Series, low: &Series, close: &Series) -> Result<Series> {
    atr(high, low, close, 14)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atr_length() {
        let high = Series::new("high", vec![105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0]);
        let low = Series::new("low", vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0]);
        let close = Series::new("close", vec![103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0]);
        let result = atr(&high, &low, &close, 14).unwrap();
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_atr_positive_values() {
        let high = Series::new("high", (1..=20).map(|x| x as f64 + 2.0).collect::<Vec<_>>());
        let low = Series::new("low", (1..=20).map(|x| x as f64 - 2.0).collect::<Vec<_>>());
        let close = Series::new("close", (1..=20).map(|x| x as f64).collect::<Vec<_>>());
        let result = atr(&high, &low, &close, 14).unwrap();
        let vals = result.f64().unwrap();

        for i in 14..vals.len() {
            let v = vals.get(i).unwrap();
            assert!(v > 0.0, "ATR should be positive, got {} at index {}", v, i);
        }
    }

    #[test]
    fn test_atr_14_matches_atr_period_14() {
        let high = Series::new("high", (1..=20).map(|x| x as f64 + 3.0).collect::<Vec<_>>());
        let low = Series::new("low", (1..=20).map(|x| x as f64 - 1.0).collect::<Vec<_>>());
        let close = Series::new("close", (1..=20).map(|x| x as f64).collect::<Vec<_>>());

        let r1 = atr_14(&high, &low, &close).unwrap();
        let r2 = atr(&high, &low, &close, 14).unwrap();

        let v1 = r1.f64().unwrap().get(18).unwrap();
        let v2 = r2.f64().unwrap().get(18).unwrap();
        assert!((v1 - v2).abs() < 0.001, "atr_14 should match atr(14)");
    }
}
