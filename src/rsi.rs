//! Relative Strength Index (RSI)
//!
//! RSI is a momentum oscillator that measures the speed and magnitude
//! of recent price changes to evaluate overbought or oversold conditions.

use anyhow::Result;
use polars::prelude::*;

/// Calculate Relative Strength Index (RSI) using Wilder's smoothing
///
/// RSI is calculated as: RSI = 100 - 100 / (1 + RS)
/// where RS = Average Gain / Average Loss
///
/// # Arguments
///
/// * `series` - Input price series
/// * `period` - RSI period (standard: 14)
///
/// # Returns
///
/// A new Series containing RSI values (0-100 scale)
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::rsi;
///
/// let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let rsi_14 = rsi(&close, 14).unwrap();
/// ```
pub fn rsi(series: &Series, period: usize) -> Result<Series> {
    let col_name = series.name();
    
    let ewm_opts = EWMOptions {
        alpha: 1.0 / period as f64,
        adjust: true,
        bias: false,
        min_periods: period,
        ignore_nulls: true,
    };
    
    let df = DataFrame::new(vec![series.clone()])?;
    
    let result = df
        .lazy()
        .with_column(col(col_name).diff(1, Default::default()).alias("diff"))
        .with_columns(vec![
            when(col("diff").gt(0.0))
                .then(col("diff"))
                .otherwise(lit(0.0))
                .alias("gain"),
            when(col("diff").lt(0.0))
                .then(col("diff").abs())
                .otherwise(lit(0.0))
                .alias("loss"),
        ])
        .with_columns(vec![
            col("gain").ewm_mean(ewm_opts).alias("avg_gain"),
            col("loss").ewm_mean(ewm_opts).alias("avg_loss"),
        ])
        .with_column(
            (lit(100.0) - (lit(100.0) / (lit(1.0) + (col("avg_gain") / col("avg_loss")))))
                .alias("rsi"),
        )
        .collect()?;
    
    Ok(result.column("rsi")?.clone())
}

/// Calculate RSI with the standard 14-period setting
///
/// # Arguments
///
/// * `series` - Input price series
///
/// # Returns
///
/// A new Series containing RSI values (0-100 scale)
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::rsi_14;
///
/// let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let rsi = rsi_14(&close).unwrap();
/// ```
pub fn rsi_14(series: &Series) -> Result<Series> {
    rsi(series, 14)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi_length() {
        let prices: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let series = Series::new("close", prices);
        let result = rsi(&series, 14).unwrap();
        assert_eq!(result.len(), series.len());
    }

    #[test]
    fn test_rsi_range() {
        let prices: Vec<f64> = (1..=50).map(|x| (x as f64 * 1.7).sin() * 10.0 + 100.0).collect();
        let series = Series::new("close", prices);
        let result = rsi(&series, 14).unwrap();
        let vals = result.f64().unwrap();

        for i in 0..vals.len() {
            if let Some(v) = vals.get(i) {
                if !v.is_nan() {
                    assert!(v >= 0.0 && v <= 100.0, "RSI must be in [0, 100], got {}", v);
                }
            }
        }
    }

    #[test]
    fn test_rsi_uptrend_high() {
        // Strict uptrend should produce high RSI
        let prices: Vec<f64> = (1..=30).map(|x| x as f64 * 2.0).collect();
        let series = Series::new("close", prices);
        let result = rsi(&series, 14).unwrap();
        let vals = result.f64().unwrap();

        let last = vals.get(vals.len() - 1).unwrap();
        assert!(last > 70.0, "RSI should be > 70 in strong uptrend, got {}", last);
    }

    #[test]
    fn test_rsi_14_matches_rsi_with_14() {
        let prices: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let series = Series::new("close", prices);

        let result_14 = rsi_14(&series).unwrap();
        let result_explicit = rsi(&series, 14).unwrap();

        let v1 = result_14.f64().unwrap();
        let v2 = result_explicit.f64().unwrap();

        let last1 = v1.get(v1.len() - 1).unwrap();
        let last2 = v2.get(v2.len() - 1).unwrap();
        assert!((last1 - last2).abs() < 0.001, "rsi_14 should match rsi(14)");
    }
}
