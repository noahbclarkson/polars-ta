//! Technical Analysis Indicators
//!
//! Core technical analysis indicators extracted from krypto.
//! Implements EMA, MACD, RSI, and ATR using Polars lazy frame operations.

use anyhow::Result;
use polars::prelude::*;

/// Calculate EMA with period 12
///
/// # Arguments
///
/// * `series` - Input price series
///
/// # Returns
///
/// A new Series containing EMA(12) values
pub fn ema_12(series: &Series) -> Result<Series> {
    let col_name = series.name().unwrap_or("close");
    
    let ewm_opts = EWMOptions {
        alpha: 1.0 / 12.0,
        adjust: true,
        bias: false,
        min_periods: 12,
        ignore_nulls: true,
    };
    
    let result = series
        .clone()
        .lazy()
        .with_column(col(col_name).ewm_mean(ewm_opts).alias("ema_12"))
        .collect()?;
    
    Ok(result.column("ema_12")?.clone())
}

/// Calculate EMA with period 20
///
/// # Arguments
///
/// * `series` - Input price series
///
/// # Returns
///
/// A new Series containing EMA(20) values
pub fn ema_20(series: &Series) -> Result<Series> {
    let col_name = series.name().unwrap_or("close");
    
    let ewm_opts = EWMOptions {
        alpha: 1.0 / 20.0,
        adjust: true,
        bias: false,
        min_periods: 20,
        ignore_nulls: true,
    };
    
    let result = series
        .clone()
        .lazy()
        .with_column(col(col_name).ewm_mean(ewm_opts).alias("ema_20"))
        .collect()?;
    
    Ok(result.column("ema_20")?.clone())
}

/// Calculate EMA with period 50
///
/// # Arguments
///
/// * `series` - Input price series
///
/// # Returns
///
/// A new Series containing EMA(50) values
pub fn ema_50(series: &Series) -> Result<Series> {
    let col_name = series.name().unwrap_or("close");
    
    let ewm_opts = EWMOptions {
        alpha: 1.0 / 50.0,
        adjust: true,
        bias: false,
        min_periods: 50,
        ignore_nulls: true,
    };
    
    let result = series
        .clone()
        .lazy()
        .with_column(col(col_name).ewm_mean(ewm_opts).alias("ema_50"))
        .collect()?;
    
    Ok(result.column("ema_50")?.clone())
}

/// Calculate EMA with period 200
///
/// # Arguments
///
/// * `series` - Input price series
///
/// # Returns
///
/// A new Series containing EMA(200) values
pub fn ema_200(series: &Series) -> Result<Series> {
    let col_name = series.name().unwrap_or("close");
    
    let ewm_opts = EWMOptions {
        alpha: 1.0 / 200.0,
        adjust: true,
        bias: false,
        min_periods: 200,
        ignore_nulls: true,
    };
    
    let result = series
        .clone()
        .lazy()
        .with_column(col(col_name).ewm_mean(ewm_opts).alias("ema_200"))
        .collect()?;
    
    Ok(result.column("ema_200")?.clone())
}

/// MACD calculation result
#[derive(Debug)]
pub struct MacdResult {
    /// MACD line (EMA(12) - EMA(26))
    pub macd: Series,
    /// Signal line (EMA(9) of MACD)
    pub signal: Series,
    /// Histogram (MACD - signal)
    pub histogram: Series,
}

/// Calculate MACD with standard parameters (12, 26, 9)
///
/// # Arguments
///
/// * `series` - Input price series (typically close prices)
///
/// # Returns
///
/// A `MacdResult` containing the MACD line, signal line, and histogram
pub fn macd(series: &Series) -> Result<MacdResult> {
    let col_name = series.name().unwrap_or("close");
    
    let ewm_fast = EWMOptions {
        alpha: 1.0 / 12.0,
        adjust: true,
        bias: false,
        min_periods: 12,
        ignore_nulls: true,
    };
    
    let ewm_slow = EWMOptions {
        alpha: 1.0 / 26.0,
        adjust: true,
        bias: false,
        min_periods: 26,
        ignore_nulls: true,
    };
    
    let ewm_signal = EWMOptions {
        alpha: 1.0 / 9.0,
        adjust: true,
        bias: false,
        min_periods: 9,
        ignore_nulls: true,
    };
    
    let df = series
        .clone()
        .lazy()
        .with_column(col(col_name).ewm_mean(ewm_fast).alias("ema_12"))
        .with_column(col(col_name).ewm_mean(ewm_slow).alias("ema_26"))
        .with_column((col("ema_12") - col("ema_26")).alias("macd"))
        .with_column(col("macd").ewm_mean(ewm_signal).alias("macd_signal"))
        .with_column((col("macd") - col("macd_signal")).alias("macd_hist"))
        .collect()?;
    
    Ok(MacdResult {
        macd: df.column("macd")?.clone(),
        signal: df.column("macd_signal")?.clone(),
        histogram: df.column("macd_hist")?.clone(),
    })
}

/// Calculate RSI with period 14 using Wilder's smoothing
///
/// # Arguments
///
/// * `series` - Input price series (typically close prices)
///
/// # Returns
///
/// A new Series containing RSI values (0-100 scale)
pub fn rsi_14(series: &Series) -> Result<Series> {
    let col_name = series.name().unwrap_or("close");
    
    let ewm_14 = EWMOptions {
        alpha: 1.0 / 14.0,
        adjust: true,
        bias: false,
        min_periods: 14,
        ignore_nulls: true,
    };
    
    let df = series
        .clone()
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
            col("gain").ewm_mean(ewm_14.clone()).alias("avg_gain"),
            col("loss").ewm_mean(ewm_14).alias("avg_loss"),
        ])
        .with_column(
            (lit(100.0) - (lit(100.0) / (lit(1.0) + (col("avg_gain") / col("avg_loss")))))
                .alias("rsi"),
        )
        .collect()?;
    
    Ok(df.column("rsi")?.clone())
}

/// Calculate True Range
///
/// True Range = max(high - low, |high - prev_close|, |low - prev_close|)
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
///
/// # Returns
///
/// A new Series containing True Range values
pub fn true_range(high: &Series, low: &Series, close: &Series) -> Result<Series> {
    let df = DataFrame::new(vec![
        high.clone(),
        low.clone(),
        close.clone(),
    ])?;
    
    let result = df
        .lazy()
        .with_columns(vec![
            (col("high") - col("low")).alias("tr1"),
            (col("high") - col("close").shift(lit(1)))
                .abs()
                .alias("tr2"),
            (col("low") - col("close").shift(lit(1)))
                .abs()
                .alias("tr3"),
        ])
        .with_column(
            when(
                col("tr1")
                    .gt_eq(col("tr2"))
                    .and(col("tr1").gt_eq(col("tr3"))),
            )
            .then(col("tr1"))
            .when(col("tr2").gt_eq(col("tr3")))
            .then(col("tr2"))
            .otherwise(col("tr3"))
            .alias("tr"),
        )
        .collect()?;
    
    Ok(result.column("tr")?.clone())
}

/// Calculate ATR with period 14
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
///
/// # Returns
///
/// A new Series containing ATR(14) values
pub fn atr_14(high: &Series, low: &Series, close: &Series) -> Result<Series> {
    let ewm_14 = EWMOptions {
        alpha: 1.0 / 14.0,
        adjust: true,
        bias: false,
        min_periods: 14,
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
            (col("high") - col("close").shift(lit(1)))
                .abs()
                .alias("tr2"),
            (col("low") - col("close").shift(lit(1)))
                .abs()
                .alias("tr3"),
        ])
        .with_column(
            when(
                col("tr1")
                    .gt_eq(col("tr2"))
                    .and(col("tr1").gt_eq(col("tr3"))),
            )
            .then(col("tr1"))
            .when(col("tr2").gt_eq(col("tr3")))
            .then(col("tr2"))
            .otherwise(col("tr3"))
            .alias("tr"),
        )
        .with_column(col("tr").ewm_mean(ewm_14).alias("atr"))
        .collect()?;
    
    Ok(result.column("atr")?.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ema_20_simple_series() {
        let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
        let result = ema_20(&close);
        
        assert!(result.is_ok());
        let ema = result.unwrap();
        assert_eq!(ema.len(), close.len());
    }
    
    #[test]
    fn test_ema_12_simple_series() {
        let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
        let result = ema_12(&close);
        
        assert!(result.is_ok());
        let ema = result.unwrap();
        assert_eq!(ema.len(), close.len());
    }
    
    #[test]
    fn test_ema_50_simple_series() {
        let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
        let result = ema_50(&close);
        
        assert!(result.is_ok());
        let ema = result.unwrap();
        assert_eq!(ema.len(), close.len());
    }
    
    #[test]
    fn test_ema_200_simple_series() {
        let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
        let result = ema_200(&close);
        
        assert!(result.is_ok());
        let ema = result.unwrap();
        assert_eq!(ema.len(), close.len());
    }
    
    #[test]
    fn test_rsi_14_basic() {
        // Test with a simple price series
        let close = Series::new("close", &[
            44.0, 44.25, 44.50, 44.75, 45.0, 
            45.25, 45.50, 45.75, 46.0, 46.25,
            46.50, 46.75, 47.0, 47.25
        ]);
        
        let result = rsi_14(&close);
        
        assert!(result.is_ok());
        let rsi = result.unwrap();
        assert_eq!(rsi.len(), close.len());
        
        // RSI values should be between 0 and 100
        for i in 0..rsi.len() {
            if let Some(val) = rsi.f64().unwrap().get(i) {
                if !val.is_nan() {
                    assert!(val >= 0.0 && val <= 100.0, "RSI value {} at index {} is out of range [0, 100]", val, i);
                }
            }
        }
    }
    
    #[test]
    fn test_macd_basic() {
        let close = Series::new("close", &[
            100.0, 101.5, 102.3, 101.8, 103.2,
            104.1, 103.5, 105.0, 106.2, 105.8,
            107.0, 106.5, 108.0, 109.2, 108.5,
            110.0, 109.8, 111.0, 110.5, 112.0
        ]);
        
        let result = macd(&close);
        
        assert!(result.is_ok());
        let macd_result = result.unwrap();
        assert_eq!(macd_result.macd.len(), close.len());
        assert_eq!(macd_result.signal.len(), close.len());
        assert_eq!(macd_result.histogram.len(), close.len());
    }
    
    #[test]
    fn test_atr_14_basic() {
        let high = Series::new("high", &[
            102.0, 103.5, 104.0, 103.5, 105.0,
            106.0, 105.5, 107.0, 108.0, 107.5,
            109.0, 108.5, 110.0, 111.0, 110.5,
            112.0, 111.5, 113.0, 112.5, 114.0
        ]);
        
        let low = Series::new("low", &[
            99.0, 100.5, 101.0, 100.5, 102.0,
            103.0, 102.5, 104.0, 105.0, 104.5,
            106.0, 105.5, 107.0, 108.0, 107.5,
            109.0, 108.5, 110.0, 109.5, 111.0
        ]);
        
        let close = Series::new("close", &[
            100.0, 101.5, 102.3, 101.8, 103.2,
            104.1, 103.5, 105.0, 106.2, 105.8,
            107.0, 106.5, 108.0, 109.2, 108.5,
            110.0, 109.8, 111.0, 110.5, 112.0
        ]);
        
        let result = atr_14(&high, &low, &close);
        
        assert!(result.is_ok());
        let atr = result.unwrap();
        assert_eq!(atr.len(), close.len());
        
        // ATR should be non-negative
        for i in 0..atr.len() {
            if let Some(val) = atr.f64().unwrap().get(i) {
                if !val.is_nan() {
                    assert!(val >= 0.0, "ATR value {} at index {} is negative", val, i);
                }
            }
        }
    }
}
