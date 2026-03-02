//! Candlestick Pattern Detection
//!
//! This module provides functions to detect common candlestick patterns
//! using Polars expressions for efficient vectorized operations.

use anyhow::Result;
use polars::prelude::*;

/// Detect Doji pattern (open ≈ close)
///
/// A Doji occurs when the open and close prices are approximately equal,
/// indicating market indecision.
///
/// # Arguments
///
/// * `open` - Open price series
/// * `close` - Close price series
/// * `threshold` - Maximum body size as a fraction of price (default: 0.001)
///
/// # Returns
///
/// A Boolean Series where `true` indicates a Doji pattern
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::patterns::doji;
///
/// let open = Series::new("open", &[100.0, 101.0, 102.0]);
/// let close = Series::new("close", &[100.1, 100.9, 102.05]);
/// let doji_pattern = doji(&open, &close, Some(0.001)).unwrap();
/// ```
pub fn doji(open: &Series, close: &Series, threshold: Option<f64>) -> Result<Series> {
    let threshold = threshold.unwrap_or(0.001);
    
    let df = DataFrame::new(vec![open.clone(), close.clone()])?;
    
    let result = df
        .lazy()
        .with_column((col("open") - col("close")).abs().alias("body"))
        .with_column(
            ((col("open") + col("close")) / lit(2.0)).alias("mid_price")
        )
        .with_column(
            (col("body") / col("mid_price")).alias("body_ratio")
        )
        .with_column(
            col("body_ratio").lt(lit(threshold)).alias("is_doji")
        )
        .collect()?;
    
    Ok(result.column("is_doji")?.clone())
}

/// Detect Hammer pattern
///
/// A Hammer has a small body at the top of the trading range with a long lower shadow.
/// It's considered a bullish reversal signal when appearing in a downtrend.
///
/// # Arguments
///
/// * `open` - Open price series
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `body_threshold` - Maximum body size as fraction of total range (default: 0.3)
/// * `shadow_ratio` - Minimum lower shadow to upper shadow ratio (default: 2.0)
///
/// # Returns
///
/// A Boolean Series where `true` indicates a Hammer pattern
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::patterns::hammer;
///
/// let open = Series::new("open", &[100.0, 101.0, 102.0]);
/// let high = Series::new("high", &[101.0, 102.0, 103.0]);
/// let low = Series::new("low", &[98.0, 99.0, 100.0]);
/// let close = Series::new("close", &[100.5, 101.5, 102.3]);
/// let hammer_pattern = hammer(&open, &high, &low, &close, None, None).unwrap();
/// ```
pub fn hammer(
    open: &Series,
    high: &Series,
    low: &Series,
    close: &Series,
    body_threshold: Option<f64>,
    shadow_ratio: Option<f64>,
) -> Result<Series> {
    let body_threshold = body_threshold.unwrap_or(0.3);
    let shadow_ratio = shadow_ratio.unwrap_or(2.0);
    
    let df = DataFrame::new(vec![
        open.clone(),
        high.clone(),
        low.clone(),
        close.clone(),
    ])?;
    
    let result = df
        .lazy()
        // Calculate body and shadows
        .with_column(
            when(col("open").lt(col("close")))
                .then(col("open"))
                .otherwise(col("close"))
                .alias("body_bottom")
        )
        .with_column(
            when(col("open").gt(col("close")))
                .then(col("open"))
                .otherwise(col("close"))
                .alias("body_top")
        )
        .with_column((col("open") - col("close")).abs().alias("body"))
        .with_column((col("high") - col("low")).alias("range"))
        
        // Upper shadow (high - body_top)
        .with_column((col("high") - col("body_top")).alias("upper_shadow"))
        
        // Lower shadow (body_bottom - low)
        .with_column((col("body_bottom") - col("low")).alias("lower_shadow"))
        
        // Hammer conditions:
        // 1. Body is small relative to total range
        // 2. Lower shadow is at least shadow_ratio times the upper shadow
        // 3. Lower shadow exists (avoid division by zero)
        .with_column(
            (col("body").lt(col("range") * lit(body_threshold)))
                .and(col("lower_shadow").gt(lit(0.0)))
                .and(
                    when(col("upper_shadow").gt(lit(0.0)))
                        .then(col("lower_shadow").gt(col("upper_shadow") * lit(shadow_ratio)))
                        .otherwise(lit(true))
                )
                .alias("is_hammer")
        )
        .collect()?;
    
    Ok(result.column("is_hammer")?.clone())
}

/// Detect Engulfing pattern
///
/// A bullish engulfing occurs when a green candle completely engulfs the previous red candle.
/// A bearish engulfing occurs when a red candle completely engulfs the previous green candle.
///
/// # Arguments
///
/// * `open` - Open price series
/// * `close` - Close price series
///
/// # Returns
///
/// An i32 Series where:
/// - `1` indicates bullish engulfing
/// - `-1` indicates bearish engulfing
/// - `0` indicates no engulfing pattern
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::patterns::engulfing;
///
/// let open = Series::new("open", &[100.0, 99.0, 101.0, 102.0]);
/// let close = Series::new("close", &[99.0, 100.0, 102.0, 101.0]);
/// let engulfing_pattern = engulfing(&open, &close).unwrap();
/// ```
pub fn engulfing(open: &Series, close: &Series) -> Result<Series> {
    let df = DataFrame::new(vec![open.clone(), close.clone()])?;
    
    let result = df
        .lazy()
        // Current candle properties
        .with_column((col("open").lt(col("close"))).alias("current_bullish"))
        .with_column(
            when(col("open").lt(col("close")))
                .then(col("open"))
                .otherwise(col("close"))
                .alias("current_body_low")
        )
        .with_column(
            when(col("open").gt(col("close")))
                .then(col("open"))
                .otherwise(col("close"))
                .alias("current_body_high")
        )
        
        // Previous candle properties (shift by 1)
        .with_column(col("open").shift(lit(1)).alias("prev_open"))
        .with_column(col("close").shift(lit(1)).alias("prev_close"))
        .with_column((col("prev_open").lt(col("prev_close"))).alias("prev_bullish"))
        .with_column(
            when(col("prev_open").lt(col("prev_close")))
                .then(col("prev_open"))
                .otherwise(col("prev_close"))
                .alias("prev_body_low")
        )
        .with_column(
            when(col("prev_open").gt(col("prev_close")))
                .then(col("prev_open"))
                .otherwise(col("prev_close"))
                .alias("prev_body_high")
        )
        
        // Bullish engulfing: current is bullish, previous is bearish, current engulfs previous
        .with_column(
            col("current_bullish")
                .and(col("prev_bullish").not())
                .and(col("current_body_low").lt(col("prev_body_low")))
                .and(col("current_body_high").gt(col("prev_body_high")))
                .alias("bullish_engulfing")
        )
        
        // Bearish engulfing: current is bearish, previous is bullish, current engulfs previous
        .with_column(
            col("current_bullish").not()
                .and(col("prev_bullish"))
                .and(col("current_body_low").lt(col("prev_body_low")))
                .and(col("current_body_high").gt(col("prev_body_high")))
                .alias("bearish_engulfing")
        )
        
        // Combine into single result
        .with_column(
            when(col("bullish_engulfing"))
                .then(lit(1))
                .when(col("bearish_engulfing"))
                .then(lit(-1))
                .otherwise(lit(0))
                .alias("engulfing")
        )
        .collect()?;
    
    Ok(result.column("engulfing")?.clone())
}

/// Detect Morning Star and Evening Star patterns
///
/// Morning Star (bullish reversal): 
/// 1. Long bearish candle
/// 2. Small body candle (gap down)
/// 3. Long bullish candle that closes above midpoint of first candle
///
/// Evening Star (bearish reversal):
/// 1. Long bullish candle
/// 2. Small body candle (gap up)
/// 3. Long bearish candle that closes below midpoint of first candle
///
/// # Arguments
///
/// * `open` - Open price series
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `body_threshold` - Maximum body size for middle candle as fraction of first candle body (default: 0.3)
///
/// # Returns
///
/// An i32 Series where:
/// - `1` indicates morning star
/// - `-1` indicates evening star
/// - `0` indicates no star pattern
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::patterns::star;
///
/// let open = Series::new("open", &[100.0, 99.0, 98.5, 99.0, 101.0]);
/// let high = Series::new("high", &[101.0, 99.5, 99.0, 101.0, 102.0]);
/// let low = Series::new("low", &[99.0, 98.0, 98.0, 98.5, 100.0]);
/// let close = Series::new("close", &[99.5, 98.5, 98.8, 100.5, 101.5]);
/// let star_pattern = star(&open, &high, &low, &close, None).unwrap();
/// ```
pub fn star(
    open: &Series,
    high: &Series,
    low: &Series,
    close: &Series,
    body_threshold: Option<f64>,
) -> Result<Series> {
    let body_threshold = body_threshold.unwrap_or(0.3);
    
    let df = DataFrame::new(vec![
        open.clone(),
        high.clone(),
        low.clone(),
        close.clone(),
    ])?;
    
    let result = df
        .lazy()
        // Candle properties
        .with_column((col("open") - col("close")).abs().alias("body"))
        .with_column((col("open").lt(col("close"))).alias("bullish"))
        .with_column(
            when(col("open").lt(col("close")))
                .then(col("open"))
                .otherwise(col("close"))
                .alias("body_low")
        )
        .with_column(
            when(col("open").gt(col("close")))
                .then(col("open"))
                .otherwise(col("close"))
                .alias("body_high")
        )
        
        // First candle (shift by 2)
        .with_column(col("body").shift(lit(2)).alias("candle1_body"))
        .with_column(col("bullish").shift(lit(2)).alias("candle1_bullish"))
        .with_column(col("body_low").shift(lit(2)).alias("candle1_body_low"))
        .with_column(col("body_high").shift(lit(2)).alias("candle1_body_high"))
        .with_column(col("close").shift(lit(2)).alias("candle1_close"))
        
        // Middle candle (shift by 1)
        .with_column(col("body").shift(lit(1)).alias("candle2_body"))
        .with_column(col("bullish").shift(lit(1)).alias("candle2_bullish"))
        
        // Current candle (third candle)
        .with_column(col("bullish").alias("candle3_bullish"))
        .with_column(col("close").alias("candle3_close"))
        
        // First candle midpoint
        .with_column(
            (col("candle1_body_low") + col("candle1_body_high")) / lit(2.0)
                .alias("candle1_midpoint")
        )
        
        // Morning Star conditions:
        // 1. First candle is bearish (red)
        // 2. Middle candle has small body
        // 3. Third candle is bullish and closes above first candle midpoint
        .with_column(
            col("candle1_bullish").not()
                .and(col("candle2_body").lt(col("candle1_body") * lit(body_threshold)))
                .and(col("candle3_bullish"))
                .and(col("candle3_close").gt(col("candle1_midpoint")))
                .alias("morning_star")
        )
        
        // Evening Star conditions:
        // 1. First candle is bullish (green)
        // 2. Middle candle has small body
        // 3. Third candle is bearish and closes below first candle midpoint
        .with_column(
            col("candle1_bullish")
                .and(col("candle2_body").lt(col("candle1_body") * lit(body_threshold)))
                .and(col("candle3_bullish").not())
                .and(col("candle3_close").lt(col("candle1_midpoint")))
                .alias("evening_star")
        )
        
        // Combine into single result
        .with_column(
            when(col("morning_star"))
                .then(lit(1))
                .when(col("evening_star"))
                .then(lit(-1))
                .otherwise(lit(0))
                .alias("star_pattern")
        )
        .collect()?;
    
    Ok(result.column("star_pattern")?.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_doji_detection() -> Result<()> {
        let open = Series::new("open", &[100.0, 101.0, 102.0, 103.0, 104.0]);
        let close = Series::new("close", &[100.0, 100.999, 102.001, 103.5, 104.0]);
        
        let result = doji(&open, &close, Some(0.001))?;
        let values = result.bool()?;
        
        // First and last should be doji (open == close)
        assert!(values.get(0).unwrap_or(false));
        assert!(values.get(4).unwrap_or(false));
        
        // Second and third are borderline (within threshold)
        assert!(values.get(1).unwrap_or(false));
        assert!(values.get(2).unwrap_or(false));
        
        // Fourth is not doji (larger body)
        assert!(!values.get(3).unwrap_or(true));
        
        Ok(())
    }
    
    #[test]
    fn test_hammer_detection() -> Result<()> {
        // Create a hammer pattern: small body at top, long lower shadow
        let open = Series::new("open", &[100.0, 101.0, 102.0]);
        let high = Series::new("high", &[100.5, 101.5, 102.2]);
        let low = Series::new("low", &[95.0, 96.0, 97.0]);
        let close = Series::new("close", &[100.3, 101.3, 102.1]);
        
        let result = hammer(&open, &high, &low, &close, None, None)?;
        let values = result.bool()?;
        
        // All should be hammers (small body, long lower shadow, minimal upper shadow)
        for i in 0..3 {
            assert!(values.get(i).unwrap_or(false), "Candle {} should be a hammer", i);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_bullish_engulfing() -> Result<()> {
        // Create bullish engulfing pattern
        let open = Series::new("open", &[100.0, 99.0, 98.0]);
        let close = Series::new("close", &[99.0, 98.0, 100.5]);
        
        let result = engulfing(&open, &close)?;
        let values = result.i32()?;
        
        // Third candle should be bullish engulfing (1)
        assert_eq!(values.get(2), Some(1));
        
        // First two should be no pattern (0)
        assert_eq!(values.get(0), Some(0));
        assert_eq!(values.get(1), Some(0));
        
        Ok(())
    }
    
    #[test]
    fn test_bearish_engulfing() -> Result<()> {
        // Create bearish engulfing pattern
        let open = Series::new("open", &[100.0, 101.0, 102.0]);
        let close = Series::new("close", &[101.0, 102.0, 99.0]);
        
        let result = engulfing(&open, &close)?;
        let values = result.i32()?;
        
        // Third candle should be bearish engulfing (-1)
        assert_eq!(values.get(2), Some(-1));
        
        Ok(())
    }
    
    #[test]
    fn test_morning_star() -> Result<()> {
        // Create morning star pattern
        // Candle 1: Bearish, large body
        // Candle 2: Small body
        // Candle 3: Bullish, closes above midpoint of candle 1
        let open = Series::new("open", &[105.0, 100.0, 98.0, 97.0, 96.0]);
        let high = Series::new("high", &[105.5, 100.5, 98.5, 102.0, 96.5]);
        let low = Series::new("low", &[100.0, 97.0, 97.0, 97.0, 94.0]);
        let close = Series::new("close", &[100.0, 98.0, 98.2, 101.0, 95.0]);
        
        let result = star(&open, &high, &low, &close, None)?;
        let values = result.i32()?;
        
        // Candle 3 (index 2) should be morning star
        assert_eq!(values.get(2), Some(1));
        
        Ok(())
    }
    
    #[test]
    fn test_evening_star() -> Result<()> {
        // Create evening star pattern
        // Candle 1: Bullish, large body
        // Candle 2: Small body
        // Candle 3: Bearish, closes below midpoint of candle 1
        let open = Series::new("open", &[95.0, 96.0, 101.0, 100.5, 102.0]);
        let high = Series::new("high", &[96.5, 102.0, 101.5, 101.0, 103.0]);
        let low = Series::new("low", &[95.0, 96.0, 100.0, 97.0, 96.0]);
        let close = Series::new("close", &[96.5, 101.0, 100.5, 97.5, 96.5]);
        
        let result = star(&open, &high, &low, &close, None)?;
        let values = result.i32()?;
        
        // Candle 3 (index 2) should be evening star
        assert_eq!(values.get(2), Some(-1));
        
        Ok(())
    }
    
    #[test]
    fn test_no_pattern() -> Result<()> {
        let open = Series::new("open", &[100.0, 101.0, 102.0]);
        let close = Series::new("close", &[101.0, 102.0, 103.0]);
        
        let result = engulfing(&open, &close)?;
        let values = result.i32()?;
        
        // No engulfing pattern in trending market
        for i in 0..3 {
            assert_eq!(values.get(i), Some(0));
        }
        
        Ok(())
    }
}
