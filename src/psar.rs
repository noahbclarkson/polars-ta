//! Parabolic SAR (Stop and Reverse)
//!
//! PSAR is a trend-following indicator that sets trailing price stops for long
//! or short positions. It's designed to help identify the direction of an asset's
//! momentum and provide entry/exit points.

use anyhow::Result;
use polars::prelude::*;

/// Calculate Parabolic SAR
///
/// If trend is up:
///   SAR = prev_SAR + AF × (EP - prev_SAR)
///   EP = highest high during trend
/// If trend is down:
///   SAR = prev_SAR - AF × (prev_SAR - EP)
///   EP = lowest low during trend
///
/// AF (Acceleration Factor) starts at af_step, increases by af_step each time
/// EP updates, up to a maximum of af_max.
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `af_step` - Acceleration factor step (default: 0.02)
/// * `af_max` - Maximum acceleration factor (default: 0.20)
///
/// # Returns
///
/// A new Series containing PSAR values
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::psar;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let psar_values = psar(&high, &low, &close, 0.02, 0.20).unwrap();
/// ```
pub fn psar(
    high: &Series,
    low: &Series,
    close: &Series,
    af_step: f64,
    af_max: f64,
) -> Result<Series> {
    let high_ca = high.f64()?;
    let low_ca = low.f64()?;
    let close_ca = close.f64()?;
    
    let len = high.len();
    let mut sar_values = vec![f64::NAN; len];
    
    // Need at least 2 data points
    if len < 2 {
        return Ok(Series::new("psar", sar_values));
    }
    
    // Initialize: determine initial trend from first two closes
    let close_0 = close_ca.get(0).unwrap_or(f64::NAN);
    let close_1 = close_ca.get(1).unwrap_or(f64::NAN);
    let high_0 = high_ca.get(0).unwrap_or(f64::NAN);
    let high_1 = high_ca.get(1).unwrap_or(f64::NAN);
    let low_0 = low_ca.get(0).unwrap_or(f64::NAN);
    let low_1 = low_ca.get(1).unwrap_or(f64::NAN);
    
    if close_0.is_nan() || close_1.is_nan() || high_0.is_nan() || high_1.is_nan() || low_0.is_nan() || low_1.is_nan() {
        return Ok(Series::new("psar", sar_values));
    }
    
    // Initial trend: up if close increased, otherwise down
    let mut is_up_trend = close_1 > close_0;
    
    // Initialize SAR, EP, and AF
    let mut af = af_step;
    let mut ep = if is_up_trend { high_1.max(high_0) } else { low_1.min(low_0) };
    let mut sar = if is_up_trend { low_0 } else { high_0 };
    
    // First SAR value is typically the extreme of the previous trend
    sar_values[0] = f64::NAN;
    sar_values[1] = sar;
    
    // Iterate through the data
    #[allow(clippy::needless_range_loop)]
    for i in 2..len {
        let high_i = high_ca.get(i).unwrap_or(f64::NAN);
        let low_i = low_ca.get(i).unwrap_or(f64::NAN);
        let low_prev = low_ca.get(i - 1).unwrap_or(f64::NAN);
        let low_prev2 = low_ca.get(i - 2).unwrap_or(f64::NAN);
        let high_prev = high_ca.get(i - 1).unwrap_or(f64::NAN);
        let high_prev2 = high_ca.get(i - 2).unwrap_or(f64::NAN);
        
        if high_i.is_nan() || low_i.is_nan() || low_prev.is_nan() || low_prev2.is_nan() || high_prev.is_nan() || high_prev2.is_nan() {
            sar_values[i] = f64::NAN;
            continue;
        }
        
        if is_up_trend {
            // Calculate new SAR for uptrend
            let mut new_sar = sar + af * (ep - sar);
            
            // SAR should not exceed recent lows
            new_sar = new_sar.min(low_prev).min(low_prev2);
            
            // Check for trend reversal
            if low_i < new_sar {
                // Reversal: switch to downtrend
                is_up_trend = false;
                sar = ep; // New SAR is the highest high of the uptrend
                ep = low_i; // New EP is the current low
                af = af_step; // Reset AF
            } else {
                // Continue uptrend
                sar = new_sar;
                
                // Update EP and AF if new high
                if high_i > ep {
                    ep = high_i;
                    af = (af + af_step).min(af_max);
                }
            }
        } else {
            // Calculate new SAR for downtrend
            let mut new_sar = sar - af * (sar - ep);
            
            // SAR should not be below recent highs
            new_sar = new_sar.max(high_prev).max(high_prev2);
            
            // Check for trend reversal
            if high_i > new_sar {
                // Reversal: switch to uptrend
                is_up_trend = true;
                sar = ep; // New SAR is the lowest low of the downtrend
                ep = high_i; // New EP is the current high
                af = af_step; // Reset AF
            } else {
                // Continue downtrend
                sar = new_sar;
                
                // Update EP and AF if new low
                if low_i < ep {
                    ep = low_i;
                    af = (af + af_step).min(af_max);
                }
            }
        }
        
        sar_values[i] = sar;
    }
    
    Ok(Series::new("psar", sar_values))
}

/// Calculate PSAR with default settings (af_step=0.02, af_max=0.20)
///
/// # Arguments
///
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
///
/// # Returns
///
/// A new Series containing PSAR values
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::psar_default;
///
/// let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
/// let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
/// let psar = psar_default(&high, &low, &close).unwrap();
/// ```
pub fn psar_default(high: &Series, low: &Series, close: &Series) -> Result<Series> {
    psar(high, low, close, 0.02, 0.20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psar_returns_correct_length() {
        let high = Series::new("high".into(), (0..30).map(|i| 105.0 + i as f64).collect::<Vec<_>>());
        let low = Series::new("low".into(), (0..30).map(|i| 100.0 + i as f64).collect::<Vec<_>>());
        let close = Series::new("close".into(), (0..30).map(|i| 103.0 + i as f64).collect::<Vec<_>>());
        
        let result = psar_default(&high, &low, &close).unwrap();
        
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_psar_uptrend_below_price() {
        // In an uptrend, SAR should be below the price
        let high = Series::new("high".into(), (0..20).map(|i| 105.0 + i as f64).collect::<Vec<_>>());
        let low = Series::new("low".into(), (0..20).map(|i| 100.0 + i as f64).collect::<Vec<_>>());
        let close = Series::new("close".into(), (0..20).map(|i| 103.0 + i as f64).collect::<Vec<_>>());
        
        let result = psar_default(&high, &low, &close).unwrap();
        let psar_ca = result.f64().unwrap();
        
        // In a strong uptrend, PSAR should be below the lows
        for i in 5..20 {
            if let Some(sar) = psar_ca.get(i) {
                if !sar.is_nan() {
                    let low_i = low.f64().unwrap().get(i).unwrap_or(f64::NAN);
                    if !low_i.is_nan() {
                        assert!(sar <= low_i + 1.0, 
                            "PSAR {} should be at or below low {} in uptrend at index {}", 
                            sar, low_i, i);
                    }
                }
            }
        }
    }

    #[test]
    fn test_psar_downtrend_above_price() {
        // In a downtrend, SAR should be above the price
        let high = Series::new("high".into(), (0..20).map(|i| 105.0 - i as f64).collect::<Vec<_>>());
        let low = Series::new("low".into(), (0..20).map(|i| 100.0 - i as f64).collect::<Vec<_>>());
        let close = Series::new("close".into(), (0..20).map(|i| 103.0 - i as f64).collect::<Vec<_>>());
        
        let result = psar_default(&high, &low, &close).unwrap();
        let psar_ca = result.f64().unwrap();
        
        // In a strong downtrend, PSAR should be above the highs
        for i in 5..20 {
            if let Some(sar) = psar_ca.get(i) {
                if !sar.is_nan() {
                    let high_i = high.f64().unwrap().get(i).unwrap_or(f64::NAN);
                    if !high_i.is_nan() {
                        assert!(sar >= high_i - 1.0, 
                            "PSAR {} should be at or above high {} in downtrend at index {}", 
                            sar, high_i, i);
                    }
                }
            }
        }
    }

    #[test]
    fn test_psar_handles_short_series() {
        let high = Series::new("high".into(), &[105.0, 106.0]);
        let low = Series::new("low".into(), &[100.0, 101.0]);
        let close = Series::new("close".into(), &[103.0, 104.0]);
        
        let result = psar_default(&high, &low, &close).unwrap();
        
        assert_eq!(result.len(), 2);
    }
}
