//! On-Balance Volume (OBV)
//!
//! OBV is a momentum indicator that uses volume flow to predict changes in price.

use anyhow::Result;
use polars::prelude::*;

/// Calculate On-Balance Volume (OBV)
///
/// OBV is a running cumulative total:
/// - If close > prev_close: OBV = prev_OBV + volume
/// - If close < prev_close: OBV = prev_OBV - volume
/// - If close = prev_close: OBV = prev_OBV
///
/// # Arguments
///
/// * `close` - Close price series
/// * `volume` - Volume series
///
/// # Returns
///
/// A new Series containing OBV values
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::obv;
///
/// let close = Series::new("close", &[100.0, 101.0, 100.5, 102.0, 101.5]);
/// let volume = Series::new("volume", &[1000.0, 1200.0, 800.0, 1500.0, 900.0]);
/// let obv_values = obv(&close, &volume).unwrap();
/// ```
pub fn obv(close: &Series, volume: &Series) -> Result<Series> {
    let close_ca = close.f64()?;
    let volume_ca = volume.f64()?;
    
    let len = close.len();
    let mut obv_values = Vec::with_capacity(len);
    
    // First OBV value is 0 as starting point
    obv_values.push(0.0);
    
    let mut cumulative = 0.0;
    
    for i in 1..len {
        let curr_close = close_ca.get(i).unwrap_or(f64::NAN);
        let prev_close = close_ca.get(i - 1).unwrap_or(f64::NAN);
        let vol = volume_ca.get(i).unwrap_or(0.0);
        
        if curr_close > prev_close {
            cumulative += vol;
        } else if curr_close < prev_close {
            cumulative -= vol;
        }
        // If equal, cumulative stays the same
        
        obv_values.push(cumulative);
    }
    
    Ok(Series::new("obv", obv_values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obv_length() {
        let close = Series::new("close", vec![100.0, 101.0, 102.0, 101.0, 103.0]);
        let volume = Series::new("volume", vec![1000.0, 1200.0, 800.0, 900.0, 1100.0]);
        let result = obv(&close, &volume).unwrap();
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_obv_increases_on_up_days() {
        // All up days - OBV should keep increasing
        let close = Series::new("close", vec![100.0, 101.0, 102.0, 103.0, 104.0]);
        let volume = Series::new("volume", vec![1000.0; 5]);
        let result = obv(&close, &volume).unwrap();
        let vals = result.f64().unwrap();

        let first = vals.get(1).unwrap();
        let last = vals.get(4).unwrap();
        assert!(last > first, "OBV should increase on all up days");
    }

    #[test]
    fn test_obv_decreases_on_down_days() {
        // All down days - OBV should decrease
        let close = Series::new("close", vec![105.0, 104.0, 103.0, 102.0, 101.0]);
        let volume = Series::new("volume", vec![1000.0; 5]);
        let result = obv(&close, &volume).unwrap();
        let vals = result.f64().unwrap();

        let first = vals.get(1).unwrap();
        let last = vals.get(4).unwrap();
        assert!(last < first, "OBV should decrease on all down days");
    }
}
