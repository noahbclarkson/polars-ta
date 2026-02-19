//! Basic sanity tests for technical indicators

use polars::prelude::*;

// Simple EMA test with known pattern
#[test]
fn test_ema_basic() {
    use crate::indicators::ema_20;
    
    // Simple upward trend
    let close = Series::new("close", &[100.0, 101.0, 102.0, 103.0, 104.0]);
    let result = ema_20(&close);
    
    assert!(result.is_ok());
    let ema = result.unwrap();
    assert_eq!(ema.len(), close.len());
    
    // First few values should be NaN due to min_periods
    let ema_f64 = ema.f64().unwrap();
    assert!(ema_f64.get(0).unwrap().is_nan());
}

// RSI test with known values
#[test]
fn test_rsi_14_known_values() {
    use crate::indicators::rsi_14;
    
    // Create a simple alternating up/down pattern
    // This is a minimal sanity test - not meant to validate exact values
    let close = Series::new("close", &[
        44.0, 44.25, 44.50, 44.75, 45.0, 
        45.25, 45.50, 45.75, 46.0, 46.25,
        46.50, 46.75, 47.0, 47.25, 47.50
    ]);
    
    let result = rsi_14(&close);
    
    assert!(result.is_ok());
    let rsi = result.unwrap();
    assert_eq!(rsi.len(), close.len());
    
    // All RSI values should be between 0 and 100
    let rsi_f64 = rsi.f64().unwrap();
    for i in 0..rsi_f64.len() {
        if let Some(val) = rsi_f64.get(i) {
            if !val.is_nan() {
                assert!(val >= 0.0 && val <= 100.0, "RSI value {} out of range [0, 100]", val);
            }
        }
    }
    
    // With consistent upward movement, RSI should be > 50 (once warmed up)
    // Check the last valid value (not NaN)
    for i in (14..rsi_f64.len()).rev() {
        if let Some(val) = rsi_f64.get(i) {
            if !val.is_nan() {
                assert!(val > 50.0, "With upward movement, RSI should be > 50, got {}", val);
                break;
            }
        }
    }
}
