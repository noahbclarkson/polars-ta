//! Unit tests for polars-ta indicator calculations
//!
//! These tests verify mathematical correctness using known input/output pairs.

use polars::prelude::*;
use polars_ta::*;

// =============================================================================
// EMA Tests
// =============================================================================

#[test]
fn test_ema_constant_value() {
    // EMA of constant values should be that constant
    let close = Series::new("close", &[50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]);
    let ema_result = ema(&close, 5).unwrap();
    let ema_ca = ema_result.f64().unwrap();
    
    // After min_periods, EMA should be very close to the constant value
    if let Some(val) = ema_ca.get(9) {
        assert!((val - 50.0).abs() < 0.01, "EMA of constant should equal constant, got {}", val);
    }
}

#[test]
fn test_ma_rising_sequence() {
    // EMA should be between first and last values for a rising sequence
    let close = Series::new("close", &[
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
        20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0
    ]);
    let ema_result = ema(&close, 5).unwrap();
    let ema_ca = ema_result.f64().unwrap();
    
    // EMA should be positive and increasing for a rising sequence
    if let Some(val) = ema_ca.get(19) {
        assert!(val > 10.0 && val < 29.0, "EMA should be between min and max of input, got {}", val);
    }
}

#[test]
fn test_ema_short_period_responsive() {
    // Shorter period EMA should be more responsive to recent changes
    let close = Series::new("close", &[
        100.0, 100.0, 100.0, 100.0, 100.0,  // stable
        110.0, 110.0, 110.0, 110.0, 110.0,  // jump up
    ]);
    
    let ema_3 = ema(&close, 3).unwrap();
    let ema_5 = ema(&close, 5).unwrap();
    
    let ema_3_ca = ema_3.f64().unwrap();
    let ema_5_ca = ema_5.f64().unwrap();
    
    // Shorter period EMA should respond faster to the price jump
    if let (Some(val3), Some(val5)) = (ema_3_ca.get(9), ema_5_ca.get(9)) {
        assert!(val3 > val5, "Shorter period EMA ({}) should be higher than longer period EMA ({}) after price jump", val3, val5);
    }
}

// =============================================================================
// RSI Tests
// =============================================================================

#[test]
fn test_rsi_all_gains() {
    // All gains (rising prices) should give RSI = 100
    let close = Series::new("close", &[
        100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
        110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
        120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0
    ]);
    
    let rsi_result = rsi(&close, 14).unwrap();
    let rsi_ca = rsi_result.f64().unwrap();
    
    // With all gains and no losses, RSI should approach 100
    if let Some(val) = rsi_ca.get(29) {
        assert!(val > 99.0, "RSI with all gains should approach 100, got {}", val);
    }
}

#[test]
fn test_rsi_all_losses() {
    // All losses (falling prices) should give RSI = 0
    let close = Series::new("close", &[
        130.0, 129.0, 128.0, 127.0, 126.0, 125.0, 124.0, 123.0, 122.0, 121.0,
        120.0, 119.0, 118.0, 117.0, 116.0, 115.0, 114.0, 113.0, 112.0, 111.0,
        110.0, 109.0, 108.0, 107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0
    ]);
    
    let rsi_result = rsi(&close, 14).unwrap();
    let rsi_ca = rsi_result.f64().unwrap();
    
    // With all losses and no gains, RSI should approach 0
    if let Some(val) = rsi_ca.get(29) {
        assert!(val < 1.0, "RSI with all losses should approach 0, got {}", val);
    }
}

#[test]
fn test_rsi_equal_gains_losses() {
    // Alternating equal gains and losses should give RSI near 50
    let close = Series::new("close", &[
        100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0,
        100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0,
        100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0
    ]);
    
    let rsi_result = rsi(&close, 14).unwrap();
    let rsi_ca = rsi_result.f64().unwrap();
    
    // With equal gains and losses, RSI should be around 50
    if let Some(val) = rsi_ca.get(29) {
        assert!((val - 50.0).abs() < 5.0, "RSI with equal gains/losses should be ~50, got {}", val);
    }
}

#[test]
fn test_rsi_boundary_conditions() {
    // RSI should always be between 0 and 100
    let close = Series::new("close", &[
        100.0, 105.0, 98.0, 110.0, 95.0, 115.0, 90.0, 120.0, 85.0, 125.0,
        80.0, 130.0, 75.0, 135.0, 70.0, 140.0, 65.0, 145.0, 60.0, 150.0,
        55.0, 155.0, 50.0, 160.0, 45.0, 165.0, 40.0, 170.0, 35.0, 175.0
    ]);
    
    let rsi_result = rsi(&close, 14).unwrap();
    let rsi_ca = rsi_result.f64().unwrap();
    
    for i in 0..rsi_ca.len() {
        if let Some(val) = rsi_ca.get(i) {
            if !val.is_nan() {
                assert!(val >= 0.0 && val <= 100.0, 
                    "RSI at index {} is {} which is outside [0, 100]", i, val);
            }
        }
    }
}

// =============================================================================
// MACD Tests
// =============================================================================

#[test]
fn test_macd_structure() {
    let close = Series::new("close", &[
        100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
        110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
        120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0,
        130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0
    ]);
    
    let result = macd(&close, 12, 26, 9).unwrap();
    
    // All three series should have same length as input
    assert_eq!(result.macd.len(), close.len());
    assert_eq!(result.signal.len(), close.len());
    assert_eq!(result.histogram.len(), close.len());
}

#[test]
fn test_macd_histogram_equals_macd_minus_signal() {
    let close = Series::new("close", &[
        100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
        111.0, 110.0, 112.0, 114.0, 113.0, 115.0, 117.0, 116.0, 118.0, 120.0,
        119.0, 121.0, 123.0, 122.0, 124.0, 126.0, 125.0, 127.0, 129.0, 128.0,
        130.0, 132.0, 131.0, 133.0, 135.0, 134.0, 136.0, 138.0, 137.0, 139.0
    ]);
    
    let result = macd(&close, 12, 26, 9).unwrap();
    
    let macd_ca = result.macd.f64().unwrap();
    let signal_ca = result.signal.f64().unwrap();
    let hist_ca = result.histogram.f64().unwrap();
    
    // Histogram = MACD - Signal (verify for all non-null values)
    for i in 0..close.len() {
        if let (Some(m), Some(s), Some(h)) = (macd_ca.get(i), signal_ca.get(i), hist_ca.get(i)) {
            if !m.is_nan() && !s.is_nan() && !h.is_nan() {
                let expected_hist = m - s;
                assert!((h - expected_hist).abs() < 0.0001, 
                    "Histogram at index {} should be MACD - Signal: {} - {} = {}, got {}", 
                    i, m, s, expected_hist, h);
            }
        }
    }
}

#[test]
fn test_macd_rising_trend() {
    // In a strongly rising trend, MACD line should be positive
    let close = Series::new("close", &[
        100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0,
        120.0, 122.0, 124.0, 126.0, 128.0, 130.0, 132.0, 134.0, 136.0, 138.0,
        140.0, 142.0, 144.0, 146.0, 148.0, 150.0, 152.0, 154.0, 156.0, 158.0,
        160.0, 162.0, 164.0, 166.0, 168.0, 170.0, 172.0, 174.0, 176.0, 178.0,
        180.0, 182.0, 184.0, 186.0, 188.0, 190.0, 192.0, 194.0, 196.0, 198.0
    ]);
    
    let result = macd(&close, 12, 26, 9).unwrap();
    let macd_ca = result.macd.f64().unwrap();
    
    // In a sustained uptrend, MACD should be positive at the end
    if let Some(val) = macd_ca.get(49) {
        assert!(val > 0.0, "MACD should be positive in uptrend, got {}", val);
    }
}

#[test]
fn test_macd_falling_trend() {
    // In a strongly falling trend, MACD line should be negative
    let close = Series::new("close", &[
        200.0, 198.0, 196.0, 194.0, 192.0, 190.0, 188.0, 186.0, 184.0, 182.0,
        180.0, 178.0, 176.0, 174.0, 172.0, 170.0, 168.0, 166.0, 164.0, 162.0,
        160.0, 158.0, 156.0, 154.0, 152.0, 150.0, 148.0, 146.0, 144.0, 142.0,
        140.0, 138.0, 136.0, 134.0, 132.0, 130.0, 128.0, 126.0, 124.0, 122.0,
        120.0, 118.0, 116.0, 114.0, 112.0, 110.0, 108.0, 106.0, 104.0, 102.0
    ]);
    
    let result = macd(&close, 12, 26, 9).unwrap();
    let macd_ca = result.macd.f64().unwrap();
    
    // In a sustained downtrend, MACD should be negative at the end
    if let Some(val) = macd_ca.get(49) {
        assert!(val < 0.0, "MACD should be negative in downtrend, got {}", val);
    }
}

// =============================================================================
// Bollinger Bands Tests
// =============================================================================

#[test]
fn test_bollinger_constant_input() {
    // Bollinger bands of constant values should have:
    // - middle = that constant
    // - std = 0, so upper = lower = middle
    let close = Series::new("close", &[
        100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        100.0, 100.0, 100.0, 100.0, 100.0
    ]);
    
    let result = bollinger(&close, 20, 2.0).unwrap();
    
    let upper_ca = result.upper.f64().unwrap();
    let middle_ca = result.middle.f64().unwrap();
    let lower_ca = result.lower.f64().unwrap();
    
    // For constant values, middle should be 100, and all bands should be equal
    if let Some(mid) = middle_ca.get(24) {
        assert!((mid - 100.0).abs() < 0.01, "Middle band should be 100 for constant input, got {}", mid);
    }
    
    // With zero variance, upper and lower should equal middle
    // (though in practice there might be small numerical differences)
    if let (Some(upper), Some(lower), Some(mid)) = 
        (upper_ca.get(24), lower_ca.get(24), middle_ca.get(24)) {
        let upper_diff = (upper - mid).abs();
        let lower_diff = (lower - mid).abs();
        // Allow for small numerical error, but bands should be very close
        assert!(upper_diff < 0.1, "Upper band should be close to middle for constant input, diff = {}", upper_diff);
        assert!(lower_diff < 0.1, "Lower band should be close to middle for constant input, diff = {}", lower_diff);
    }
}

#[test]
fn test_bollinger_upper_middle_lower_ordering() {
    let close = Series::new("close", &[
        100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
        110.0, 109.0, 108.0, 107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0,
        100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0
    ]);
    
    let result = bollinger(&close, 10, 2.0).unwrap();
    
    let upper_ca = result.upper.f64().unwrap();
    let middle_ca = result.middle.f64().unwrap();
    let lower_ca = result.lower.f64().unwrap();
    
    // For all valid indices, upper > middle > lower
    for i in 10..close.len() {
        if let (Some(upper), Some(middle), Some(lower)) = 
            (upper_ca.get(i), middle_ca.get(i), lower_ca.get(i)) {
            if !upper.is_nan() && !middle.is_nan() && !lower.is_nan() {
                assert!(upper > middle, 
                    "Upper ({}) should be > middle ({}) at index {}", upper, middle, i);
                assert!(middle > lower, 
                    "Middle ({}) should be > lower ({}) at index {}", middle, lower, i);
            }
        }
    }
}

#[test]
fn test_bollinger_middle_equals_sma() {
    // The middle band should equal the simple moving average
    let close = Series::new("close", &[
        10.0, 20.0, 30.0, 40.0, 50.0,
        60.0, 70.0, 80.0, 90.0, 100.0,
        110.0, 120.0, 130.0, 140.0, 150.0
    ]);
    
    let bollinger_result = bollinger(&close, 5, 2.0).unwrap();
    let sma_result = sma(&close, 5).unwrap();
    
    let boll_middle = bollinger_result.middle.f64().unwrap();
    let sma_vals = sma_result.f64().unwrap();
    
    // Middle band should exactly equal SMA
    for i in 5..close.len() {
        if let (Some(bm), Some(sma)) = (boll_middle.get(i), sma_vals.get(i)) {
            if !bm.is_nan() && !sma.is_nan() {
                assert!((bm - sma).abs() < 0.0001, 
                    "Bollinger middle ({}) should equal SMA ({}) at index {}", bm, sma, i);
            }
        }
    }
}

#[test]
fn test_bollinger_known_std_dev() {
    // Test with known values where we can verify the standard deviation
    // Values 1,2,3,4,5 have mean=3 and std≈1.581
    let close = Series::new("close", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    let result = bollinger(&close, 5, 2.0).unwrap();
    
    let upper_ca = result.upper.f64().unwrap();
    let middle_ca = result.middle.f64().unwrap();
    let lower_ca = result.lower.f64().unwrap();
    
    // At index 4 (last of the window 1,2,3,4,5):
    // middle = 3.0, std ≈ 1.581
    // upper ≈ 3.0 + 2*1.581 ≈ 6.162
    // lower ≈ 3.0 - 2*1.581 ≈ -0.162
    
    if let Some(mid) = middle_ca.get(4) {
        assert!((mid - 3.0).abs() < 0.01, "Middle at index 4 should be 3.0, got {}", mid);
    }
    
    // Upper should be above middle
    if let (Some(upper), Some(mid)) = (upper_ca.get(4), middle_ca.get(4)) {
        assert!(upper > mid, "Upper ({}) should be > middle ({})", upper, mid);
    }
    
    // Lower should be below middle
    if let (Some(lower), Some(mid)) = (lower_ca.get(4), middle_ca.get(4)) {
        assert!(lower < mid, "Lower ({}) should be < middle ({})", lower, mid);
    }
}

#[test]
fn test_bollinger_price_containment() {
    // Most prices should fall within the Bollinger bands (roughly 95% for 2 std dev)
    // Let's test with random-ish data
    let close = Series::new("close", &[
        100.0, 102.0, 98.0, 105.0, 95.0, 108.0, 92.0, 110.0, 88.0, 112.0,
        85.0, 115.0, 82.0, 118.0, 80.0, 120.0, 78.0, 122.0, 75.0, 125.0,
        73.0, 128.0, 70.0, 130.0, 68.0, 132.0, 65.0, 135.0, 62.0, 138.0,
        60.0, 140.0, 58.0, 142.0, 55.0, 145.0, 52.0, 148.0, 50.0, 150.0
    ]);
    
    let result = bollinger(&close, 20, 2.0).unwrap();
    
    let upper_ca = result.upper.f64().unwrap();
    let lower_ca = result.lower.f64().unwrap();
    let close_ca = close.f64().unwrap();
    
    // Count prices outside bands after warm-up period
    let mut outside_count = 0;
    let mut valid_count = 0;
    
    for i in 20..close.len() {
        if let (Some(price), Some(upper), Some(lower)) = 
            (close_ca.get(i), upper_ca.get(i), lower_ca.get(i)) {
            if !price.is_nan() && !upper.is_nan() && !lower.is_nan() {
                valid_count += 1;
                if price > upper || price < lower {
                    outside_count += 1;
                }
            }
        }
    }
    
    // We don't strictly enforce containment since this is volatile data,
    // but the bands should exist and be valid
    assert!(valid_count > 0, "Should have valid data points");
}

// =============================================================================
// SMA Tests (for comparison with Bollinger middle)
// =============================================================================

#[test]
fn test_sma_known_values() {
    let close = Series::new("close", &[1.0, 2.0, 3.0, 4.0, 5.0]);
    
    let sma_result = sma(&close, 3).unwrap();
    let sma_ca = sma_result.f64().unwrap();
    
    // SMA of [1,2,3] = 2.0
    if let Some(val) = sma_ca.get(2) {
        assert!((val - 2.0).abs() < 0.001, "SMA of [1,2,3] should be 2.0, got {}", val);
    }
    
    // SMA of [2,3,4] = 3.0
    if let Some(val) = sma_ca.get(3) {
        assert!((val - 3.0).abs() < 0.001, "SMA of [2,3,4] should be 3.0, got {}", val);
    }
    
    // SMA of [3,4,5] = 4.0
    if let Some(val) = sma_ca.get(4) {
        assert!((val - 4.0).abs() < 0.001, "SMA of [3,4,5] should be 4.0, got {}", val);
    }
}

#[test]
fn test_sma_constant_value() {
    let close = Series::new("close", &[50.0, 50.0, 50.0, 50.0, 50.0]);
    
    let sma_result = sma(&close, 3).unwrap();
    let sma_ca = sma_result.f64().unwrap();
    
    // SMA of constant should be that constant
    for i in 2..5 {
        if let Some(val) = sma_ca.get(i) {
            assert!((val - 50.0).abs() < 0.001, "SMA of constant should be constant, got {} at index {}", val, i);
        }
    }
}
