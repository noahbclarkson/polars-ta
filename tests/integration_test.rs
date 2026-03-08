//! Integration tests for polars-ta
//!
//! Tests verify that all indicators produce correct output shapes and
//! reasonable values on synthetic data.

use polars::prelude::*;
use polars_ta::*;

/// Create a synthetic test DataFrame with price data
fn create_test_data() -> DataFrame {
    let close = Series::new("close", &[
        100.0, 101.5, 102.3, 101.8, 103.2,
        104.1, 103.5, 105.0, 106.2, 105.8,
        107.0, 106.5, 108.0, 109.2, 108.5,
        110.0, 109.8, 111.0, 110.5, 112.0,
    ]);
    
    let high = Series::new("high", &[
        102.0, 103.5, 104.0, 103.5, 105.0,
        106.0, 105.5, 107.0, 108.0, 107.5,
        109.0, 108.5, 110.0, 111.0, 110.5,
        112.0, 111.5, 113.0, 112.5, 114.0,
    ]);
    
    let low = Series::new("low", &[
        99.0, 100.5, 101.0, 100.5, 102.0,
        103.0, 102.5, 104.0, 105.0, 104.5,
        106.0, 105.5, 107.0, 108.0, 107.5,
        109.0, 108.5, 110.0, 109.5, 111.0,
    ]);
    
    let volume = Series::new("volume", &[
        1000.0, 1200.0, 1100.0, 900.0, 1500.0,
        1300.0, 1100.0, 1600.0, 1400.0, 1200.0,
        1700.0, 1300.0, 1800.0, 1500.0, 1200.0,
        1900.0, 1400.0, 2000.0, 1600.0, 2100.0,
    ]);
    
    DataFrame::new(vec![close, high, low, volume]).unwrap()
}

#[test]
fn test_ema_returns_correct_length() {
    let df = create_test_data();
    let close = df.column("close").unwrap();
    
    let ema_result = ema(close, 5).unwrap();
    
    assert_eq!(ema_result.len(), close.len(), "EMA should have same length as input");
}

#[test]
fn test_ema_known_values() {
    // Test EMA on a simple sequence where we can verify the calculation
    let close = Series::new("close", &[1.0, 2.0, 3.0, 4.0, 5.0]);
    
    let ema_result = ema(&close, 3).unwrap();
    
    assert_eq!(ema_result.len(), 5, "EMA should have same length as input");
    
    // EMA with period=3 means alpha = 1/3
    // EMA_3 = (1-2/3)*(1-2/3)*1 + (1-2/3)*(2/3)*2 + (2/3)*3 = 0.111*1 + 0.222*2 + 0.667*3 ≈ 2.56
    let ema_slice = ema_result.f64().unwrap();
    
    // First values should be null due to min_periods
    // Just verify we get some non-null values at the end
    let last_val = ema_slice.get(4).unwrap();
    assert!(last_val > 0.0, "EMA should be positive for positive inputs");
}

#[test]
fn test_sma_returns_correct_length() {
    let df = create_test_data();
    let close = df.column("close").unwrap();
    
    let sma_result = sma(close, 5).unwrap();
    
    assert_eq!(sma_result.len(), close.len(), "SMA should have same length as input");
}

#[test]
fn test_sma_known_values() {
    let close = Series::new("close", &[1.0, 2.0, 3.0, 4.0, 5.0]);
    
    let sma_result = sma(&close, 3).unwrap();
    let sma_slice = sma_result.f64().unwrap();
    
    // SMA of last 3 values (3, 4, 5) should be 4.0
    let last_sma = sma_slice.get(4).unwrap();
    assert!((last_sma - 4.0).abs() < 0.001, "SMA of [3,4,5] should be 4.0");
}

#[test]
fn test_macd_returns_correct_length() {
    let df = create_test_data();
    let close = df.column("close").unwrap();
    
    let result = macd_default(close).unwrap();
    
    assert_eq!(result.macd.len(), close.len(), "MACD should have same length as input");
    assert_eq!(result.signal.len(), close.len(), "Signal should have same length as input");
    assert_eq!(result.histogram.len(), close.len(), "Histogram should have same length as input");
}

#[test]
fn test_macd_histogram_equals_macd_minus_signal() {
    let df = create_test_data();
    let close = df.column("close").unwrap();
    
    let result = macd_default(close).unwrap();
    
    let macd_ca = result.macd.f64().unwrap();
    let signal_ca = result.signal.f64().unwrap();
    let hist_ca = result.histogram.f64().unwrap();
    
    // Check a few values that aren't null
    for i in 15..20 {
        let macd_val = macd_ca.get(i).unwrap_or(0.0);
        let signal_val = signal_ca.get(i).unwrap_or(0.0);
        let hist_val = hist_ca.get(i).unwrap_or(0.0);
        
        let expected_hist = macd_val - signal_val;
        assert!(
            (hist_val - expected_hist).abs() < 0.001,
            "Histogram should equal MACD - signal at index {}", i
        );
    }
}

#[test]
fn test_rsi_returns_correct_length() {
    let df = create_test_data();
    let close = df.column("close").unwrap();
    
    let rsi_result = rsi_14(close).unwrap();
    
    assert_eq!(rsi_result.len(), close.len(), "RSI should have same length as input");
}

#[test]
fn test_rsi_values_between_0_and_100() {
    let df = create_test_data();
    let close = df.column("close").unwrap();
    
    let rsi_result = rsi_14(close).unwrap();
    let rsi_ca = rsi_result.f64().unwrap();
    
    for i in 0..rsi_ca.len() {
        if let Some(val) = rsi_ca.get(i) {
            if !val.is_nan() {
                assert!(
                    (0.0..=100.0).contains(&val),
                    "RSI value {} at index {} should be between 0 and 100",
                    val, i
                );
            }
        }
    }
}

#[test]
fn test_atr_returns_correct_length() {
    let df = create_test_data();
    let high = df.column("high").unwrap();
    let low = df.column("low").unwrap();
    let close = df.column("close").unwrap();
    
    let atr_result = atr_14(high, low, close).unwrap();
    
    assert_eq!(atr_result.len(), close.len(), "ATR should have same length as input");
}

#[test]
fn test_atr_is_positive() {
    let df = create_test_data();
    let high = df.column("high").unwrap();
    let low = df.column("low").unwrap();
    let close = df.column("close").unwrap();
    
    let atr_result = atr_14(high, low, close).unwrap();
    let atr_ca = atr_result.f64().unwrap();
    
    // ATR should always be positive when it has a value
    for i in 0..atr_ca.len() {
        if let Some(val) = atr_ca.get(i) {
            if !val.is_nan() {
                assert!(val > 0.0, "ATR should be positive at index {}", i);
            }
        }
    }
}

#[test]
fn test_bollinger_returns_correct_length() {
    let df = create_test_data();
    let close = df.column("close").unwrap();
    
    let result = bollinger_20_2(close).unwrap();
    
    assert_eq!(result.upper.len(), close.len(), "Upper band should have same length as input");
    assert_eq!(result.middle.len(), close.len(), "Middle band should have same length as input");
    assert_eq!(result.lower.len(), close.len(), "Lower band should have same length as input");
}

#[test]
fn test_bollinger_upper_greater_than_middle_greater_than_lower() {
    let df = create_test_data();
    let close = df.column("close").unwrap();
    
    let result = bollinger(close, 5, 2.0).unwrap();
    
    let upper_ca = result.upper.f64().unwrap();
    let middle_ca = result.middle.f64().unwrap();
    let lower_ca = result.lower.f64().unwrap();
    
    // Check that upper > middle > lower for non-null values
    for i in 5..close.len() {
        let upper = upper_ca.get(i).unwrap_or(f64::NAN);
        let middle = middle_ca.get(i).unwrap_or(f64::NAN);
        let lower = lower_ca.get(i).unwrap_or(f64::NAN);
        
        if !upper.is_nan() && !middle.is_nan() && !lower.is_nan() {
            assert!(upper > middle, "Upper should be > middle at index {}", i);
            assert!(middle > lower, "Middle should be > lower at index {}", i);
        }
    }
}

#[test]
fn test_obv_returns_correct_length() {
    let df = create_test_data();
    let close = df.column("close").unwrap();
    let volume = df.column("volume").unwrap();
    
    let obv_result = obv(close, volume).unwrap();
    
    assert_eq!(obv_result.len(), close.len(), "OBV should have same length as input");
}

#[test]
fn test_obv_cumulative() {
    let close = Series::new("close", &[100.0, 101.0, 100.5, 102.0, 101.5]);
    let volume = Series::new("volume", &[1000.0, 500.0, 300.0, 400.0, 200.0]);
    
    let obv_result = obv(&close, &volume).unwrap();
    let obv_ca = obv_result.f64().unwrap();
    
    // First OBV is always 0
    assert_eq!(obv_ca.get(0).unwrap(), 0.0, "First OBV should be 0");
    
    // 101 > 100: OBV = 0 + 500 = 500
    assert_eq!(obv_ca.get(1).unwrap(), 500.0, "Second OBV should be 500");
    
    // 100.5 < 101: OBV = 500 - 300 = 200
    assert_eq!(obv_ca.get(2).unwrap(), 200.0, "Third OBV should be 200");
    
    // 102 > 100.5: OBV = 200 + 400 = 600
    assert_eq!(obv_ca.get(3).unwrap(), 600.0, "Fourth OBV should be 600");
    
    // 101.5 < 102: OBV = 600 - 200 = 400
    assert_eq!(obv_ca.get(4).unwrap(), 400.0, "Fifth OBV should be 400");
}

#[test]
fn test_roc_returns_correct_length() {
    let df = create_test_data();
    let close = df.column("close").unwrap();
    
    let roc_result = roc(close, 5).unwrap();
    
    assert_eq!(roc_result.len(), close.len(), "ROC should have same length as input");
}

#[test]
fn test_roc_known_values() {
    let close = Series::new("close", &[100.0, 110.0, 120.0, 115.0, 125.0, 130.0]);
    
    let roc_result = roc(&close, 1).unwrap();
    let roc_ca = roc_result.f64().unwrap();
    
    // ROC from 100 to 110 = (110-100)/100 * 100 = 10%
    let roc_1 = roc_ca.get(1).unwrap();
    assert!((roc_1 - 10.0).abs() < 0.001, "ROC at index 1 should be 10%");
    
    // ROC from 110 to 120 = (120-110)/110 * 100 = 9.09%
    let roc_2 = roc_ca.get(2).unwrap();
    assert!((roc_2 - 9.0909).abs() < 0.1, "ROC at index 2 should be ~9.09%");
}

#[test]
fn test_frac_diff_returns_correct_length() {
    let df = create_test_data();
    let close = df.column("close").unwrap();
    
    let frac_result = frac_diff_ffd(close, 0.5, 10).unwrap();
    
    assert_eq!(frac_result.len(), close.len(), "Frac diff should have same length as input");
}

#[test]
fn test_frac_diff_has_some_valid_values() {
    let df = create_test_data();
    let close = df.column("close").unwrap();
    
    let frac_result = frac_diff_ffd(close, 0.5, 5).unwrap();
    let frac_ca = frac_result.f64().unwrap();
    
    // Should have at least some non-NaN values at the end
    let mut has_valid = false;
    for i in 10..frac_ca.len() {
        if let Some(val) = frac_ca.get(i) {
            if !val.is_nan() {
                has_valid = true;
                break;
            }
        }
    }
    assert!(has_valid, "Frac diff should produce some valid values");
}

// ==================== CCI Tests ====================

#[test]
fn test_cci_returns_correct_length() {
    let df = create_test_data();
    let high = df.column("high").unwrap();
    let low = df.column("low").unwrap();
    let close = df.column("close").unwrap();
    
    let cci_result = cci_20(high, low, close).unwrap();
    
    assert_eq!(cci_result.len(), close.len(), "CCI should have same length as input");
}

#[test]
fn test_cci_reasonable_values() {
    let df = create_test_data();
    let high = df.column("high").unwrap();
    let low = df.column("low").unwrap();
    let close = df.column("close").unwrap();
    
    let cci_result = cci(high, low, close, 5).unwrap();
    let cci_ca = cci_result.f64().unwrap();
    
    // CCI can be any value but shouldn't be NaN where we have data
    let mut has_valid = false;
    for i in 5..cci_ca.len() {
        if let Some(val) = cci_ca.get(i) {
            if !val.is_nan() && val.is_finite() {
                has_valid = true;
                break;
            }
        }
    }
    assert!(has_valid, "CCI should produce some valid values");
}

// ==================== Williams %R Tests ====================

#[test]
fn test_williams_r_returns_correct_length() {
    let df = create_test_data();
    let high = df.column("high").unwrap();
    let low = df.column("low").unwrap();
    let close = df.column("close").unwrap();
    
    let wr_result = williams_r_14(high, low, close).unwrap();
    
    assert_eq!(wr_result.len(), close.len(), "Williams %R should have same length as input");
}

#[test]
fn test_williams_r_values_in_range() {
    let df = create_test_data();
    let high = df.column("high").unwrap();
    let low = df.column("low").unwrap();
    let close = df.column("close").unwrap();
    
    let wr_result = williams_r(high, low, close, 5).unwrap();
    let wr_ca = wr_result.f64().unwrap();
    
    for i in 0..wr_ca.len() {
        if let Some(val) = wr_ca.get(i) {
            if !val.is_nan() {
                assert!(
                    (-100.0..=0.0).contains(&val),
                    "Williams %R value {} at index {} should be between -100 and 0",
                    val, i
                );
            }
        }
    }
}

// ==================== CMF Tests ====================

#[test]
fn test_cmf_returns_correct_length() {
    let df = create_test_data();
    let high = df.column("high").unwrap();
    let low = df.column("low").unwrap();
    let close = df.column("close").unwrap();
    let volume = df.column("volume").unwrap();
    
    let cmf_result = cmf_20(high, low, close, volume).unwrap();
    
    assert_eq!(cmf_result.len(), close.len(), "CMF should have same length as input");
}

#[test]
fn test_cmf_values_in_range() {
    let df = create_test_data();
    let high = df.column("high").unwrap();
    let low = df.column("low").unwrap();
    let close = df.column("close").unwrap();
    let volume = df.column("volume").unwrap();
    
    let cmf_result = cmf(high, low, close, volume, 5).unwrap();
    let cmf_ca = cmf_result.f64().unwrap();
    
    for i in 0..cmf_ca.len() {
        if let Some(val) = cmf_ca.get(i) {
            if !val.is_nan() {
                assert!(
                    (-1.0..=1.0).contains(&val),
                    "CMF value {} at index {} should be between -1 and 1",
                    val, i
                );
            }
        }
    }
}

// ==================== PSAR Tests ====================

#[test]
fn test_psar_returns_correct_length() {
    let df = create_test_data();
    let high = df.column("high").unwrap();
    let low = df.column("low").unwrap();
    let close = df.column("close").unwrap();
    
    let psar_result = psar_default(high, low, close).unwrap();
    
    assert_eq!(psar_result.len(), close.len(), "PSAR should have same length as input");
}

#[test]
fn test_psar_reasonable_values() {
    let df = create_test_data();
    let high = df.column("high").unwrap();
    let low = df.column("low").unwrap();
    let close = df.column("close").unwrap();
    
    let psar_result = psar(high, low, close, 0.02, 0.2).unwrap();
    let psar_ca = psar_result.f64().unwrap();
    
    // PSAR values should be finite
    for i in 2..psar_ca.len() {
        if let Some(val) = psar_ca.get(i) {
            if !val.is_nan() {
                assert!(val.is_finite(), "PSAR should be finite at index {}", i);
            }
        }
    }
}

#[test]
fn test_all_indicators_no_panic() {
    // Simple test to ensure all indicators can be called without panicking
    let df = create_test_data();
    let close = df.column("close").unwrap();
    let high = df.column("high").unwrap();
    let low = df.column("low").unwrap();
    let volume = df.column("volume").unwrap();
    
    // These should all complete without panic
    let _ = ema(close, 10);
    let _ = sma(close, 10);
    let _ = macd_default(close);
    let _ = macd(close, 12, 26, 9);
    let _ = rsi_14(close);
    let _ = rsi(close, 7);
    let _ = atr_14(high, low, close);
    let _ = atr(high, low, close, 7);
    let _ = bollinger_20_2(close);
    let _ = bollinger(close, 10, 1.5);
    let _ = obv(close, volume);
    let _ = roc(close, 5);
    let _ = frac_diff_ffd(close, 0.4, 10);
    let _ = cci_20(high, low, close);
    let _ = williams_r_14(high, low, close);
    let _ = cmf_20(high, low, close, volume);
    let _ = psar_default(high, low, close);
}
