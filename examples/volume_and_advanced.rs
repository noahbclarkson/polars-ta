//! Volume Indicators and Advanced Features Example
//!
//! This example demonstrates volume-based indicators (OBV, MFI, CMF, VWAP)
//! and advanced features like fractional differentiation.

use polars::prelude::*;
use polars_ta::{obv, mfi, cmf_20, rolling_vwap, frac_diff_ffd, psar_default, roc};

fn main() -> anyhow::Result<()> {
    println!("=== Volume Indicators and Advanced Features Example ===\n");

    // Create sample OHLCV data
    let high = Series::new(
        "high",
        &[
            105.0, 106.0, 107.0, 106.5, 108.0, 109.0, 108.5, 110.0, 111.0, 110.5,
            112.0, 113.0, 112.5, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
        ],
    );

    let low = Series::new(
        "low",
        &[
            100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
            107.0, 108.0, 107.5, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0,
        ],
    );

    let close = Series::new(
        "close",
        &[
            103.0, 104.0, 105.0, 104.5, 106.0, 107.0, 106.5, 108.0, 109.0, 108.5,
            110.0, 111.0, 110.5, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0,
        ],
    );

    let volume = Series::new(
        "volume",
        &[
            1000.0, 1200.0, 800.0, 1500.0, 900.0, 1100.0, 1300.0, 750.0, 1600.0, 850.0,
            1250.0, 950.0, 1400.0, 1150.0, 1050.0, 1350.0, 800.0, 1450.0, 1000.0, 1200.0,
        ],
    );

    // Volume Indicators
    println!("1. On-Balance Volume (OBV)");
    println!("   Cumulative volume indicator based on price direction");
    let obv_values = obv(&close, &volume)?;
    println!("   OBV (last 5): {:?}", &obv_values.f64()?.into_no_null_iter().rev().take(5).collect::<Vec<_>>());
    println!();

    println!("2. Money Flow Index (MFI)");
    println!("   Volume-weighted RSI (requires DataFrame with OHLCV)");
    let df = DataFrame::new(vec![
        high.clone(),
        low.clone(),
        close.clone(),
        volume.clone(),
    ])?;
    let mfi_values = mfi(&df, 14)?;
    println!("   MFI(14) last value: {:?}", mfi_values.f64()?.into_no_null_iter().last());
    println!();

    println!("3. Chaikin Money Flow (CMF)");
    println!("   Measures buying and selling pressure");
    let cmf_values = cmf_20(&high, &low, &close, &volume)?;
    println!("   CMF(20) last value: {:?}", cmf_values.f64()?.into_no_null_iter().last());
    println!();

    println!("4. Rolling VWAP");
    println!("   VWAP over a rolling window instead of cumulative");
    let rolling_vwap_values = rolling_vwap(&high, &low, &close, &volume, 10)?;
    println!("   Rolling VWAP(10) last 3: {:?}", &rolling_vwap_values.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!();

    // Advanced Indicators
    println!("5. Parabolic SAR (PSAR)");
    println!("   Trend-following indicator with stop and reverse points");
    let psar = psar_default(&high, &low, &close)?;
    println!("   PSAR (last 5): {:?}", &psar.f64()?.into_no_null_iter().rev().take(5).collect::<Vec<_>>());
    println!();

    println!("6. Rate of Change (ROC)");
    println!("   Momentum oscillator showing percentage price change");
    let roc_10 = roc(&close, 10)?;
    println!("   ROC(10) last value: {:?}", roc_10.f64()?.into_no_null_iter().last());
    println!();

    // Advanced: Fractional Differentiation
    println!("7. Fractional Differentiation");
    println!("   Advanced: Makes series stationary while preserving memory");
    println!("   Useful for ML applications in finance");
    let frac_diff = frac_diff_ffd(&close, 0.5, 10)?;
    println!("   Frac Diff (d=0.5, last 3): {:?}", &frac_diff.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!();

    // Building a Complete Analysis
    println!("8. Complete Analysis - Combining Multiple Indicators");
    let analysis_df = DataFrame::new(vec![
        close.clone().with_name("close"),
        volume.clone().with_name("volume"),
        obv_values.with_name("obv"),
        roc_10.with_name("roc_10"),
    ])?;
    
    println!("   Combined indicators DataFrame (last 3 rows):");
    let last_3 = analysis_df
        .lazy()
        .tail(3)
        .collect()?;
    println!("{:?}", last_3);
    println!();

    println!("=== Example Complete ===");
    Ok(())
}
