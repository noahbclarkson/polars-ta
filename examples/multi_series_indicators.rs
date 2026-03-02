//! Multi-Series Technical Indicators Example
//!
//! This example demonstrates indicators that require multiple price series
//! (high, low, close, volume) such as ATR, Stochastic, VWAP, and ADX.

use polars::prelude::*;
use polars_ta::{atr_14, stochastic_14_3, vwap, adx_14, cci_20, williams_r_14};

fn main() -> anyhow::Result<()> {
    println!("=== Multi-Series Technical Indicators Example ===\n");

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

    // Calculate ATR (14-period)
    println!("1. Average True Range (ATR) - Volatility Indicator");
    let atr = atr_14(&high, &low, &close)?;
    println!("   ATR(14) last value: {:?}", atr.f64()?.into_no_null_iter().last());
    println!();

    // Calculate Stochastic Oscillator
    println!("2. Stochastic Oscillator (14/3)");
    let stoch = stochastic_14_3(&high, &low, &close)?;
    println!("   %K (last 3): {:?}", &stoch.k.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!("   %D (last 3): {:?}", &stoch.d.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!();

    // Calculate VWAP
    println!("3. Volume Weighted Average Price (VWAP)");
    let vwap_values = vwap(&high, &low, &close, &volume)?;
    println!("   VWAP (last 3): {:?}", &vwap_values.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!();

    // Calculate ADX
    println!("4. Average Directional Index (ADX) - Trend Strength");
    let adx = adx_14(&high, &low, &close)?;
    println!("   ADX (last 3): {:?}", &adx.adx.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!("   +DI (last 3): {:?}", &adx.plus_di.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!("   -DI (last 3): {:?}", &adx.minus_di.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!();

    // Calculate CCI
    println!("5. Commodity Channel Index (CCI)");
    let cci = cci_20(&high, &low, &close)?;
    println!("   CCI(20) last value: {:?}", cci.f64()?.into_no_null_iter().last());
    println!();

    // Calculate Williams %R
    println!("6. Williams %R (14-period)");
    let willr = williams_r_14(&high, &low, &close)?;
    println!("   Williams %R last value: {:?}", willr.f64()?.into_no_null_iter().last());
    println!();

    println!("=== Example Complete ===");
    Ok(())
}
