//! Basic Technical Indicators Example
//!
//! This example demonstrates how to use fundamental technical indicators
//! like RSI, MACD, and Bollinger Bands with Polars DataFrames.

use polars::prelude::*;
use polars_ta::{rsi_14, macd_default, bollinger_20_2, ema, sma};

fn main() -> anyhow::Result<()> {
    println!("=== Basic Technical Indicators Example ===\n");

    // Create sample price data (20 periods)
    let close = Series::new(
        "close",
        &[
            100.0, 101.5, 102.3, 101.8, 103.2, 104.1, 103.5, 105.0, 106.2, 105.8, 107.0, 106.5,
            108.0, 109.2, 108.5, 110.0, 109.8, 111.0, 110.5, 112.0,
        ],
    );

    // Calculate RSI (14-period standard)
    println!("1. RSI (14-period)");
    let rsi = rsi_14(&close)?;
    println!("   Last 5 RSI values: {:?}", &rsi.f64()?.into_no_null_iter().rev().take(5).collect::<Vec<_>>());
    println!();

    // Calculate MACD with default parameters (12/26/9)
    println!("2. MACD (12/26/9)");
    let macd = macd_default(&close)?;
    println!("   MACD Line (last 3): {:?}", &macd.macd.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!("   Signal Line (last 3): {:?}", &macd.signal.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!("   Histogram (last 3): {:?}", &macd.histogram.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!();

    // Calculate Bollinger Bands (20-period, 2 std dev)
    println!("3. Bollinger Bands (20-period, 2 std dev)");
    let bands = bollinger_20_2(&close)?;
    println!("   Upper Band (last 3): {:?}", &bands.upper.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!("   Middle Band (last 3): {:?}", &bands.middle.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!("   Lower Band (last 3): {:?}", &bands.lower.f64()?.into_no_null_iter().rev().take(3).collect::<Vec<_>>());
    println!();

    // Calculate different EMAs
    println!("4. Exponential Moving Averages");
    let ema_12 = ema(&close, 12)?;
    let ema_20 = ema(&close, 20)?;
    println!("   EMA(12) last value: {:?}", ema_12.f64()?.into_no_null_iter().last());
    println!("   EMA(20) last value: {:?}", ema_20.f64()?.into_no_null_iter().last());
    println!();

    // Calculate SMA
    println!("5. Simple Moving Average");
    let sma_20 = sma(&close, 20)?;
    println!("   SMA(20) last value: {:?}", sma_20.f64()?.into_no_null_iter().last());
    println!();

    println!("=== Example Complete ===");
    Ok(())
}
