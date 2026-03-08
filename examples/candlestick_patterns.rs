//! Candlestick Pattern Detection Example
//!
//! This example demonstrates how to detect common candlestick patterns
//! like Doji, Hammer, Engulfing, and Star patterns.

use polars::prelude::*;
use polars_ta::patterns::{doji, hammer, engulfing, star};

fn main() -> anyhow::Result<()> {
    println!("=== Candlestick Pattern Detection Example ===\n");

    // Create sample OHLC data with various patterns
    let open = Series::new(
        "open",
        &[
            100.0, // 0: Regular candle
            101.0, // 1: Doji (open ≈ close)
            102.0, // 2: Regular candle
            101.0, // 3: Bullish candle
            100.0, // 4: Bullish engulfing (engulfs previous bearish)
            105.0, // 5: Bearish candle
            106.0, // 6: Bearish engulfing (engulfs previous bullish)
            104.0, // 7: Small body (star middle)
            102.0, // 8: Bullish candle (morning star)
        ],
    );

    let high = Series::new(
        "high",
        &[
            102.0, 101.1, 103.0, 102.0, 103.0, 107.0, 106.5, 104.5, 103.0,
        ],
    );

    let low = Series::new(
        "low",
        &[
            99.0, 100.9, 101.0, 100.5, 99.0, 104.0, 102.0, 103.5, 101.0,
        ],
    );

    let close = Series::new(
        "close",
        &[
            101.0, 101.05, 101.5, 102.0, 103.0, 104.0, 102.0, 104.2, 103.5,
        ],
    );

    // Detect Doji patterns
    println!("1. Doji Pattern Detection");
    println!("   Doji occurs when open ≈ close (market indecision)");
    let doji_pattern = doji(&open, &close, Some(0.001))?;
    let doji_values = doji_pattern.bool()?;
    println!("   Pattern detected at indices: ");
    for i in 0..doji_values.len() {
        if doji_values.get(i).unwrap_or(false) {
            print!("{} ", i);
        }
    }
    println!("\n");

    // Detect Hammer patterns
    println!("2. Hammer Pattern Detection");
    println!("   Hammer: small body at top, long lower shadow (bullish reversal)");
    let hammer_pattern = hammer(&open, &high, &low, &close, None, None)?;
    let hammer_values = hammer_pattern.bool()?;
    println!("   Pattern detected at indices: ");
    for i in 0..hammer_values.len() {
        if hammer_values.get(i).unwrap_or(false) {
            print!("{} ", i);
        }
    }
    println!("\n");

    // Detect Engulfing patterns
    println!("3. Engulfing Pattern Detection");
    println!("   Bullish (+1): green candle engulfs previous red");
    println!("   Bearish (-1): red candle engulfs previous green");
    let engulfing_pattern = engulfing(&open, &close)?;
    let engulfing_values = engulfing_pattern.i32()?;
    println!("   Pattern detected:");
    for i in 0..engulfing_values.len() {
        match engulfing_values.get(i) {
            Some(1) => println!("     Index {}: Bullish Engulfing", i),
            Some(-1) => println!("     Index {}: Bearish Engulfing", i),
            _ => {}
        }
    }
    println!();

    // Detect Star patterns
    println!("4. Star Pattern Detection");
    println!("   Morning Star (+1): bullish reversal pattern");
    println!("   Evening Star (-1): bearish reversal pattern");
    let star_pattern = star(&open, &high, &low, &close, None)?;
    let star_values = star_pattern.i32()?;
    println!("   Pattern detected:");
    for i in 0..star_values.len() {
        match star_values.get(i) {
            Some(1) => println!("     Index {}: Morning Star", i),
            Some(-1) => println!("     Index {}: Evening Star", i),
            _ => {}
        }
    }
    println!();

    // Practical usage example
    println!("5. Practical Usage - Filtering Patterns");
    println!("   You can use these patterns in trading strategies:");
    println!("   Example: Find all bullish engulfing patterns");
    
    let df = DataFrame::new(vec![
        open.clone(),
        high.clone(),
        low.clone(),
        close.clone(),
        engulfing_pattern.with_name("engulfing"),
    ])?;
    
    let bullish_signals = df
        .lazy()
        .filter(col("engulfing").eq(lit(1)))
        .select([col("open"), col("close"), col("engulfing")])
        .collect()?;
    
    println!("   Bullish engulfing signals:");
    println!("{:?}", bullish_signals);
    println!();

    println!("=== Example Complete ===");
    Ok(())
}
