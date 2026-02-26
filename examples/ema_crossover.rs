use polars::prelude::*;
use polars_ta::{ema_20, ema_50};
use anyhow::Result;

/// Simulates a basic EMA crossover trading strategy.
/// 
/// This example demonstrates how to use `polars-ta` to calculate technical indicators
/// and build trading signals using the Polars DataFrame API, similar to what
/// algorithmic trading bots (like `krypto`) might do.
fn main() -> Result<()> {
    // 1. Create a dummy dataset representing daily closing prices
    // In reality, this would be loaded from a CSV, database, or API.
    let prices = &[
        100.0, 102.0, 101.5, 103.0, 105.0, 107.0, 108.0, 106.0, 105.5, 104.0, // Uptrend
        103.0, 102.0, 100.0, 98.0,  97.0,  95.0,  96.0,  98.0,  100.0, 102.0, // Downtrend & reversal
        104.0, 106.0, 108.0, 110.0, 112.0, 115.0, 114.0, 113.0, 116.0, 118.0, // Strong uptrend
        120.0, 119.0, 121.0, 122.0, 124.0, 123.0, 125.0, 127.0, 126.0, 128.0, // Continued uptrend
        127.0, 125.0, 123.0, 121.0, 119.0, 118.0, 116.0, 115.0, 114.0, 112.0, // Downtrend
        110.0, 108.0, 106.0, 105.0, 107.0, 109.0, 111.0, 113.0, 115.0, 117.0, // Reversal
    ];
    
    // We need at least 50 points for the EMA_50 to have some data
    let close = Series::new("close", prices);
    
    // 2. Calculate the technical indicators
    let fast_ema = ema_20(&close)?;
    let slow_ema = ema_50(&close)?;
    
    // Create new names so they are distinct in the DataFrame
    let mut fast_ema = fast_ema.clone();
    fast_ema.rename("ema_fast");
    
    let mut slow_ema = slow_ema.clone();
    slow_ema.rename("ema_slow");
    
    // 3. Combine into a DataFrame for easier manipulation
    let df = DataFrame::new(vec![
        close.clone(),
        fast_ema,
        slow_ema,
    ])?;
    
    println!("Initial DataFrame (tail):\n{}", df.tail(Some(5)));

    // 4. Generate Trading Signals (Fast EMA crosses Slow EMA)
    // We use the Lazy API for expressive column operations
    let signal_df = df
        .lazy()
        // Filter out rows where indicators aren't calculated yet
        .filter(col("ema_slow").is_not_null().and(col("ema_fast").is_not_null()))
        // Calculate the difference between fast and slow
        .with_column(
            (col("ema_fast") - col("ema_slow")).alias("diff")
        )
        // A crossover occurs when the difference changes sign from the previous day.
        // We calculate if the current difference is positive.
        .with_column(
            (col("diff").gt(lit(0.0))).alias("is_fast_above_slow")
        )
        // Check if the previous day was positive
        .with_column(
            col("is_fast_above_slow").shift(lit(1)).alias("prev_fast_above_slow")
        )
        // Generate buy/sell signals based on crossovers
        .with_columns([
            // Buy signal: Fast crosses ABOVE Slow (currently above, previously below or equal)
            (col("is_fast_above_slow").and(col("prev_fast_above_slow").not())).alias("buy_signal"),
            // Sell signal: Fast crosses BELOW Slow (currently below, previously above)
            (col("is_fast_above_slow").not().and(col("prev_fast_above_slow"))).alias("sell_signal"),
        ])
        .collect()?;

    println!("\nDataFrame with Signals (tail):\n{}", signal_df.tail(Some(10)));
    
    // 5. Extract just the rows where a signal occurred
    let only_signals = signal_df
        .lazy()
        .filter(col("buy_signal").or(col("sell_signal")))
        .select([
            col("close"),
            col("ema_fast"),
            col("ema_slow"),
            col("buy_signal"),
            col("sell_signal")
        ])
        .collect()?;
        
    println!("\nTrading Crossover Signals found:\n{}", only_signals);

    Ok(())
}