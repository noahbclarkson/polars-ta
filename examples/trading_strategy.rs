//! Trading Strategy Example
//!
//! This example demonstrates how to combine multiple technical indicators
//! to create a simple trading strategy with buy/sell signals.

use polars::prelude::*;
use polars_ta::{
    rsi_14, macd_default, ema, sma, bollinger_20_2, 
    atr_14, adx_14, stochastic_14_3,
};

fn main() -> anyhow::Result<()> {
    println!("=== Trading Strategy Example ===\n");

    // Create sample price data (50 periods for a realistic example)
    let close: Vec<f64> = (0..50)
        .map(|i| 100.0 + (i as f64 * 0.5) + ((i as f64 / 10.0).sin() * 5.0))
        .collect();
    
    let close = Series::new("close", &close);

    let high: Vec<f64> = close
        .f64()?
        .into_no_null_iter()
        .map(|c| c + 2.0)
        .collect();
    let high = Series::new("high", &high);

    let low: Vec<f64> = close
        .f64()?
        .into_no_null_iter()
        .map(|c| c - 2.0)
        .collect();
    let low = Series::new("low", &low);

    // Calculate multiple indicators
    println!("1. Calculating Technical Indicators...");
    let rsi = rsi_14(&close)?;
    let macd = macd_default(&close)?;
    let ema_20 = ema(&close, 20)?;
    let sma_50 = sma(&close, 50)?;
    let bb = bollinger_20_2(&close)?;
    let atr = atr_14(&high, &low, &close)?;
    let adx = adx_14(&high, &low, &close)?;
    let stoch = stochastic_14_3(&high, &low, &close)?;
    println!("   ✓ Indicators calculated\n");

    // Build analysis DataFrame
    println!("2. Building Analysis DataFrame...");
    let df = df![
        "close" => &close,
        "rsi" => &rsi,
        "macd" => &macd.macd,
        "macd_signal" => &macd.signal,
        "macd_histogram" => &macd.histogram,
        "ema_20" => &ema_20,
        "sma_50" => &sma_50,
        "bb_upper" => &bb.upper,
        "bb_lower" => &bb.lower,
        "atr" => &atr,
        "adx" => &adx.adx,
        "stoch_k" => &stoch.k,
        "stoch_d" => &stoch.d,
    ]?;
    println!("   ✓ DataFrame created\n");

    // Define trading rules
    println!("3. Defining Trading Rules...");
    println!("   Strategy: Multi-indicator confirmation\n");

    let signals = df
        .lazy()
        // RSI conditions
        .with_column(
            when(col("rsi").lt(lit(30.0)))
                .then(lit(1))   // Oversold - potential buy
                .when(col("rsi").gt(lit(70.0)))
                .then(lit(-1))  // Overbought - potential sell
                .otherwise(lit(0))
                .alias("rsi_signal"),
        )
        // MACD conditions
        .with_column(
            when(col("macd").gt(col("macd_signal")))
                .then(lit(1))   // Bullish
                .otherwise(lit(-1))  // Bearish
                .alias("macd_signal_type"),
        )
        // Trend conditions (EMA vs SMA)
        .with_column(
            when(col("ema_20").gt(col("sma_50")))
                .then(lit(1))   // Uptrend
                .otherwise(lit(-1))  // Downtrend
                .alias("trend"),
        )
        // Bollinger Band position
        .with_column(
            when(col("close").lt(col("bb_lower")))
                .then(lit(1))   // Below lower band - potential buy
                .when(col("close").gt(col("bb_upper")))
                .then(lit(-1))  // Above upper band - potential sell
                .otherwise(lit(0))
                .alias("bb_signal"),
        )
        // ADX trend strength
        .with_column(
            when(col("adx").gt(lit(25.0)))
                .then(lit(1))   // Strong trend
                .otherwise(lit(0))
                .alias("trend_strength"),
        )
        // Stochastic conditions
        .with_column(
            when(col("stoch_k").lt(lit(20.0)))
                .then(lit(1))   // Oversold
                .when(col("stoch_k").gt(lit(80.0)))
                .then(lit(-1))  // Overbought
                .otherwise(lit(0))
                .alias("stoch_signal"),
        )
        // Combined buy signal (multiple confirmations)
        .with_column(
            when(
                col("rsi_signal").eq(lit(1))
                    .and(col("macd_signal_type").eq(lit(1)))
                    .and(col("trend").eq(lit(1)))
                    .and(col("trend_strength").eq(lit(1)))
            )
            .then(lit(1))
            .when(
                col("rsi_signal").eq(lit(-1))
                    .and(col("macd_signal_type").eq(lit(-1)))
                    .and(col("trend").eq(lit(-1)))
            )
            .then(lit(-1))
            .otherwise(lit(0))
            .alias("combined_signal"),
        )
        .collect()?;

    // Display results
    println!("4. Analysis Results (last 5 periods):\n");
    let display_cols = vec![
        col("close"),
        col("rsi"),
        col("macd_histogram"),
        col("adx"),
        col("stoch_k"),
        col("combined_signal"),
    ];

    let last_5 = signals
        .clone()
        .lazy()
        .select(display_cols)
        .tail(5)
        .collect()?;

    println!("{:?}\n", last_5);

    // Count signals
    println!("5. Signal Summary:");
    let buy_count = signals
        .clone()
        .lazy()
        .filter(col("combined_signal").eq(lit(1)))
        .select([len().alias("count")])
        .collect()?;
    
    let sell_count = signals
        .lazy()
        .filter(col("combined_signal").eq(lit(-1)))
        .select([len().alias("count")])
        .collect()?;

    let buy_signals = buy_count.column("count")?.idx()?.get(0).unwrap_or(0);
    let sell_signals = sell_count.column("count")?.idx()?.get(0).unwrap_or(0);

    println!("   Buy signals generated: {}", buy_signals);
    println!("   Sell signals generated: {}\n", sell_signals);

    // Risk management example
    println!("6. Risk Management with ATR:");
    println!("   Using ATR for stop-loss calculation:");
    
    let last_atr = atr.f64()?.into_no_null_iter().last().unwrap_or(0.0);
    let last_close = close.f64()?.into_no_null_iter().last().unwrap_or(0.0);
    
    println!("   Current Price: {:.2}", last_close);
    println!("   Current ATR: {:.2}", last_atr);
    println!("   Suggested Stop-Loss (2x ATR): {:.2}", last_close - (2.0 * last_atr));
    println!("   Suggested Take-Profit (3x ATR): {:.2}\n", last_close + (3.0 * last_atr));

    println!("=== Example Complete ===");
    println!("\nNote: This is a demonstration strategy. Always backtest thoroughly");
    println!("and consider transaction costs, slippage, and market conditions.");

    Ok(())
}
