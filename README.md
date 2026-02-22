# polars-ta

[![Crates.io](https://img.shields.io/crates/v/polars-ta.svg)](https://crates.io/crates/polars-ta)
[![Docs.rs](https://docs.rs/polars-ta/badge.svg)](https://docs.rs/polars-ta)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-quality technical analysis for Polars DataFrames in Rust.**

A comprehensive library of technical indicators designed for algorithmic trading, quantitative finance, and time series analysis. Built on [Polars](https://pola.rs/) for blazing-fast performance with a clean, ergonomic API.

## Features

- 📈 **EMA** - Exponential Moving Average (Note: uses `α = 1/period`, not TA-Lib's `2/(period+1)`)
- 📊 **SMA** - Simple Moving Average (configurable period)
- 📉 **MACD** - Moving Average Convergence Divergence (12/26/9 standard or custom)
- 💪 **RSI** - Relative Strength Index (Wilder's smoothing, 14-period standard)
- 📏 **ATR** - Average True Range (volatility indicator)
- 🎯 **Bollinger Bands** - Volatility bands (20-period, 2 std dev standard)
- 📦 **OBV** - On-Balance Volume (momentum indicator)
- ⚡ **ROC** - Rate of Change (momentum oscillator)
- 🔬 **Frac Diff** - Fractional differentiation (unique feature for stationary time series)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
polars-ta = "0.1"
```

Or with `cargo add`:

```bash
cargo add polars-ta
```

## Quick Example

```rust
use polars::prelude::*;
use polars_ta::{rsi_14, macd_default, bollinger_20_2, ema, sma};

fn main() -> anyhow::Result<()> {
    // Create price data
    let close = Series::new("close", &[
        100.0, 101.5, 102.3, 101.8, 103.2,
        104.1, 103.5, 105.0, 106.2, 105.8,
        107.0, 106.5, 108.0, 109.2, 108.5,
        110.0, 109.8, 111.0, 110.5, 112.0,
    ]);
    
    // Calculate RSI (14-period standard)
    let rsi = rsi_14(&close)?;
    println!("RSI: {:?}", rsi);
    
    // Calculate MACD
    let macd = macd_default(&close)?;
    println!("MACD Line: {:?}", macd.macd);
    println!("Signal: {:?}", macd.signal);
    println!("Histogram: {:?}", macd.histogram);
    
    // Calculate Bollinger Bands
    let bands = bollinger_20_2(&close)?;
    println!("Upper Band: {:?}", bands.upper);
    println!("Middle Band: {:?}", bands.middle);
    println!("Lower Band: {:?}", bands.lower);
    
    // Calculate EMAs
    let ema_20 = ema(&close, 20)?;
    let ema_50 = ema(&close, 50)?;
    
    // Calculate SMAs
    let sma_20 = sma(&close, 20)?;
    
    Ok(())
}
```

## ATR Example (requires high/low/close)

```rust
use polars::prelude::*;
use polars_ta::atr_14;

let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);

let atr = atr_14(&high, &low, &close)?;
println!("ATR: {:?}", atr);
```

## OBV Example (requires close and volume)

```rust
use polars::prelude::*;
use polars_ta::obv;

let close = Series::new("close", &[100.0, 101.0, 100.5, 102.0, 101.5]);
let volume = Series::new("volume", &[1000.0, 500.0, 300.0, 400.0, 200.0]);

let obv_values = obv(&close, &volume)?;
println!("OBV: {:?}", obv_values);
```

## Fractional Differentiation

A unique feature of this library is fractional differentiation, which helps preserve memory (autocorrelation) in time series while achieving stationarity. This is particularly valuable for financial machine learning applications.

Based on Marcos Lopez de Prado's "Advances in Financial Machine Learning".

```rust
use polars::prelude::*;
use polars_ta::frac_diff_ffd;

let prices = Series::new("close", &[/* your price data */]);

// d=0.5 is a common choice, window_size depends on your data frequency
let stationary = frac_diff_ffd(&prices, 0.5, 20)?;
```

## API Overview

| Function | Description | Parameters |
|----------|-------------|------------|
| `ema(series, period)` | Exponential Moving Average | series, period |
| `sma(series, period)` | Simple Moving Average | series, period |
| `macd(series, fast, slow, signal)` | MACD with custom params | series, periods |
| `macd_default(series)` | MACD with 12/26/9 | series |
| `rsi(series, period)` | RSI with custom period | series, period |
| `rsi_14(series)` | RSI with 14-period | series |
| `atr(high, low, close, period)` | ATR with custom period | H/L/C, period |
| `atr_14(high, low, close)` | ATR with 14-period | H/L/C |
| `bollinger(series, period, std_dev)` | Bollinger Bands | series, params |
| `bollinger_20_2(series)` | Bollinger 20/2 | series |
| `obv(close, volume)` | On-Balance Volume | close, volume |
| `roc(series, period)` | Rate of Change | series, period |
| `frac_diff_ffd(series, d, window)` | Fractional differentiation | series, params |

## Design Principles

- **No panics** - All functions return `Result<T>` for proper error handling
- **Documented** - Every public function has doc comments with examples
- **Ergonomic** - Simple standalone functions that take `Series` and return `Series`
- **Fast** - Built on Polars' lazy evaluation where possible
- **Tested** - Comprehensive integration tests verify correctness

## Requirements

- Rust 2021 edition
- Polars 0.37

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure:
- All code has doc comments
- No panics in library code (use `Result`)
- Tests pass (`cargo test`)
- Code compiles without warnings (`cargo clippy`)

## Changelog

### 0.1.0
- Initial release with 9 core indicators
- EMA, SMA, MACD, RSI, ATR, Bollinger Bands, OBV, ROC, Frac Diff
- Full documentation and integration tests
