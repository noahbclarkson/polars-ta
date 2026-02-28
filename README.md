# polars-ta

[![Crates.io](https://img.shields.io/crates/v/polars-ta.svg)](https://crates.io/crates/polars-ta)
[![Docs.rs](https://docs.rs/polars-ta/badge.svg)](https://docs.rs/polars-ta)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-quality technical analysis for Polars DataFrames in Rust.**

A comprehensive library of technical indicators designed for algorithmic trading, quantitative finance, and time series analysis. Built on [Polars](https://pola.rs/) for blazing-fast performance with a clean, ergonomic API.

## Features

### Trend Indicators
- 📈 **EMA** - Exponential Moving Average (Note: uses `α = 1/period`, not TA-Lib's `2/(period+1)`)
- 📊 **SMA** - Simple Moving Average (configurable period)
- 📉 **MACD** - Moving Average Convergence Divergence (12/26/9 standard or custom)

### Momentum Indicators
- 💪 **RSI** - Relative Strength Index (Wilder's smoothing, 14-period standard)
- ⚡ **ROC** - Rate of Change (momentum oscillator)
- 🔄 **Stochastic** - Stochastic Oscillator (%K and %D lines, 14/3 standard)
- 📊 **CCI** - Commodity Channel Index (20-period standard)
- 📉 **Williams %R** - Williams Percent Range (14-period standard)
- 📈 **TRIX** - Triple Smoothed EMA Rate of Change
- 🎯 **TSI** - True Strength Index (double-smoothed momentum)

### Volatility Indicators
- 📏 **ATR** - Average True Range (14-period standard)
- 🎯 **Bollinger Bands** - Volatility bands (20-period, 2 std dev standard)

### Trend Strength
- 📐 **ADX** - Average Directional Index (14-period standard, includes +DI/-DI)

### Volume Indicators
- 📦 **OBV** - On-Balance Volume (momentum indicator)
- 💰 **VWAP** - Volume Weighted Average Price (cumulative and rolling)
- 💵 **MFI** - Money Flow Index (volume-weighted RSI, 14-period standard)
- 💠 **CMF** - Chaikin Money Flow (20-period standard)

### Special
- 🔬 **Frac Diff** - Fractional differentiation (unique feature for stationary time series)
- 🎯 **PSAR** - Parabolic SAR (trend-following stop and reverse)

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

## Stochastic Oscillator Example

```rust
use polars::prelude::*;
use polars_ta::stochastic_14_3;

let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);

let stoch = stochastic_14_3(&high, &low, &close)?;
println!("%K: {:?}", stoch.k);
println!("%D: {:?}", stoch.d);
```

## VWAP Example

```rust
use polars::prelude::*;
use polars_ta::vwap;

let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
let volume = Series::new("volume", &[1000.0, 1200.0, 800.0, 1500.0, 900.0]);

let vwap_values = vwap(&high, &low, &close, &volume)?;
println!("VWAP: {:?}", vwap_values);
```

## ADX Example

```rust
use polars::prelude::*;
use polars_ta::adx_14;

let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);

let adx = adx_14(&high, &low, &close)?;
println!("ADX: {:?}", adx.adx);
println!("+DI: {:?}", adx.plus_di);
println!("-DI: {:?}", adx.minus_di);
```

## MFI Example

```rust
use polars::prelude::*;
use polars_ta::mfi;

let high = Series::new("high", &[105.0, 106.0, 107.0, 108.0, 107.5]);
let low = Series::new("low", &[100.0, 101.0, 102.0, 103.0, 102.5]);
let close = Series::new("close", &[103.0, 104.0, 105.0, 106.0, 105.5]);
let volume = Series::new("volume", &[1000.0, 1200.0, 900.0, 1500.0, 800.0]);

let df = DataFrame::new(vec![high, low, close, volume])?;
let mfi_values = mfi(&df, 14)?;
println!("MFI: {:?}", mfi_values);
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

### Moving Averages

| Function | Description | Parameters |
|----------|-------------|------------|
| `ema(series, period)` | Exponential Moving Average | series, period |
| `sma(series, period)` | Simple Moving Average | series, period |
| `ema_12(series)` | EMA with period 12 | series |
| `ema_20(series)` | EMA with period 20 | series |
| `ema_50(series)` | EMA with period 50 | series |
| `ema_200(series)` | EMA with period 200 | series |

### Momentum Indicators

| Function | Description | Parameters |
|----------|-------------|------------|
| `macd(series, fast, slow, signal)` | MACD with custom params | series, periods |
| `macd_default(series)` | MACD with 12/26/9 | series |
| `rsi(series, period)` | RSI with custom period | series, period |
| `rsi_14(series)` | RSI with 14-period | series |
| `roc(series, period)` | Rate of Change | series, period |
| `stochastic(h, l, c, k_period, d_period)` | Stochastic Oscillator | H/L/C, periods |
| `stochastic_14_3(h, l, c)` | Stochastic 14/3 | H/L/C |
| `cci(h, l, c, period)` | CCI with custom period | H/L/C, period |
| `cci_20(h, l, c)` | CCI 20-period | H/L/C |
| `williams_r(h, l, c, period)` | Williams %R custom period | H/L/C, period |
| `williams_r_14(h, l, c)` | Williams %R 14-period | H/L/C |
| `trix(df, period)` | TRIX indicator | DataFrame with close |
| `tsi(df, slow, fast)` | True Strength Index | DataFrame with close |

### Volatility & Trend Strength

| Function | Description | Parameters |
|----------|-------------|------------|
| `atr(h, l, c, period)` | ATR with custom period | H/L/C, period |
| `atr_14(h, l, c)` | ATR with 14-period | H/L/C |
| `true_range(h, l, c)` | True Range | H/L/C |
| `bollinger(series, period, std_dev)` | Bollinger Bands | series, params |
| `bollinger_20_2(series)` | Bollinger 20/2 | series |
| `adx(h, l, c, period)` | ADX with custom period | H/L/C, period |
| `adx_14(h, l, c)` | ADX 14-period | H/L/C |
| `psar(h, l, c, af_step, af_max)` | Parabolic SAR | H/L/C, params |
| `psar_default(h, l, c)` | PSAR with defaults | H/L/C |

### Volume Indicators

| Function | Description | Parameters |
|----------|-------------|------------|
| `obv(close, volume)` | On-Balance Volume | close, volume |
| `vwap(h, l, c, volume)` | VWAP (cumulative) | H/L/C, volume |
| `rolling_vwap(h, l, c, volume, period)` | Rolling VWAP | H/L/C, volume, period |
| `mfi(df, period)` | Money Flow Index | DataFrame with H/L/C/V |
| `cmf(h, l, c, volume, period)` | Chaikin Money Flow | H/L/C/V, period |
| `cmf_20(h, l, c, volume)` | CMF 20-period | H/L/C/V |

### Special

| Function | Description | Parameters |
|----------|-------------|------------|
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
- Initial release with 20+ technical indicators
- **Trend**: EMA, SMA, MACD
- **Momentum**: RSI, ROC, Stochastic, CCI, Williams %R, TRIX, TSI
- **Volatility**: ATR, Bollinger Bands, True Range
- **Trend Strength**: ADX (+DI/-DI)
- **Volume**: OBV, VWAP, MFI, CMF
- **Special**: Frac Diff, Parabolic SAR
- Full documentation and integration tests
