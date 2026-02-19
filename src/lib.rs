//! # polars-ta
//!
//! Production-quality technical analysis indicators for Polars DataFrames in Rust.
//!
//! This library provides efficient, well-tested implementations of common technical
//! analysis indicators used in financial markets and algorithmic trading.
//!
//! ## Features
//!
//! - **EMA** - Exponential Moving Average
//! - **SMA** - Simple Moving Average
//! - **MACD** - Moving Average Convergence Divergence
//! - **RSI** - Relative Strength Index
//! - **ATR** - Average True Range
//! - **Bollinger Bands** - Volatility bands with configurable period and std dev
//! - **OBV** - On-Balance Volume
//! - **ROC** - Rate of Change
//! - **Frac Diff** - Fractional differentiation for stationary time series
//!
//! ## Example
//!
//! ```rust,no_run
//! use polars::prelude::*;
//! use polars_ta::{rsi_14, macd_default};
//!
//! // Create a simple price series
//! let close = Series::new("close", &[100.0, 101.5, 102.3, 101.8, 103.2]);
//!
//! // Calculate RSI
//! let rsi = rsi_14(&close).unwrap();
//!
//! // Calculate MACD
//! let macd_result = macd_default(&close).unwrap();
//! println!("MACD: {:?}", macd_result.macd);
//! println!("Signal: {:?}", macd_result.signal);
//! println!("Histogram: {:?}", macd_result.histogram);
//! ```

pub mod ema;
pub mod sma;
pub mod macd;
pub mod rsi;
pub mod atr;
pub mod bollinger;
pub mod obv;
pub mod roc;
pub mod frac_diff;

// Re-export main types and functions
pub use ema::ema;
pub use sma::sma;
pub use macd::{macd, macd_default, MacdResult};
pub use rsi::{rsi, rsi_14};
pub use atr::{atr, atr_14};
pub use bollinger::{bollinger, bollinger_20_2, BollingerResult};
pub use obv::obv;
pub use roc::roc;
pub use frac_diff::frac_diff_ffd;
