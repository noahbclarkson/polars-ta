//! # polars-ta
//!
//! Production-quality technical analysis indicators for Polars DataFrames in Rust.
//!
//! This library provides efficient, well-tested implementations of common technical
//! analysis indicators used in financial markets and algorithmic trading.
//!
//! ## Features
//!
//! **Trend Indicators:**
//! - **EMA** - Exponential Moving Average
//! - **SMA** - Simple Moving Average
//! - **MACD** - Moving Average Convergence Divergence
//!
//! **Momentum Indicators:**
//! - **RSI** - Relative Strength Index
//! - **ROC** - Rate of Change
//! - **Stochastic** - Stochastic Oscillator
//!
//! **Volatility Indicators:**
//! - **ATR** - Average True Range
//! - **Bollinger Bands** - Volatility bands
//!
//! **Volume Indicators:**
//! - **OBV** - On-Balance Volume
//! - **VWAP** - Volume Weighted Average Price
//!
//! **Trend Strength:**
//! - **ADX** - Average Directional Index
//!
//! **Special:**
//! - **Frac Diff** - Fractional differentiation for stationary time series
//!
//! ## Example
//!
//! ```rust,no_run
//! use polars::prelude::*;
//! use polars_ta::{rsi_14, macd_default, stochastic_14_3, vwap, adx_14};
//!
//! // Create price series
//! let high = Series::new("high", &[105.0, 106.0, 107.0, 106.5, 108.0]);
//! let low = Series::new("low", &[100.0, 101.0, 102.0, 101.5, 103.0]);
//! let close = Series::new("close", &[103.0, 104.0, 105.0, 104.5, 106.0]);
//! let volume = Series::new("volume", &[1000.0, 1200.0, 800.0, 1500.0, 900.0]);
//!
//! // Calculate RSI
//! let rsi = rsi_14(&close).unwrap();
//!
//! // Calculate MACD
//! let macd_result = macd_default(&close).unwrap();
//! println!("MACD: {:?}", macd_result.macd);
//!
//! // Calculate Stochastic
//! let stoch = stochastic_14_3(&high, &low, &close).unwrap();
//! println!("%K: {:?}", stoch.k);
//!
//! // Calculate VWAP
//! let vwap_val = vwap(&high, &low, &close, &volume).unwrap();
//!
//! // Calculate ADX
//! let adx = adx_14(&high, &low, &close).unwrap();
//! println!("ADX: {:?}", adx.adx);
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
pub mod stochastic;
pub mod vwap;
pub mod adx;
pub mod cci;
pub mod williams_r;
pub mod cmf;
pub mod psar;
pub mod indicators;

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
pub use stochastic::{stochastic, stochastic_14_3, StochasticResult};
pub use vwap::{vwap, rolling_vwap};
pub use adx::{adx, adx_14, AdxResult};
pub use cci::{cci, cci_20};
pub use williams_r::{williams_r, williams_r_14};
pub use cmf::{cmf, cmf_20};
pub use psar::{psar, psar_default};

// Re-export from indicators module (extracted from krypto)
pub use indicators::{
    ema_12, ema_20, ema_50, ema_200,
    macd as macd_indicators, rsi_14 as rsi_14_indicators, atr_14 as atr_14_indicators, true_range,
    MacdResult as MacdResultIndicators,
};
