//! Fractional Differentiation
//!
//! Fractional differentiation is used to make a time series stationary while
//! preserving as much memory (autocorrelation) as possible. This is particularly
//! useful in financial time series analysis.
//!
//! Based on Marcos Lopez de Prado's "Advances in Financial Machine Learning"

use anyhow::Result;
use polars::prelude::*;

/// Calculates weights for fractional differentiation
/// w_k = -w_{k-1} * (d - k + 1) / k
fn get_weights(d: f64, size: usize, threshold: f64) -> Vec<f64> {
    let mut weights = vec![1.0];
    let mut k = 1;

    loop {
        let w_prev = *weights.last().unwrap();
        let w_new = -w_prev * (d - k as f64 + 1.0) / k as f64;

        if k >= size || w_new.abs() < threshold {
            break;
        }

        weights.push(w_new);
        k += 1;
    }
    weights.into_iter().rev().collect()
}

/// Applies Fixed-Window Fractional Differentiation to a Series
///
/// This implements the FFD (Fixed-Width Fractional Differentiation) algorithm
/// from Advances in Financial Machine Learning by Marcos Lopez de Prado.
///
/// Fractional differentiation preserves more memory than integer differentiation
/// while still achieving stationarity, making it valuable for financial ML.
///
/// # Arguments
///
/// * `series` - Input price series
/// * `d` - Differentiation order (typically between 0 and 1)
/// * `window_size` - Maximum number of lagged observations to consider
///
/// # Returns
///
/// A new Series containing the fractionally differentiated values
///
/// # Example
///
/// ```rust,no_run
/// use polars::prelude::*;
/// use polars_ta::frac_diff_ffd;
///
/// let close = Series::new("close", &[100.0, 101.0, 102.0, 101.5, 103.0]);
/// let frac_diff = frac_diff_ffd(&close, 0.5, 10).unwrap();
/// ```
pub fn frac_diff_ffd(series: &Series, d: f64, window_size: usize) -> Result<Series> {
    let ca = series.f64()?;
    let data: Vec<f64> = (0..ca.len())
        .map(|i| ca.get(i).unwrap_or(f64::NAN))
        .collect();
    let weights = get_weights(d, window_size, 1e-5);
    let w_len = weights.len();

    let mut output = vec![f64::NAN; data.len()];

    for i in (w_len - 1)..data.len() {
        let mut val = 0.0;
        for (j, &w) in weights.iter().enumerate() {
            val += w * data[i - (w_len - 1) + j];
        }
        output[i] = val;
    }

    Ok(Series::new("frac_diff", output))
}
