use std::f64::consts::PI;

use anyhow::{Result, anyhow};
use candle_core::{D, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module};
use num::integer::gcd;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum ResamplingMethod {
    SincInterpHann,
    SincInterpKaiser,
}

pub struct SincFilterConfig {
    pub lowpass_filter_width: i64,
    pub rolloff: f64,
    pub method: ResamplingMethod,
    pub beta: Option<f32>,
}

impl Default for SincFilterConfig {
    fn default() -> Self {
        Self {
            lowpass_filter_width: 6,
            rolloff: 0.99,
            method: ResamplingMethod::SincInterpHann,
            beta: None,
        }
    }
}

pub fn i0(x: f32) -> f32 {
    let mut result = 1.0;
    let mut term = 1.0;
    let half_x_sq = x * x / 4.0;
    for k in 1..100 {
        term = term * half_x_sq / (k * k) as f32;
        result += term;
        if term < 1e-12 {
            break;
        }
    }
    result
}

pub fn get_sinc_resample_kernel(
    orig_freq: i64,
    new_freq: i64,
    gcd_val: i64,
    filter: &SincFilterConfig,
    device: &Device,
) -> Result<(Tensor, i64)> {
    let lowpass_filter_width = filter.lowpass_filter_width;
    let rolloff = filter.rolloff;
    if orig_freq <= 0 || new_freq <= 0 {
        return Err(anyhow!("Frequencies must be positive"));
    }
    if lowpass_filter_width <= 0 {
        return Err(anyhow!("Low pass filter width should be positive"));
    }
    let orig_freq = orig_freq / gcd_val;
    let new_freq = new_freq / gcd_val;
    let base_freq = (orig_freq.min(new_freq) as f64) * rolloff;
    let width_f = (lowpass_filter_width as f64) * (orig_freq as f64) / base_freq;
    let width = width_f.ceil() as i64;
    let idx = Tensor::arange(-width as f32, (width + orig_freq) as f32, device)?
        .affine(1.0 / orig_freq as f64, 0.0)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    let t = Tensor::arange_step(0.0, -new_freq as f32, -1.0, device)?
        .affine(1.0 / new_freq as f64, 0.0)?
        .unsqueeze(D::Minus1)?
        .unsqueeze(D::Minus1)?
        .broadcast_add(&idx)?
        .affine(base_freq, 0.0)?;
    let t = t.clamp(-lowpass_filter_width as f32, lowpass_filter_width as f32)?;
    let window = match filter.method {
        ResamplingMethod::SincInterpHann => {
            let window_arg = t.affine(PI / (lowpass_filter_width as f64) / 2.0, 0.0)?;
            window_arg.cos()?.sqr()?
        }
        ResamplingMethod::SincInterpKaiser => {
            let beta_val = filter.beta.unwrap_or(14.769_656_f32);
            let i0_beta = i0(beta_val);
            let normalized_t = t.affine(1.0 / lowpass_filter_width as f64, 0.0)?;
            let arg = (1.0 - normalized_t.sqr()?)?;
            let sqrt_arg = arg.relu()?.sqrt()?;
            let sqrt_dims = sqrt_arg.dims();
            let sqrt_arg_vec = sqrt_arg.flatten_all()?.to_vec1::<f32>()?;
            let window_val: Vec<f32> = sqrt_arg_vec
                .iter()
                .map(|x| i0(beta_val * x) / i0_beta)
                .collect();
            Tensor::new(window_val, device)?.reshape(sqrt_dims)?
        }
    };
    let scale = base_freq / (orig_freq as f64);
    let t_scaled = t.affine(PI, 0.0)?;
    let t_zeros = Tensor::zeros_like(&t_scaled)?;
    let t_ones = Tensor::ones_like(&t_scaled)?;
    let mask = t_scaled.eq(&t_zeros)?;
    let sinc = mask.where_cond(&t_ones, &t_scaled.sin()?.div(&t_scaled)?)?;
    let kernels = sinc.mul(&window)?.affine(scale, 0.0)?;
    Ok((kernels, width))
}

pub fn apply_sinc_resample_kernel(
    waveform: &Tensor,
    orig_freq: i64,
    new_freq: i64,
    gcd_val: i64,
    kernel: &Tensor,
    width: i64,
) -> Result<Tensor> {
    let orig_freq = orig_freq / gcd_val;
    let new_freq = new_freq / gcd_val;
    let dims = waveform.dims();
    let waveform_flat = waveform.reshape(((), dims[dims.len() - 1]))?;
    let (num_wavs, length) = waveform_flat.dims2()?;
    let padded_waveform =
        waveform.pad_with_zeros(D::Minus1, width as usize, (width + orig_freq) as usize)?;
    let waveform_3d = padded_waveform.unsqueeze(1)?;
    let config = Conv1dConfig {
        padding: 0,
        stride: orig_freq as usize,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    };
    let conv1d = Conv1d::new(kernel.clone(), None, config);
    let conv_output = conv1d.forward(&waveform_3d)?;
    let conv_transposed = conv_output.transpose(1, 2)?.reshape((num_wavs, ()))?;
    let target_length = ((new_freq as f64 * length as f64) / orig_freq as f64).ceil() as usize;
    let resampled_flat =
        conv_transposed.narrow(1, 0, target_length.min(conv_transposed.dim(1)?))?;
    let mut new_dims = dims.to_vec();
    let last_dim = new_dims.len() - 1;
    new_dims[last_dim] = resampled_flat.dim(1)?;
    let resampled = resampled_flat.reshape(new_dims)?;
    Ok(resampled)
}

pub fn resample(
    waveform: &Tensor,
    orig_freq: i64,
    new_freq: i64,
    filter: &SincFilterConfig,
) -> Result<Tensor> {
    if orig_freq <= 0 || new_freq <= 0 {
        return Err(anyhow!("Frequencies must be positive"));
    }
    if orig_freq == new_freq {
        return Ok(waveform.clone());
    }
    let gcd_val = gcd(orig_freq, new_freq);
    let device = waveform.device();
    let (kernel, width) = get_sinc_resample_kernel(orig_freq, new_freq, gcd_val, filter, device)?;
    apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd_val, &kernel, width)
}

pub fn resample_simple(waveform: &Tensor, orig_freq: i64, new_freq: i64) -> Result<Tensor> {
    resample(waveform, orig_freq, new_freq, &SincFilterConfig::default())
}
