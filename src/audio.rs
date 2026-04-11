use std::f64::consts::PI;
use std::io::Cursor;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use base64::{Engine, prelude::BASE64_STANDARD};
use candle_core::{Device, Tensor};
use candle_nn::Module;
use hound::{SampleFormat, WavSpec, WavWriter};
use num::integer::gcd;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use url::Url;

pub fn encode_wav(audio: &Tensor, sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    assert_eq!(
        audio.dim(0)?,
        1,
        "audio must be mono (1 channel), got {} channels",
        audio.dim(0)?
    );

    let max = audio
        .abs()?
        .max_all()?
        .to_scalar::<f32>()
        .context("get max amplitude")?;
    let ratio = if max > 1.0 { 32767.0 / max } else { 32767.0 };

    let samples = audio.squeeze(0)?.to_vec1::<f32>()?;
    let mut cursor = Cursor::new(Vec::with_capacity(samples.len() * 2 + 44));
    let mut writer = WavWriter::new(&mut cursor, spec)?;
    for &s in &samples {
        writer.write_sample((s * ratio).round().clamp(-32768.0, 32767.0) as i16)?;
    }
    writer.finalize()?;
    Ok(cursor.into_inner())
}

pub fn decode(path: &str, device: &Device, target_sr: usize) -> Result<Tensor> {
    let bytes = if path.starts_with("http://") || path.starts_with("https://") {
        download_audio(path)?
    } else if path.starts_with("file://") {
        let file_path = Url::parse(path)
            .ok()
            .and_then(|u| u.to_file_path().ok())
            .unwrap_or_else(|| PathBuf::from(&path[7..]));
        std::fs::read(&file_path).context("read audio file")?
    } else if path.starts_with("data:audio") && path.contains("base64,") {
        let data = path
            .split_once("base64,")
            .map(|x| x.1)
            .ok_or_else(|| anyhow!("invalid base64 audio data URI"))?;
        BASE64_STANDARD.decode(data)?
    } else {
        let file_path = PathBuf::from(path);
        if file_path.exists() {
            std::fs::read(&file_path).context("read audio file")?
        } else {
            bail!("audio file not found: {}", path);
        }
    };
    let (audio, sr) = decode_symphonia(&bytes, device).context("symphonia decode")?;
    if sr == target_sr {
        Ok(audio)
    } else {
        resample_simple(&audio, sr as i64, target_sr as i64, device)
    }
}

fn download_audio(url: &str) -> Result<Vec<u8>> {
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            let resp = reqwest::get(url).await?;
            if !resp.status().is_success() {
                bail!("HTTP download failed: {}", resp.status());
            }
            Ok(resp.bytes().await?.to_vec())
        })
    })
}

fn decode_symphonia(bytes: &[u8], device: &Device) -> Result<(Tensor, usize)> {
    let cursor = std::io::Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut hint = Hint::new();
    if bytes.starts_with(b"RIFF") {
        hint.with_extension("wav");
    } else if bytes.starts_with(b"\xff\xfb")
        || bytes.starts_with(b"\xff\xf3")
        || bytes.starts_with(b"ID3")
    {
        hint.with_extension("mp3");
    } else if bytes.starts_with(b"fLaC") {
        hint.with_extension("flac");
    } else if bytes.starts_with(b"OggS") {
        hint.with_extension("ogg");
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("no audio track found"))?;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("unknown sample rate"))?;
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut all_samples: Vec<Vec<f32>> = Vec::new();
    let mut channels = 1usize;

    while let Ok(packet) = format.next_packet() {
        match decoder.decode(&packet) {
            Ok(decoded) => match decoded {
                AudioBufferRef::F32(buf) => {
                    channels = buf.spec().channels.count();
                    for ch in 0..channels {
                        if all_samples.len() <= ch {
                            all_samples.push(Vec::new());
                        }
                        all_samples[ch].extend_from_slice(buf.chan(ch));
                    }
                }
                AudioBufferRef::S16(buf) => {
                    channels = buf.spec().channels.count();
                    for ch in 0..channels {
                        if all_samples.len() <= ch {
                            all_samples.push(Vec::new());
                        }
                        all_samples[ch].extend(buf.chan(ch).iter().map(|&s| s as f32 / 32768.0));
                    }
                }
                AudioBufferRef::S24(buf) => {
                    channels = buf.spec().channels.count();
                    for ch in 0..channels {
                        if all_samples.len() <= ch {
                            all_samples.push(Vec::new());
                        }
                        all_samples[ch]
                            .extend(buf.chan(ch).iter().map(|s| s.inner() as f32 / 8388608.0));
                    }
                }
                AudioBufferRef::S32(buf) => {
                    channels = buf.spec().channels.count();
                    for ch in 0..channels {
                        if all_samples.len() <= ch {
                            all_samples.push(Vec::new());
                        }
                        all_samples[ch]
                            .extend(buf.chan(ch).iter().map(|&s| s as f32 / 2147483648.0));
                    }
                }
                AudioBufferRef::U8(buf) => {
                    channels = buf.spec().channels.count();
                    for ch in 0..channels {
                        if all_samples.len() <= ch {
                            all_samples.push(Vec::new());
                        }
                        all_samples[ch]
                            .extend(buf.chan(ch).iter().map(|&s| (s as f32 - 128.0) / 128.0));
                    }
                }
                _ => {
                    bail!("unsupported audio sample format");
                }
            },
            Err(_) => break,
        }
    }

    let mut audio = Tensor::new(all_samples, device)?;
    if channels > 1 {
        audio = audio.mean_keepdim(0)?;
    }
    Ok((audio, sample_rate as usize))
}

fn resample_simple(
    waveform: &Tensor,
    orig_freq: i64,
    new_freq: i64,
    device: &Device,
) -> Result<Tensor> {
    resample(waveform, orig_freq, new_freq, 6, 0.99, device)
}

fn resample(
    waveform: &Tensor,
    orig_freq: i64,
    new_freq: i64,
    lowpass_filter_width: i64,
    rolloff: f64,
    device: &Device,
) -> Result<Tensor> {
    if orig_freq == new_freq {
        return Ok(waveform.clone());
    }
    let gcd_val = gcd(orig_freq, new_freq);
    let (kernel, width) = get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd_val,
        lowpass_filter_width,
        rolloff,
        device,
    )?;
    apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd_val, &kernel, width)
}

fn get_sinc_resample_kernel(
    orig_freq: i64,
    new_freq: i64,
    gcd_val: i64,
    lowpass_filter_width: i64,
    rolloff: f64,
    device: &Device,
) -> Result<(Tensor, i64)> {
    let orig_freq = orig_freq / gcd_val;
    let new_freq = new_freq / gcd_val;
    let base_freq = (orig_freq.min(new_freq) as f64) * rolloff;
    let width = ((lowpass_filter_width as f64) * (orig_freq as f64) / base_freq).ceil() as i64;

    let idx = Tensor::arange(-width as f32, (width + orig_freq) as f32, device)?
        .affine(1.0 / orig_freq as f64, 0.0)?
        .unsqueeze(0)?
        .unsqueeze(0)?;

    let t = Tensor::arange_step(0.0f32, -(new_freq as f32), -1.0f32, device)?
        .affine(1.0 / new_freq as f64, 0.0)?
        .unsqueeze(candle_core::D::Minus1)?
        .unsqueeze(candle_core::D::Minus1)?
        .broadcast_add(&idx)?
        .affine(base_freq, 0.0)?;
    let t = t.clamp(-(lowpass_filter_width as f32), lowpass_filter_width as f32)?;

    let window_arg = t.affine(PI / (lowpass_filter_width as f64) / 2.0, 0.0)?;
    let window = window_arg.cos()?.sqr()?;

    let scale = base_freq / (orig_freq as f64);
    let t_scaled = t.affine(PI, 0.0)?;
    let t_zeros = Tensor::zeros_like(&t_scaled)?;
    let t_ones = Tensor::ones_like(&t_scaled)?;
    let mask = t_scaled.eq(&t_zeros)?;
    let sinc = mask.where_cond(&t_ones, &t_scaled.sin()?.div(&t_scaled)?)?;
    let kernels = sinc.mul(&window)?.affine(scale, 0.0)?;

    Ok((kernels, width))
}

fn apply_sinc_resample_kernel(
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

    let padded = waveform_flat.pad_with_zeros(
        candle_core::D::Minus1,
        width as usize,
        (width + orig_freq) as usize,
    )?;
    let waveform_3d = padded.unsqueeze(1)?;

    let conv = candle_nn::Conv1d::new(
        kernel.clone(),
        None,
        candle_nn::Conv1dConfig {
            padding: 0,
            stride: orig_freq as usize,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        },
    );
    let conv_output = conv.forward(&waveform_3d)?;
    let conv_transposed = conv_output.transpose(1, 2)?.reshape((num_wavs, ()))?;

    let target_length = ((new_freq as f64 * length as f64) / orig_freq as f64).ceil() as usize;
    let resampled = conv_transposed.narrow(1, 0, target_length.min(conv_transposed.dim(1)?))?;

    let resampled_dim = resampled.dim(1)?;
    let mut new_dims = dims.to_vec();
    if let Some(d) = new_dims.last_mut() {
        *d = resampled_dim;
    }
    Ok(resampled.reshape(new_dims)?)
}
