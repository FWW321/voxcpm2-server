use std::io::Cursor;
use std::path::PathBuf;

use anyhow::{Result, anyhow};
use base64::{Engine, prelude::BASE64_STANDARD};
use candle_core::{Device, Tensor};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use url::Url;

use crate::audio::resample::resample_simple;

fn load_audio_bytes_from_url(url: &str) -> Result<Vec<u8>> {
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            let response = reqwest::get(url).await?;
            if !response.status().is_success() {
                return Err(anyhow::anyhow!(
                    "Failed to download file: {}",
                    response.status()
                ));
            }
            let bytes = response.bytes().await?.to_vec();
            Ok(bytes)
        })
    })
}

fn get_audio_format_from_bytes(bytes: &[u8]) -> Result<String> {
    if bytes.len() < 12 {
        return Err(anyhow!("bytes too short: {}", bytes.len()));
    }
    if bytes.starts_with(&[0x52, 0x49, 0x46, 0x46])
        && bytes.len() >= 12
        && bytes[8..12] == [0x57, 0x41, 0x56, 0x45]
    {
        return Ok("wav".to_string());
    }
    if bytes.starts_with(&[0xFF, 0xFB])
        || bytes.starts_with(&[0xFF, 0xF3])
        || bytes.starts_with(&[0xFF, 0xF2])
    {
        return Ok("mp3".to_string());
    }
    if bytes.len() >= 3 && bytes[0..3] == [0x49, 0x44, 0x33] {
        return Ok("mp3".to_string());
    }
    Err(anyhow!("Unknown audio format"))
}

pub fn get_audio_bytes_vec(path_str: &str) -> Result<Vec<u8>> {
    if path_str.starts_with("http://") || path_str.starts_with("https://") {
        load_audio_bytes_from_url(path_str)
    } else if path_str.starts_with("file://") {
        let path = Url::parse(path_str)?;
        let path = path.to_file_path();
        let path = match path {
            Ok(p) => p,
            Err(_) => {
                let mut p = path_str.to_owned();
                p = p.split_off(7);
                PathBuf::from(p)
            }
        };
        let bytes = std::fs::read(path)?;
        Ok(bytes)
    } else if path_str.starts_with("data:audio") && path_str.contains("base64,") {
        let parts: Vec<&str> = path_str.splitn(2, "base64,").collect();
        let data = parts
            .get(1)
            .ok_or_else(|| anyhow!("invalid base64 audio data URI"))?;
        let decoded = BASE64_STANDARD.decode(data)?;
        Ok(decoded)
    } else {
        let path = PathBuf::from(path_str);
        if path.exists() {
            let bytes = std::fs::read(path)?;
            Ok(bytes)
        } else {
            Err(anyhow!("get audio path error: {}", path_str))
        }
    }
}

fn load_audio_use_symphonia(audio_vec: Vec<u8>, device: &Device) -> Result<(Tensor, usize)> {
    let extension = get_audio_format_from_bytes(&audio_vec)?;
    let content = Cursor::new(audio_vec);
    let mss = MediaSourceStream::new(Box::new(content), Default::default());
    let mut hint = Hint::new();
    hint.with_extension(&extension);
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or("No default track found")
        .map_err(|e| anyhow!("symphonia read err: {}", e))?;
    let mut channels = 1;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut all_samples: Vec<Vec<f32>> = Vec::new();
    while let Ok(packet) = format.next_packet() {
        match decoder.decode(&packet) {
            Ok(decoded) => match decoded {
                AudioBufferRef::F32(buf) => {
                    channels = buf.spec().channels.count();
                    for channel in 0..channels {
                        if all_samples.len() <= channel {
                            all_samples.push(Vec::new());
                        }
                        let channel_data = buf.chan(channel);
                        all_samples[channel].extend_from_slice(channel_data);
                    }
                }
                AudioBufferRef::S16(buf) => {
                    channels = buf.spec().channels.count();
                    for channel in 0..channels {
                        if all_samples.len() <= channel {
                            all_samples.push(Vec::new());
                        }
                        let channel_data = buf.chan(channel);
                        let float_samples: Vec<f32> =
                            channel_data.iter().map(|&s| s as f32 / 32768.0).collect();
                        all_samples[channel].extend(float_samples);
                    }
                }
                AudioBufferRef::S24(buf) => {
                    channels = buf.spec().channels.count();
                    for channel in 0..channels {
                        if all_samples.len() <= channel {
                            all_samples.push(Vec::new());
                        }
                        let channel_data = buf.chan(channel);
                        let float_samples: Vec<f32> = channel_data
                            .iter()
                            .map(|&s| s.inner() as f32 / 8388608.0)
                            .collect();
                        all_samples[channel].extend(float_samples);
                    }
                }
                _ => {}
            },
            Err(_) => break,
        }
    }
    let mut audio_tensor = Tensor::new(all_samples, device)?;
    if channels > 1 {
        audio_tensor = audio_tensor.mean_keepdim(0)?;
    }
    Ok((audio_tensor, sample_rate as usize))
}

pub fn load_audio_with_resample(
    path: &str,
    device: &Device,
    target_sample_rate: Option<usize>,
) -> Result<Tensor> {
    let audio_vec = get_audio_bytes_vec(path)?;
    let (mut audio, sr) = load_audio_use_symphonia(audio_vec, device)?;
    if let Some(target_sr) = target_sample_rate
        && target_sr != sr
    {
        audio = resample_simple(&audio, sr as i64, target_sr as i64)?;
    }
    Ok(audio)
}

pub fn get_audio_wav_u8(audio: &Tensor, sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let max = audio.abs()?.max_all()?.to_scalar::<f32>()?;
    let ratio = if max > 1.0 { 32767.0 / max } else { 32767.0 };
    let audio = audio.squeeze(0)?;
    let audio_vec = audio.to_vec1::<f32>()?;
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
    for i in audio_vec {
        let sample_i16 = (i * ratio).round() as i16;
        writer.write_sample(sample_i16)?;
    }
    writer.finalize()?;
    Ok(cursor.into_inner())
}
