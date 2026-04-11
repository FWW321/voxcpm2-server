use std::io::Write;
use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use base64::{Engine, prelude::BASE64_STANDARD};
use candle_core::{DType, Device, Tensor};
use ffmpeg_next as ffmpeg;
use url::Url;

struct FormatDef {
    codec: &'static str,
    container: &'static str,
    content_type: &'static str,
}

static FORMATS: &[(&str, FormatDef)] = &[
    (
        "mp3",
        FormatDef {
            codec: "libmp3lame",
            container: "mp3",
            content_type: "audio/mpeg",
        },
    ),
    (
        "opus",
        FormatDef {
            codec: "libopus",
            container: "ogg",
            content_type: "audio/ogg; codecs=opus",
        },
    ),
    (
        "aac",
        FormatDef {
            codec: "aac",
            container: "adts",
            content_type: "audio/aac",
        },
    ),
    (
        "flac",
        FormatDef {
            codec: "flac",
            container: "flac",
            content_type: "audio/flac",
        },
    ),
    (
        "wav",
        FormatDef {
            codec: "pcm_s16le",
            container: "wav",
            content_type: "audio/wav",
        },
    ),
    (
        "pcm",
        FormatDef {
            codec: "pcm_s16le",
            container: "raw",
            content_type: "audio/pcm;rate=24000",
        },
    ),
];

fn find_format(name: &str) -> Result<&'static FormatDef> {
    FORMATS
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| v)
        .ok_or_else(|| {
            let supported: Vec<&str> = FORMATS.iter().map(|(k, _)| *k).collect();
            anyhow!(
                "Unsupported format '{}'. Supported: {}",
                name,
                supported.join(", ")
            )
        })
}

pub fn content_type(format: &str) -> Result<&'static str> {
    Ok(find_format(format)?.content_type)
}

pub fn decode(path: &str, device: &Device, target_sr: usize) -> Result<Tensor> {
    let bytes = resolve_audio_bytes(path)?;
    decode_bytes(&bytes, device, target_sr)
}

pub fn encode(
    tensor: &Tensor,
    sample_rate: u32,
    format: &str,
    speed: Option<f64>,
) -> Result<Vec<u8>> {
    let fmt = find_format(format)?;
    let samples = tensor.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let samples = match speed {
        Some(s) if (s - 1.0).abs() > f64::EPSILON => speed_adjust(&samples, sample_rate, s)?,
        _ => samples,
    };
    match fmt.container {
        "wav" => encode_wav_manual(&samples, sample_rate),
        "raw" => encode_pcm_raw(&samples),
        _ => encode_via_ffmpeg(&samples, sample_rate, fmt),
    }
}

fn speed_adjust(samples: &[f32], sample_rate: u32, speed: f64) -> Result<Vec<f32>> {
    if !(0.25..=4.0).contains(&speed) {
        bail!("speed must be between 0.25 and 4.0, got {}", speed);
    }

    let factors = build_atempo_chain(speed);

    let mut graph = ffmpeg::filter::Graph::new();

    let abuffer =
        ffmpeg::filter::find("abuffer").ok_or_else(|| anyhow!("abuffer filter not found"))?;
    let atempo_filt =
        ffmpeg::filter::find("atempo").ok_or_else(|| anyhow!("atempo filter not found"))?;
    let abuffersink = ffmpeg::filter::find("abuffersink")
        .ok_or_else(|| anyhow!("abuffersink filter not found"))?;

    let in_args = format!("sample_rate={sample_rate}:sample_fmt=flt:channel_layout=mono");
    graph.add(&abuffer, "in", &in_args)?;

    let atempo_names: Vec<String> = (0..factors.len()).map(|i| format!("atempo{i}")).collect();
    for (i, &factor) in factors.iter().enumerate() {
        graph.add(&atempo_filt, &atempo_names[i], &format!("{factor}"))?;
    }

    graph.add(&abuffersink, "out", "")?;

    {
        let mut in_ctx = graph.get("in").expect("in context");
        let mut first = graph.get(&atempo_names[0]).expect("atempo0");
        in_ctx.link(0, &mut first, 0);
    }
    for i in 1..factors.len() {
        let mut prev = graph.get(&atempo_names[i - 1]).expect("prev atempo");
        let mut cur = graph.get(&atempo_names[i]).expect("cur atempo");
        prev.link(0, &mut cur, 0);
    }
    {
        let mut last = graph
            .get(&atempo_names[factors.len() - 1])
            .expect("last atempo");
        let mut out_ctx = graph.get("out").expect("out context");
        last.link(0, &mut out_ctx, 0);
    }

    graph.validate()?;

    let chunk_size = 4096;
    let mut next_pts: i64 = 0;

    for chunk in samples.chunks(chunk_size) {
        let mut frame = ffmpeg::util::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            chunk.len(),
            ffmpeg::channel_layout::ChannelLayout::MONO,
        );
        frame.set_rate(sample_rate);
        frame.set_pts(Some(next_pts));
        let data = frame.plane_mut::<f32>(0);
        data.copy_from_slice(chunk);

        {
            let mut in_ctx = graph.get("in").expect("in context");
            in_ctx.source().add(&frame)?;
        }

        next_pts += chunk.len() as i64;
    }

    {
        let mut in_ctx = graph.get("in").expect("in context");
        in_ctx.source().flush()?;
    }

    let mut result = Vec::with_capacity(samples.len());
    loop {
        let mut out_frame = ffmpeg::util::frame::Audio::empty();
        let mut out_ctx = graph.get("out").expect("out context");
        match out_ctx.sink().frame(&mut out_frame) {
            Ok(()) => {
                result.extend_from_slice(out_frame.plane::<f32>(0));
            }
            Err(_) => break,
        }
    }

    Ok(result)
}

fn build_atempo_chain(speed: f64) -> Vec<f64> {
    let mut factors = Vec::new();
    let mut remaining = speed;
    while remaining > 2.0 {
        factors.push(2.0);
        remaining /= 2.0;
    }
    while remaining < 0.5 {
        factors.push(0.5);
        remaining /= 0.5;
    }
    factors.push(remaining.clamp(0.5, 2.0));
    factors
}

fn resolve_audio_bytes(path_str: &str) -> Result<Vec<u8>> {
    if path_str.starts_with("http://") || path_str.starts_with("https://") {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let resp = reqwest::get(path_str).await?;
                if !resp.status().is_success() {
                    bail!("HTTP download failed: {}", resp.status());
                }
                Ok(resp.bytes().await?.to_vec())
            })
        })
    } else if path_str.starts_with("file://") {
        let path = Url::parse(path_str)
            .ok()
            .and_then(|u| u.to_file_path().ok())
            .unwrap_or_else(|| PathBuf::from(&path_str[7..]));
        Ok(std::fs::read(path)?)
    } else if path_str.starts_with("data:audio") && path_str.contains("base64,") {
        let data = path_str
            .split_once("base64,")
            .map(|x| x.1)
            .ok_or_else(|| anyhow!("invalid base64 audio data URI"))?;
        Ok(BASE64_STANDARD.decode(data)?)
    } else {
        let path = PathBuf::from(path_str);
        if path.exists() {
            Ok(std::fs::read(path)?)
        } else {
            Err(anyhow!("audio file not found: {}", path_str))
        }
    }
}

fn decode_bytes(bytes: &[u8], device: &Device, target_sr: usize) -> Result<Tensor> {
    let tmp = tempfile::NamedTempFile::new()?;
    tmp.as_file().write_all(bytes)?;
    let path = tmp
        .path()
        .to_str()
        .ok_or_else(|| anyhow!("temp path not utf-8"))?;

    let mut ictx = ffmpeg::format::input(path)?;
    let input_stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .ok_or_else(|| anyhow!("no audio stream found"))?;
    let stream_idx = input_stream.index();

    let ctx = ffmpeg::codec::context::Context::from_parameters(input_stream.parameters())?;
    let mut decoder = ctx.decoder().audio()?;

    let out_format = ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed);
    let out_layout = ffmpeg::channel_layout::ChannelLayout::MONO;

    let mut resampler = ffmpeg::software::resampling::Context::get(
        decoder.format(),
        decoder.channel_layout(),
        decoder.rate(),
        out_format,
        out_layout,
        target_sr as u32,
    )?;

    let mut all_samples = Vec::new();

    for (stream, packet) in ictx.packets() {
        if stream.index() != stream_idx {
            continue;
        }
        decoder.send_packet(&packet)?;
        let mut frame = ffmpeg::util::frame::Audio::empty();
        while decoder.receive_frame(&mut frame).is_ok() {
            let mut resampled = ffmpeg::util::frame::Audio::empty();
            resampler.run(&frame, &mut resampled)?;
            let data = resampled.data(0);
            let f32_data: Vec<f32> = bytemuck::cast_slice(data).to_vec();
            all_samples.extend_from_slice(&f32_data);
        }
    }

    decoder.send_eof()?;
    let mut frame = ffmpeg::util::frame::Audio::empty();
    while decoder.receive_frame(&mut frame).is_ok() {
        let mut resampled = ffmpeg::util::frame::Audio::empty();
        resampler.run(&frame, &mut resampled)?;
        let data = resampled.data(0);
        let f32_data: Vec<f32> = bytemuck::cast_slice(data).to_vec();
        all_samples.extend_from_slice(&f32_data);
    }

    {
        let mut resampled = ffmpeg::util::frame::Audio::empty();
        while resampler.flush(&mut resampled).is_ok() && resampled.samples() > 0 {
            let data = resampled.data(0);
            let f32_data: Vec<f32> = bytemuck::cast_slice(data).to_vec();
            all_samples.extend_from_slice(&f32_data);
        }
    }

    drop(tmp);
    let len = all_samples.len();
    Ok(Tensor::from_vec(all_samples, (1, len), device)?)
}

fn drain_encoder(
    encoder: &mut ffmpeg::codec::encoder::Audio,
    octx: &mut ffmpeg::format::context::Output,
) -> Result<()> {
    let mut pkt = ffmpeg::Packet::empty();
    while encoder.receive_packet(&mut pkt).is_ok() {
        pkt.write_interleaved(octx)?;
    }
    Ok(())
}

fn send_audio_frame(
    f32_data: &[f32],
    format: ffmpeg::format::Sample,
    layout: ffmpeg::channel_layout::ChannelLayout,
    channels: usize,
    frame_size: usize,
    encoder: &mut ffmpeg::codec::encoder::Audio,
    octx: &mut ffmpeg::format::context::Output,
) -> Result<()> {
    for chunk in f32_data.chunks(frame_size * channels) {
        let chunk_samples = chunk.len() / channels;
        let mut frame = ffmpeg::util::frame::Audio::empty();
        // SAFETY: ffmpeg frame alloc - format/layout/chunk_samples are valid
        unsafe {
            frame.alloc(format, chunk_samples, layout);
        }
        let dst = frame.data_mut(0);
        let src_bytes = bytemuck::cast_slice::<f32, u8>(chunk);
        dst[..src_bytes.len()].copy_from_slice(src_bytes);
        encoder.send_frame(&frame)?;
        drain_encoder(encoder, octx)?;
    }
    Ok(())
}

fn encode_via_ffmpeg(samples: &[f32], sample_rate: u32, fmt: &FormatDef) -> Result<Vec<u8>> {
    let tmp_out = tempfile::NamedTempFile::new()?;

    let mut octx = ffmpeg::format::output(&tmp_out.path())?;
    let ocodec = ffmpeg::encoder::find_by_name(fmt.codec)
        .ok_or_else(|| anyhow!("codec '{}' not found", fmt.codec))?;
    let audio_codec = ocodec.audio()?;

    let has_global_header = octx
        .format()
        .flags()
        .contains(ffmpeg::format::flag::Flags::GLOBAL_HEADER);

    let mut ostream = octx.add_stream(audio_codec)?;
    let enc_ctx = ffmpeg::codec::context::Context::from_parameters(ostream.parameters())?;
    let mut encoder = enc_ctx.encoder().audio()?;

    let layout = ffmpeg::channel_layout::ChannelLayout::MONO;
    let enc_format =
        audio_codec
            .formats()
            .and_then(|mut f| f.next())
            .unwrap_or(ffmpeg::format::Sample::F32(
                ffmpeg::format::sample::Type::Packed,
            ));

    encoder.set_rate(sample_rate as i32);
    encoder.set_channel_layout(layout);
    encoder.set_format(enc_format);
    encoder.set_time_base((1, sample_rate as i32));
    ostream.set_time_base((1, sample_rate as i32));

    if has_global_header {
        encoder.set_flags(ffmpeg::codec::flag::Flags::GLOBAL_HEADER);
    }

    let mut encoder = encoder.open_as(audio_codec)?;
    ostream.set_parameters(&encoder);

    let frame_size = encoder.frame_size().max(1) as usize;

    octx.write_header()?;
    send_audio_frame(
        samples,
        enc_format,
        layout,
        1,
        frame_size,
        &mut encoder,
        &mut octx,
    )?;
    encoder.send_eof()?;
    drain_encoder(&mut encoder, &mut octx)?;
    octx.write_trailer()?;

    let result = std::fs::read(tmp_out.path())?;
    drop(tmp_out);
    Ok(result)
}

fn f32_to_i16_le(samples: &[f32]) -> Vec<u8> {
    let max_val = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let scale = if max_val > 1.0 {
        32767.0 / max_val
    } else {
        32767.0
    };
    samples
        .iter()
        .flat_map(|&s| {
            let v = (s * scale).round().clamp(-32768.0, 32767.0) as i16;
            v.to_le_bytes()
        })
        .collect()
}

fn encode_pcm_raw(samples: &[f32]) -> Result<Vec<u8>> {
    Ok(f32_to_i16_le(samples))
}

fn encode_wav_manual(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let pcm = f32_to_i16_le(samples);
    let data_size = pcm.len() as u32;
    let file_size = 36 + data_size;

    let mut buf = Vec::with_capacity(44 + data_size as usize);

    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * 2;
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());

    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    buf.extend_from_slice(&pcm);

    Ok(buf)
}
