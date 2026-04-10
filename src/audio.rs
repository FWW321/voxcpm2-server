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
        "wav",
        FormatDef {
            codec: "pcm_s16le",
            container: "wav",
            content_type: "audio/wav",
        },
    ),
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
            content_type: "audio/opus",
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

pub fn encode(tensor: &Tensor, sample_rate: u32, format: &str) -> Result<Vec<u8>> {
    let fmt = find_format(format)?;
    let samples = tensor.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    if fmt.container == "wav" {
        return encode_wav_manual(&samples, sample_rate);
    }
    encode_via_ffmpeg(&samples, sample_rate, fmt)
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

    ffmpeg::init()?;
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
    let tmp_wav = tempfile::NamedTempFile::new()?;
    {
        let data = encode_wav_manual(samples, sample_rate)?;
        std::fs::write(tmp_wav.path(), &data)?;
    }

    let tmp_out = tempfile::NamedTempFile::new()?;

    ffmpeg::init()?;
    let mut ictx = ffmpeg::format::input(&tmp_wav.path())?;
    let in_stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .ok_or_else(|| anyhow!("no audio stream in temp wav"))?;
    let in_idx = in_stream.index();
    let in_ctx = ffmpeg::codec::context::Context::from_parameters(in_stream.parameters())?;
    let mut decoder = in_ctx.decoder().audio()?;

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
    let best_format =
        audio_codec
            .formats()
            .and_then(|mut f| f.next())
            .unwrap_or(ffmpeg::format::Sample::F32(
                ffmpeg::format::sample::Type::Packed,
            ));

    encoder.set_rate(sample_rate as i32);
    encoder.set_channel_layout(layout);
    encoder.set_format(best_format);
    encoder.set_time_base((1, sample_rate as i32));
    ostream.set_time_base((1, sample_rate as i32));

    if has_global_header {
        encoder.set_flags(ffmpeg::codec::flag::Flags::GLOBAL_HEADER);
    }

    let mut encoder = encoder.open_as(audio_codec)?;
    ostream.set_parameters(&encoder);

    let mut resampler = ffmpeg::software::resampling::Context::get(
        decoder.format(),
        decoder.channel_layout(),
        decoder.rate(),
        best_format,
        layout,
        sample_rate,
    )?;

    let frame_size = encoder.frame_size().max(1) as usize;

    octx.write_header()?;

    for (stream, packet) in ictx.packets() {
        if stream.index() != in_idx {
            continue;
        }
        decoder.send_packet(&packet)?;
        let mut dframe = ffmpeg::util::frame::Audio::empty();
        while decoder.receive_frame(&mut dframe).is_ok() {
            let mut rframe = ffmpeg::util::frame::Audio::empty();
            resampler.run(&dframe, &mut rframe)?;
            let data = rframe.data(0);
            let f32_data: Vec<f32> = bytemuck::cast_slice(data).to_vec();
            let channels = rframe.channels() as usize;
            let rfmt = rframe.format();
            send_audio_frame(
                &f32_data,
                rfmt,
                layout,
                channels,
                frame_size,
                &mut encoder,
                &mut octx,
            )?;
        }
    }

    decoder.send_eof()?;
    let mut dframe = ffmpeg::util::frame::Audio::empty();
    while decoder.receive_frame(&mut dframe).is_ok() {
        let mut rframe = ffmpeg::util::frame::Audio::empty();
        resampler.run(&dframe, &mut rframe)?;
        let data = rframe.data(0);
        let f32_data: Vec<f32> = bytemuck::cast_slice(data).to_vec();
        let channels = rframe.channels() as usize;
        let rfmt = rframe.format();
        send_audio_frame(
            &f32_data,
            rfmt,
            layout,
            channels,
            frame_size,
            &mut encoder,
            &mut octx,
        )?;
    }

    {
        let mut rframe = ffmpeg::util::frame::Audio::empty();
        while resampler.flush(&mut rframe).is_ok() && rframe.samples() > 0 {
            let data = rframe.data(0);
            let f32_data: Vec<f32> = bytemuck::cast_slice(data).to_vec();
            let channels = rframe.channels() as usize;
            let rfmt = rframe.format();
            send_audio_frame(
                &f32_data,
                rfmt,
                layout,
                channels,
                frame_size,
                &mut encoder,
                &mut octx,
            )?;
        }
    }

    encoder.send_eof()?;
    drain_encoder(&mut encoder, &mut octx)?;
    octx.write_trailer()?;

    drop(tmp_wav);
    let result = std::fs::read(tmp_out.path())?;
    drop(tmp_out);
    Ok(result)
}

fn encode_wav_manual(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let num_samples = samples.len();
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    let mut buf = Vec::with_capacity(44 + data_size);

    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(file_size as u32).to_le_bytes());
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
    buf.extend_from_slice(&(data_size as u32).to_le_bytes());

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

    for &s in samples {
        let v = (s * scale).round().clamp(-32768.0, 32767.0) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }

    Ok(buf)
}
