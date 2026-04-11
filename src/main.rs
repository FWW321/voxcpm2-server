mod audio;
mod model;
mod nn;
mod utils;

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use clap::{Parser, Subcommand};
use tracing::{Level, info};

use crate::model::generate::{GenerateRequest, VoxCPM2Engine};

const MODEL_REPO: &str = "openbmb/VoxCPM2";
const HF_BASE: &str = "https://huggingface.co";

const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "model.safetensors",
    "audiovae.pth",
    "tokenizer.json",
];

#[derive(Parser)]
#[command(name = "voxcpm2", about = "VoxCPM2 TTS CLI")]
struct Cli {
    #[arg(long, help = "Path to VoxCPM2 model directory")]
    model: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Generate {
        #[arg(short, long, help = "Text to synthesize")]
        text: String,

        #[arg(short, long, help = "Output WAV file path")]
        output: String,

        #[arg(long, help = "Voice name (must be registered first)")]
        voice: Option<String>,

        #[arg(long, help = "Prompt text for cloning")]
        prompt_text: Option<String>,

        #[arg(long, help = "Prompt audio file path")]
        prompt_wav: Option<String>,

        #[arg(long, help = "Reference audio file path (controllable cloning)")]
        reference_wav: Option<String>,

        #[arg(long, help = "Control instruction, e.g. '(gentle female voice)'")]
        control_instruction: Option<String>,

        #[arg(long, default_value_t = 10, help = "Inference timesteps")]
        inference_timesteps: usize,

        #[arg(long, default_value_t = 2.0, help = "CFG value")]
        cfg_value: f64,

        #[arg(long, default_value_t = 2, help = "Min generation length")]
        min_len: usize,

        #[arg(long, default_value_t = 4096, help = "Max generation length")]
        max_len: usize,
    },

    RegisterVoice {
        #[arg(long, help = "Voice name")]
        name: String,

        #[arg(long, help = "Prompt text (transcript of prompt audio)")]
        prompt_text: Option<String>,

        #[arg(long, help = "Prompt audio file path")]
        prompt_wav: Option<String>,

        #[arg(long, help = "Reference audio file path")]
        reference_wav: Option<String>,
    },

    ListVoices,
}

fn default_model_dir() -> Result<PathBuf> {
    let base = dirs::data_local_dir()
        .or_else(dirs::data_dir)
        .ok_or_else(|| anyhow!("Cannot determine local data directory"))?;
    Ok(base.join("voxcpm2-server").join("model"))
}

fn missing_files(model_dir: &Path) -> Vec<&'static str> {
    REQUIRED_FILES
        .iter()
        .filter(|f| !model_dir.join(f).exists())
        .copied()
        .collect()
}

fn download_file(url: &str, dest: &Path) -> Result<()> {
    use std::io::Write;
    info!("Downloading {}", url);
    let resp = reqwest::blocking::Client::new().get(url).send()?;
    if !resp.status().is_success() {
        bail!("Download failed: HTTP {}", resp.status());
    }
    let total = resp.content_length();
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(dest)?;
    let body = resp.bytes()?;
    for chunk in body.chunks(8192) {
        file.write_all(chunk)?;
    }
    if let Some(total) = total {
        info!(
            "  {:.1}/{:.1} MB",
            body.len() as f64 / 1_048_576.0,
            total as f64 / 1_048_576.0
        );
    }
    Ok(())
}

fn ensure_model(model_dir: &Path) -> Result<()> {
    let missing = missing_files(model_dir);
    if missing.is_empty() {
        info!("Model files found in {}", model_dir.display());
        return Ok(());
    }
    info!("Missing model files: {:?}", missing);
    info!("Downloading from HuggingFace ({})...", MODEL_REPO);
    for file in &missing {
        let url = format!("{}/{}/resolve/main/{}", HF_BASE, MODEL_REPO, file);
        let dest = model_dir.join(file);
        match download_file(&url, &dest) {
            Ok(()) => info!("  ✓ {}", file),
            Err(e) => {
                let _ = std::fs::remove_file(&dest);
                bail!("Failed to download {}: {}", file, e);
            }
        }
    }
    info!("All model files downloaded to {}", model_dir.display());
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let cli = Cli::parse();

    let model_path = match &cli.model {
        Some(p) => PathBuf::from(p),
        None => {
            let dir = default_model_dir()?;
            if !dir.exists() {
                info!("No --model specified, using default: {}", dir.display());
            }
            dir
        }
    };

    ensure_model(&model_path)?;

    info!("Loading model from: {}", model_path.display());
    let mut engine = VoxCPM2Engine::init(
        model_path
            .to_str()
            .ok_or_else(|| anyhow!("Invalid model path"))?,
        None,
        None,
    )?;
    info!("Model loaded, sample_rate: {}", engine.sample_rate());

    match cli.command {
        Commands::Generate {
            text,
            output,
            voice,
            prompt_text,
            prompt_wav,
            reference_wav,
            control_instruction,
            inference_timesteps,
            cfg_value,
            min_len,
            max_len,
        } => {
            let config = crate::model::config::InferenceConfig {
                min_len,
                max_len,
                inference_timesteps,
                cfg_value,
                ..Default::default()
            };
            let audio_tensor = engine.generate(GenerateRequest {
                text: &text,
                prompt_text: prompt_text.as_deref(),
                prompt_wav_path: prompt_wav.as_deref(),
                reference_wav_path: reference_wav.as_deref(),
                control_instruction,
                voice,
                config: &config,
            })?;
            let sr = engine.sample_rate() as u32;
            let wav_bytes = crate::audio::encode_wav(&audio_tensor, sr)?;
            std::fs::write(&output, &wav_bytes)?;
            info!(
                "Written {} ({:.1} KB) to {}",
                text,
                wav_bytes.len() as f64 / 1024.0,
                output
            );
        }
        Commands::RegisterVoice {
            name,
            prompt_text,
            prompt_wav,
            reference_wav,
        } => {
            if prompt_wav.is_none() && reference_wav.is_none() {
                bail!("At least one of --prompt-wav or --reference-wav is required");
            }
            engine.register_voice(
                &name,
                prompt_text.as_deref(),
                prompt_wav.as_deref(),
                reference_wav.as_deref(),
            )?;
            info!("Voice '{}' registered", name);
        }
        Commands::ListVoices => {
            let voices = engine.list_voices();
            if voices.is_empty() {
                println!("No voices registered.");
            } else {
                for name in voices {
                    println!("  {}", name);
                }
            }
        }
    }

    Ok(())
}
