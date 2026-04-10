mod audio;
mod model;
mod nn;
mod server;
mod utils;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use clap::Parser;
use tracing::{Level, info, warn};

use crate::model::generate::VoxCPM2Engine;
use crate::server::{AppState, create_router};

const MODEL_REPO: &str = "openbmb/VoxCPM2";
const HF_BASE: &str = "https://huggingface.co";

const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "model.safetensors",
    "audiovae.pth",
    "tokenizer.json",
];

#[derive(Parser, Debug)]
#[command(name = "voxcpm2-server", about = "VoxCPM2 TTS inference server")]
struct Args {
    #[arg(short, long, help = "Path to VoxCPM2 model directory")]
    model: Option<String>,

    #[arg(short, long, default_value = "0.0.0.0", help = "Host to bind")]
    host: String,

    #[arg(short, long, default_value_t = 5800, help = "Port to bind")]
    port: u16,
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
    let resp = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async { reqwest::get(url).await })
    })?;
    if !resp.status().is_success() {
        bail!("Download failed: HTTP {}", resp.status());
    }
    let total = resp.content_length();
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(dest)?;
    let mut downloaded: u64 = 0;
    let body = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            use futures_util::StreamExt;
            let mut buf = Vec::new();
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                buf.extend_from_slice(&chunk);
            }
            Ok::<Vec<u8>, anyhow::Error>(buf)
        })
    })?;
    for chunk in body.chunks(8192) {
        file.write_all(chunk)?;
        downloaded += chunk.len() as u64;
        if let Some(total) = total {
            let pct = downloaded as f64 / total as f64 * 100.0;
            let mb_dl = downloaded as f64 / 1_048_576.0;
            let mb_total = total as f64 / 1_048_576.0;
            eprint!("\r  {:.1}/{:.1} MB ({:.1}%)", mb_dl, mb_total, pct);
        }
    }
    println!();
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let args = Args::parse();

    let model_path = match &args.model {
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

    info!("Loading VoxCPM2 model from: {}", model_path.display());
    let model_str = model_path
        .to_str()
        .ok_or_else(|| anyhow!("Model path contains non-UTF-8 characters"))?
        .to_string();
    let engine =
        tokio::task::spawn_blocking(move || VoxCPM2Engine::init(&model_str, None, None)).await??;
    info!("Model loaded, sample_rate: {}", engine.sample_rate());

    let state = Arc::new(AppState {
        engine: std::sync::Mutex::new(engine),
    });

    let app = create_router(state);
    let addr = format!("{}:{}", args.host, args.port);
    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Server listening on {}", addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if tokio::signal::ctrl_c().await.is_err() {
            warn!("failed to listen for ctrl+c");
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                warn!("failed to listen for SIGTERM: {e}");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("Received Ctrl+C, shutting down"),
        _ = terminate => info!("Received SIGTERM, shutting down"),
    }
}
