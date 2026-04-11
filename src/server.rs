use std::sync::{Arc, Mutex};

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tracing::{error, info};

use crate::model::{
    config::InferenceConfig,
    generate::{GenerateRequest, VoxCPM2Engine},
};

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub input: String,
    pub voice: Option<String>,
    pub prompt_text: Option<String>,
    pub prompt_wav_url: Option<String>,
    pub reference_wav_url: Option<String>,
    pub control_instruction: Option<String>,
    pub inference_timesteps: Option<usize>,
    pub cfg_value: Option<f64>,
    pub min_len: Option<usize>,
    pub max_len: Option<usize>,
}

impl SpeechRequest {
    fn config(&self) -> InferenceConfig {
        InferenceConfig {
            min_len: self.min_len.unwrap_or(2),
            max_len: self.max_len.unwrap_or(4096),
            inference_timesteps: self.inference_timesteps.unwrap_or(10),
            cfg_value: self.cfg_value.unwrap_or(2.0),
            ..Default::default()
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct VoiceRequest {
    pub name: String,
    pub prompt_text: Option<String>,
    #[serde(default)]
    pub prompt_wav_url: Option<String>,
    #[serde(default)]
    pub reference_wav_url: Option<String>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

pub struct AppState {
    pub engine: Mutex<VoxCPM2Engine>,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/speech", post(speech_handler))
        .route("/voices", post(register_voice_handler))
        .route("/voices", get(list_voices_handler))
        .route("/voices/{name}", delete(delete_voice_handler))
        .route("/health", get(health_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn speech_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    if req.input.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "input is required");
    }

    let config = req.config();
    let input_preview: String = req.input.chars().take(50).collect();
    let prompt_text = req.prompt_text;
    let prompt_wav_path = req.prompt_wav_url;
    let reference_wav_path = req.reference_wav_url;
    let control_instruction = req.control_instruction;
    let voice = req.voice;
    info!(
        "TTS: input='{}...', voice={:?}, prompt_text={:?}, ref_wav={}, control={:?}",
        input_preview,
        voice,
        prompt_text,
        reference_wav_path.is_some(),
        control_instruction,
    );

    let state = Arc::clone(&state);
    let input = req.input;
    let result = tokio::task::spawn_blocking(move || -> Result<Response, anyhow::Error> {
        let mut engine = state
            .engine
            .lock()
            .map_err(|e| anyhow::anyhow!("Engine lock poisoned: {}", e))?;
        let audio_tensor = engine.generate(GenerateRequest {
            text: &input,
            prompt_text: prompt_text.as_deref(),
            prompt_wav_path: prompt_wav_path.as_deref(),
            reference_wav_path: reference_wav_path.as_deref(),
            control_instruction,
            voice,
            config: &config,
        })?;

        let sr = engine.sample_rate() as u32;
        let wav_bytes = crate::audio::encode_wav(&audio_tensor, sr)?;
        info!("TTS complete: {} bytes WAV", wav_bytes.len());

        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
        Ok((StatusCode::OK, headers, wav_bytes).into_response())
    })
    .await;

    match result {
        Ok(Ok(resp)) => resp,
        Ok(Err(e)) => {
            error!("TTS error: {:?}", e);
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string())
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn register_voice_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VoiceRequest>,
) -> Response {
    if req.name.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "name is required");
    }
    if req.prompt_wav_url.is_none() && req.reference_wav_url.is_none() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "prompt_wav_url or reference_wav_url is required",
        );
    }

    info!("Registering voice: name='{}'", req.name);

    let state = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || {
        let mut engine = state
            .engine
            .lock()
            .map_err(|e| anyhow::anyhow!("Engine lock poisoned: {}", e))?;
        engine.register_voice(
            req.name,
            req.prompt_text.as_deref(),
            req.prompt_wav_url.as_deref(),
            req.reference_wav_url.as_deref(),
        )
    })
    .await;

    match result {
        Ok(Ok(())) => {
            info!("Voice registered successfully");
            (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))).into_response()
        }
        Ok(Err(e)) => {
            error!("Voice registration error: {:?}", e);
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string())
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn list_voices_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
    let voices: Vec<&String> = engine.list_voices();
    Json(serde_json::json!({ "voices": voices }))
}

async fn delete_voice_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Response {
    let mut engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
    if engine.remove_voice(&name) {
        info!("Voice deleted: {}", name);
        (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))).into_response()
    } else {
        error_response(
            StatusCode::NOT_FOUND,
            &format!("Voice '{}' not found", name),
        )
    }
}

async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
    let sample_rate = engine.sample_rate();
    let voice_count = engine.list_voices().len();
    Json(serde_json::json!({
        "status": "ok",
        "sample_rate": sample_rate,
        "voices": voice_count,
    }))
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: message.to_string(),
        }),
    )
        .into_response()
}
