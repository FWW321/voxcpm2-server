use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::{error, info};

use crate::model::{config::InferenceConfig, generate::VoxCPM2Engine};

const SUPPORTED_MODEL: &str = "voxcpm2";

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub model: Option<String>,
    pub input: String,
    pub voice: Option<String>,
    #[serde(default = "default_response_format")]
    pub response_format: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub speed: Option<f64>,
    pub prompt_text: Option<String>,
    pub prompt_wav_url: Option<String>,
    pub control_instruction: Option<String>,
    #[serde(default = "default_inference_timesteps")]
    pub inference_timesteps: Option<usize>,
    #[serde(default = "default_cfg_value")]
    pub cfg_value: Option<f64>,
    #[serde(default = "default_min_len")]
    pub min_len: Option<usize>,
    #[serde(default = "default_max_len")]
    pub max_len: Option<usize>,
}

fn default_response_format() -> Option<String> {
    Some("wav".to_string())
}
fn default_inference_timesteps() -> Option<usize> {
    Some(10)
}
fn default_cfg_value() -> Option<f64> {
    Some(2.0)
}
fn default_min_len() -> Option<usize> {
    Some(2)
}
fn default_max_len() -> Option<usize> {
    Some(4096)
}

#[derive(Debug, Deserialize)]
pub struct VoiceRequest {
    pub name: String,
    pub prompt_text: String,
    pub prompt_wav_url: String,
}

#[derive(Debug, Serialize)]
struct ModelObject {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
    code: Option<String>,
}

fn error_response(status: StatusCode, message: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.to_string(),
                r#type: "invalid_request_error".to_string(),
                code: None,
            },
        }),
    )
}

pub struct AppState {
    pub engine: RwLock<VoxCPM2Engine>,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/audio/speech", post(speech_handler))
        .route("/v1/audio/voices", post(register_voice_handler))
        .route("/v1/audio/voices", get(list_voices_handler))
        .route("/v1/audio/voices/{name}", delete(delete_voice_handler))
        .route("/v1/models", get(models_handler))
        .route("/health", get(health_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn speech_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    if req.input.is_empty() {
        let (status, body) = error_response(StatusCode::BAD_REQUEST, "input is required");
        return (status, body).into_response();
    }

    if let Some(ref model) = req.model
        && model != SUPPORTED_MODEL
    {
        let (status, body) = error_response(
            StatusCode::NOT_FOUND,
            &format!(
                "Model '{}' not found. Supported: {}",
                model, SUPPORTED_MODEL
            ),
        );
        return (status, body).into_response();
    }

    if let Some(ref fmt) = req.response_format
        && fmt != "wav"
    {
        let (status, body) = error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "Unsupported response_format '{}'. Only 'wav' is supported.",
                fmt
            ),
        );
        return (status, body).into_response();
    }

    let prompt_text = req.prompt_text;
    let prompt_wav_path = req.prompt_wav_url;
    let control_instruction = req.control_instruction;
    let voice = req.voice;
    let config = InferenceConfig {
        min_len: req.min_len.unwrap_or(2),
        max_len: req.max_len.unwrap_or(4096),
        inference_timesteps: req.inference_timesteps.unwrap_or(10),
        cfg_value: req.cfg_value.unwrap_or(2.0),
        ..Default::default()
    };

    let input_preview: String = req.input.chars().take(50).collect();
    info!(
        "TTS request: input='{}...', voice={:?}, prompt_text={:?}, has_ref_audio={}, control_instruction={:?}",
        input_preview,
        voice,
        prompt_text,
        prompt_wav_path.is_some(),
        control_instruction,
    );

    let mut engine = state.engine.write().await;
    match engine.generate(
        req.input,
        prompt_text,
        prompt_wav_path,
        control_instruction,
        voice,
        &config,
    ) {
        Ok(wav_bytes) => {
            info!("TTS complete: {} bytes", wav_bytes.len());
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
            (StatusCode::OK, headers, wav_bytes).into_response()
        }
        Err(e) => {
            error!("TTS inference error: {:?}", e);
            let (status, body) = error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Inference failed: {}", e),
            );
            (status, body).into_response()
        }
    }
}

async fn register_voice_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VoiceRequest>,
) -> Response {
    if req.name.is_empty() {
        let (status, body) = error_response(StatusCode::BAD_REQUEST, "name is required");
        return (status, body).into_response();
    }
    if req.prompt_text.is_empty() {
        let (status, body) = error_response(StatusCode::BAD_REQUEST, "prompt_text is required");
        return (status, body).into_response();
    }
    if req.prompt_wav_url.is_empty() {
        let (status, body) = error_response(StatusCode::BAD_REQUEST, "prompt_wav_url is required");
        return (status, body).into_response();
    }

    info!("Registering voice: name='{}'", req.name);

    let mut engine = state.engine.write().await;
    match engine.register_voice(req.name, req.prompt_text, req.prompt_wav_url) {
        Ok(()) => {
            info!("Voice registered successfully");
            (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))).into_response()
        }
        Err(e) => {
            error!("Voice registration error: {:?}", e);
            let (status, body) = error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Failed to register voice: {}", e),
            );
            (status, body).into_response()
        }
    }
}

async fn list_voices_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let engine = state.engine.read().await;
    let voices: Vec<&String> = engine.list_voices();
    Json(serde_json::json!({
        "voices": voices,
    }))
}

async fn delete_voice_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Response {
    let mut engine = state.engine.write().await;
    if engine.remove_voice(&name) {
        info!("Voice deleted: {}", name);
        (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))).into_response()
    } else {
        let (status, body) = error_response(
            StatusCode::NOT_FOUND,
            &format!("Voice '{}' not found", name),
        );
        (status, body).into_response()
    }
}

async fn models_handler() -> impl IntoResponse {
    let response = ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: SUPPORTED_MODEL.to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "openbmb".to_string(),
        }],
    };
    Json(response)
}

async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let engine = state.engine.read().await;
    let sample_rate = engine.sample_rate();
    let voice_count = engine.list_voices().len();
    Json(serde_json::json!({
        "status": "ok",
        "model": SUPPORTED_MODEL,
        "sample_rate": sample_rate,
        "voices": voice_count,
    }))
}
