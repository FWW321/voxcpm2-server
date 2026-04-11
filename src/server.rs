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

use crate::model::{config::InferenceConfig, generate::VoxCPM2Engine};

const SUPPORTED_MODEL: &str = "voxcpm2";

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub model: Option<String>,
    pub input: String,
    pub voice: Option<String>,
    pub response_format: Option<String>,
    #[serde(default)]
    pub speed: Option<f64>,
    pub prompt_text: Option<String>,
    pub prompt_wav_url: Option<String>,
    pub control_instruction: Option<String>,
    pub inference_timesteps: Option<usize>,
    pub cfg_value: Option<f64>,
    pub min_len: Option<usize>,
    pub max_len: Option<usize>,
}

impl SpeechRequest {
    fn response_format(&self) -> &str {
        self.response_format.as_deref().unwrap_or("mp3")
    }

    fn config(&self) -> InferenceConfig {
        InferenceConfig {
            min_len: self.min_len.unwrap_or(2),
            max_len: self.max_len.unwrap_or(4096),
            inference_timesteps: self.inference_timesteps.unwrap_or(10),
            cfg_value: self.cfg_value.unwrap_or(2.0),
            ..Default::default()
        }
    }

    fn validate(&self) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
        if self.input.is_empty() {
            return Err(error_response(StatusCode::BAD_REQUEST, "input is required"));
        }
        if let Some(ref model) = self.model
            && model != SUPPORTED_MODEL
        {
            return Err(error_response(
                StatusCode::NOT_FOUND,
                format!(
                    "Model '{}' not found. Supported: {}",
                    model, SUPPORTED_MODEL
                ),
            ));
        }
        let fmt = self.response_format();
        if crate::audio::content_type(fmt, 24000).is_err() {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                format!("Unsupported response_format '{}'", fmt),
            ));
        }
        if let Some(speed) = self.speed
            && !(0.25..=4.0).contains(&speed)
        {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                format!("speed must be between 0.25 and 4.0, got {}", speed),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub struct VoiceRequest {
    pub name: String,
    pub prompt_text: String,
    pub prompt_wav_url: String,
}

#[derive(Debug, Serialize)]
struct ModelObject {
    id: &'static str,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
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

fn error_response(
    status: StatusCode,
    message: impl AsRef<str>,
) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.as_ref().to_string(),
                r#type: "invalid_request_error".into(),
                code: None,
            },
        }),
    )
}

pub struct AppState {
    pub engine: Mutex<VoxCPM2Engine>,
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
    if let Err((status, body)) = req.validate() {
        return (status, body).into_response();
    }

    let response_format = req.response_format().to_string();
    let speed = req.speed;
    let config = req.config();

    let input_preview: String = req.input.chars().take(50).collect();
    let prompt_text = req.prompt_text;
    let prompt_wav_path = req.prompt_wav_url;
    let control_instruction = req.control_instruction;
    let voice = req.voice;
    let has_ref_audio = prompt_wav_path.is_some();
    info!(
        "TTS request: input='{}...', voice={:?}, format={}, prompt_text={:?}, has_ref_audio={}, control_instruction={:?}",
        input_preview, voice, response_format, prompt_text, has_ref_audio, control_instruction,
    );

    let state = Arc::clone(&state);
    let input = req.input;
    let result = tokio::task::spawn_blocking(move || -> Result<Response, anyhow::Error> {
        let mut engine = state
            .engine
            .lock()
            .map_err(|e| anyhow::anyhow!("Engine lock poisoned: {}", e))?;
        let audio_tensor = engine.generate(
            &input,
            prompt_text,
            prompt_wav_path,
            control_instruction,
            voice,
            &config,
        )?;

        let sr = engine.sample_rate() as u32;
        let ct = crate::audio::content_type(&response_format, sr)?;
        let bytes = crate::audio::encode(&audio_tensor, sr, &response_format, speed)?;
        info!("TTS complete: {} bytes ({})", bytes.len(), response_format);

        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, HeaderValue::from_str(&ct)?);
        Ok((StatusCode::OK, headers, bytes).into_response())
    })
    .await;

    match result {
        Ok(Ok(resp)) => resp,
        Ok(Err(e)) => {
            let (status, body) =
                error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("{}", e));
            (status, body).into_response()
        }
        Err(e) => {
            let (status, body) = error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Task join error: {}", e),
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

    let state = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || {
        let mut engine = state
            .engine
            .lock()
            .map_err(|e| anyhow::anyhow!("Engine lock poisoned: {}", e))?;
        engine.register_voice(req.name, req.prompt_text, req.prompt_wav_url)
    })
    .await;

    match result {
        Ok(Ok(())) => {
            info!("Voice registered successfully");
            (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))).into_response()
        }
        Ok(Err(e)) => {
            error!("Voice registration error: {:?}", e);
            let (status, body) = error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to register voice: {}", e),
            );
            (status, body).into_response()
        }
        Err(e) => {
            error!("Voice registration task error: {:?}", e);
            let (status, body) = error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Task join error: {}", e),
            );
            (status, body).into_response()
        }
    }
}

async fn list_voices_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
    let voices: Vec<&String> = engine.list_voices();
    Json(serde_json::json!({
        "voices": voices,
    }))
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
        let (status, body) =
            error_response(StatusCode::NOT_FOUND, format!("Voice '{}' not found", name));
        (status, body).into_response()
    }
}

async fn models_handler() -> impl IntoResponse {
    let response = ModelsResponse {
        object: "list".into(),
        data: vec![ModelObject {
            id: SUPPORTED_MODEL,
            object: "model",
            created: 0,
            owned_by: "openbmb",
        }],
    };
    Json(response)
}

async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
    let sample_rate = engine.sample_rate();
    let voice_count = engine.list_voices().len();
    Json(serde_json::json!({
        "status": "ok",
        "model": SUPPORTED_MODEL,
        "sample_rate": sample_rate,
        "voices": voice_count,
    }))
}
