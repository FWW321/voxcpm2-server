use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use iced::widget::{
    Scrollable, button, column, container, pick_list, row, rule, space, text, text_input,
};
use iced::{Center, Element, Fill, Font, Length, Task, Theme};
use rodio::{Decoder, DeviceSinkBuilder, Player};
use tokio::sync::Mutex;

use crate::model::config::InferenceConfig;
use crate::model::generate::{GenerateRequest, VoxCPM2Engine};

const WINDOW_WIDTH: f32 = 720.0;
const WINDOW_HEIGHT: f32 = 600.0;

fn secondary_text_color(theme: &Theme) -> iced::widget::text::Style {
    iced::widget::text::Style {
        color: Some(theme.extended_palette().background.weakest.text),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lang {
    Zh,
    En,
}

impl Lang {
    const ALL: [Lang; 2] = [Lang::Zh, Lang::En];

    fn label(self) -> &'static str {
        match self {
            Lang::Zh => "中文",
            Lang::En => "EN",
        }
    }
}

macro_rules! t {
    ($lang:expr, $zh:literal, $en:literal) => {
        match $lang {
            Lang::Zh => $zh,
            Lang::En => $en,
        }
    };
}

#[derive(Debug, Clone, Copy)]
enum VoiceMode {
    VoiceDesign,
    ControllableCloning,
    Continuation,
    Ultimate,
}

impl VoiceMode {
    const ALL: [VoiceMode; 4] = [
        VoiceMode::VoiceDesign,
        VoiceMode::ControllableCloning,
        VoiceMode::Continuation,
        VoiceMode::Ultimate,
    ];

    fn label(self, lang: Lang) -> &'static str {
        match self {
            VoiceMode::VoiceDesign => t!(lang, "语音设计", "Voice Design"),
            VoiceMode::ControllableCloning => t!(lang, "可控克隆", "Controllable"),
            VoiceMode::Continuation => t!(lang, "续接生成", "Continuation"),
            VoiceMode::Ultimate => t!(lang, "极致克隆", "Ultimate"),
        }
    }

    fn description(self, lang: Lang) -> &'static str {
        match self {
            VoiceMode::VoiceDesign => t!(lang, "仅文本", "Text only"),
            VoiceMode::ControllableCloning => t!(lang, "参考音频", "Ref audio"),
            VoiceMode::Continuation => t!(lang, "提示词文本+音频", "Prompt text+audio"),
            VoiceMode::Ultimate => t!(lang, "参考+提示词", "Ref + prompt"),
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Message {
    TextChanged(String),
    ControlInstructionChanged(String),
    ModeSelected(VoiceMode),
    ReferenceWavBrowse,
    ReferenceWavSelected(Option<PathBuf>),
    PromptWavBrowse,
    PromptWavSelected(Option<PathBuf>),
    PromptTextChanged(String),
    VoiceSelected(String),
    Generate,
    PlayToggle,
    StopPlayback,
    GenerationFinished(Result<Vec<u8>, String>),
    EngineReady(EngineResult),
    RegisterVoice,
    VoiceRegistered(String),
    VoiceRegisterFailed(String),
    VoiceNameChanged(String),
    LanguageSwitched,
}

#[derive(Clone)]
struct EngineResult(Result<Arc<Mutex<VoxCPM2Engine>>, String>);

impl std::fmt::Debug for EngineResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Ok(_) => write!(f, "EngineResult(Ok)"),
            Err(e) => write!(f, "EngineResult(Err({:?}))", e),
        }
    }
}

struct VoxCPM2App {
    lang: Lang,
    text: String,
    control_instruction: String,
    mode: VoiceMode,
    reference_wav: Option<PathBuf>,
    prompt_wav: Option<PathBuf>,
    prompt_text: String,
    voice_name: String,
    selected_voice: Option<String>,
    registered_voices: Vec<String>,

    engine: Option<Arc<Mutex<VoxCPM2Engine>>>,
    model_loading: bool,
    generating: bool,
    generated_audio: Option<Vec<u8>>,
    status_message: String,

    playing: bool,
    sink_handle: Option<rodio::MixerDeviceSink>,
    player: Option<rodio::Player>,
}

impl VoxCPM2App {
    fn new() -> (Self, Task<Message>) {
        let model_dir = crate::default_model_dir().unwrap_or_else(|_| PathBuf::from("model"));

        let task = Task::perform(load_engine(model_dir), |result| {
            Message::EngineReady(EngineResult(result))
        });

        (
            Self {
                lang: Lang::Zh,
                text: String::new(),
                control_instruction: String::new(),
                mode: VoiceMode::VoiceDesign,
                reference_wav: None,
                prompt_wav: None,
                prompt_text: String::new(),
                voice_name: String::new(),
                selected_voice: None,
                registered_voices: Vec::new(),
                engine: None,
                model_loading: true,
                generating: false,
                generated_audio: None,
                status_message: t!(Lang::Zh, "正在加载模型…", "Loading model...").to_string(),
                playing: false,
                sink_handle: None,
                player: None,
            },
            task,
        )
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::TextChanged(s) => self.text = s,
            Message::ControlInstructionChanged(s) => self.control_instruction = s,
            Message::ModeSelected(mode) => self.mode = mode,
            Message::LanguageSwitched => {
                self.lang = if self.lang == Lang::Zh { Lang::En } else { Lang::Zh };
                self.status_message = t!(self.lang, "已切换语言", "Language switched").to_string();
            }
            Message::ReferenceWavBrowse => {
                return Task::perform(
                    pick_file(t!(self.lang, "选择参考音频", "Select reference audio")),
                    Message::ReferenceWavSelected,
                );
            }
            Message::ReferenceWavSelected(path) => self.reference_wav = path,
            Message::PromptWavBrowse => {
                return Task::perform(pick_file(t!(self.lang, "选择提示音频", "Select prompt audio")), Message::PromptWavSelected);
            }
            Message::PromptWavSelected(path) => self.prompt_wav = path,
            Message::PromptTextChanged(s) => self.prompt_text = s,
            Message::VoiceNameChanged(s) => self.voice_name = s,
            Message::VoiceSelected(name) => self.selected_voice = Some(name),

            Message::RegisterVoice => {
                if let Some(engine) = &self.engine {
                    let name = self.voice_name.clone();
                    let prompt_text = if self.prompt_text.is_empty() {
                        None
                    } else {
                        Some(self.prompt_text.clone())
                    };
                    let prompt_wav = self.prompt_wav.clone();
                    let reference_wav = self.reference_wav.clone();
                    let engine = engine.clone();

                    self.status_message = format!(
                        "{} '{}'",
                        t!(self.lang, "正在注册音色", "Registering voice"),
                        name
                    );
                    let name_clone = name.clone();
                    return Task::perform(
                        async move {
                            let mut eng = engine.lock().await;
                            eng.register_voice(
                                &name,
                                prompt_text.as_deref(),
                                prompt_wav.as_ref().and_then(|p| p.to_str()),
                                reference_wav.as_ref().and_then(|p| p.to_str()),
                            )
                            .map_err(|e| e.to_string())
                        },
                        move |result| match result {
                            Ok(()) => Message::VoiceRegistered(name_clone),
                            Err(e) => Message::VoiceRegisterFailed(e),
                        },
                    );
                }
            }
            Message::Generate => {
                if self.engine.is_none() || self.generating {
                    return Task::none();
                }
                if self.text.is_empty() {
                    self.status_message = t!(self.lang, "请输入要合成的文本", "Please enter text to synthesize").to_string();
                    return Task::none();
                }

                self.generating = true;
                self.generated_audio = None;
                self.stop_playback();
                self.status_message = t!(self.lang, "正在生成音频…", "Generating audio...").to_string();

                let engine = self.engine.clone().unwrap();
                let text = self.text.clone();
                let control_instruction = if self.control_instruction.is_empty() {
                    None
                } else {
                    Some(self.control_instruction.clone())
                };
                let voice = self.selected_voice.clone();
                let prompt_text = if self.prompt_text.is_empty() {
                    None
                } else {
                    Some(self.prompt_text.clone())
                };
                let prompt_wav = self.prompt_wav.clone();
                let reference_wav = self.reference_wav.clone();

                return Task::perform(
                    generate_audio(
                        engine,
                        text,
                        control_instruction,
                        voice,
                        prompt_text,
                        prompt_wav,
                        reference_wav,
                    ),
                    Message::GenerationFinished,
                );
            }

            Message::GenerationFinished(result) => {
                self.generating = false;
                match result {
                    Ok(bytes) => {
                        let size_kb = bytes.len() as f64 / 1024.0;
                        self.status_message = format!(
                            "{} ({:.1} KB)",
                            t!(self.lang, "生成完成", "Generated"),
                            size_kb
                        );
                        self.generated_audio = Some(bytes);
                    }
                    Err(e) => {
                        self.status_message = format!(
                            "{}: {}",
                            t!(self.lang, "错误", "Error"),
                            e
                        );
                    }
                }
            }

            Message::EngineReady(EngineResult(result)) => {
                self.model_loading = false;
                match result {
                    Ok(engine) => {
                        let voices = {
                            let eng = engine.blocking_lock();
                            eng.list_voices().into_iter().cloned().collect()
                        };
                        self.registered_voices = voices;
                        self.engine = Some(engine);
                        self.status_message = t!(self.lang, "模型加载完成，可以生成", "Model loaded. Ready to generate.").to_string();
                    }
                    Err(e) => {
                        self.status_message = format!(
                            "{}: {}",
                            t!(self.lang, "模型加载失败", "Model load failed"),
                            e
                        );
                    }
                }
            }

            Message::PlayToggle => {
                if self.playing {
                    self.stop_playback();
                } else {
                    let audio = self.generated_audio.clone();
                    if let Some(audio) = audio {
                        self.play_audio(&audio);
                    }
                }
            }
            Message::StopPlayback => self.stop_playback(),

            Message::VoiceRegistered(name) => {
                self.status_message = format!(
                    "{} '{}'",
                    t!(self.lang, "音色已注册", "Voice registered"),
                    name
                );
                if let Some(engine) = &self.engine {
                    let voices = {
                        let eng = engine.blocking_lock();
                        eng.list_voices().into_iter().cloned().collect()
                    };
                    self.registered_voices = voices;
                    self.selected_voice = Some(name);
                }
            }
            Message::VoiceRegisterFailed(e) => {
                self.status_message = format!(
                    "{}: {}",
                    t!(self.lang, "注册失败", "Registration failed"),
                    e
                );
            }
        }
        Task::none()
    }

    fn play_audio(&mut self, wav_bytes: &[u8]) {
        let sink = match DeviceSinkBuilder::open_default_sink() {
            Ok(s) => s,
            Err(_) => {
                self.status_message = t!(self.lang, "音频播放失败", "Audio playback failed").to_string();
                return;
            }
        };
        let player = Player::connect_new(sink.mixer());
        let cursor = Cursor::new(wav_bytes.to_vec());
        let source = match Decoder::new(cursor) {
            Ok(s) => s,
            Err(_) => {
                self.status_message = t!(self.lang, "音频解码失败", "Audio decode failed").to_string();
                return;
            }
        };
        player.append(source);
        player.detach();
        self.sink_handle = Some(sink);
        self.player = None;
        self.playing = true;
    }

    fn stop_playback(&mut self) {
        if let Some(p) = &self.player {
            p.stop();
        }
        self.player = None;
        self.sink_handle = None;
        self.playing = false;
    }

    fn view(&self) -> Element<'_, Message> {
        let lang = self.lang;

        let lang_button = button(text(Lang::ALL.iter().find(|&&l| l != lang).unwrap().label()).size(12))
            .on_press(Message::LanguageSwitched)
            .padding([4.0, 10.0])
            .style(button::secondary);

        let title = text("VoxCPM2").size(28).font(Font {
            weight: iced::font::Weight::Bold,
            ..Font::DEFAULT
        });

        let subtitle = text(t!(lang, "语音合成系统", "Text-to-Speech Synthesizer"))
            .size(13)
            .style(secondary_text_color);

        let header = row![column![title, subtitle].spacing(2), space::horizontal(), lang_button]
            .padding([8.0, 0.0])
            .align_y(Center);

        let text_input_section = column![
            text(t!(lang, "合成文本", "Text to synthesize"))
                .size(13)
                .style(secondary_text_color),
            text_input(t!(lang, "在此输入文本…", "Enter text here..."), &self.text)
                .on_input(Message::TextChanged)
                .padding([10.0, 14.0])
                .size(15),
        ]
        .spacing(6);

        let control_input = column![
            text(t!(lang, "音色描述（可选）", "Voice description (optional)"))
                .size(13)
                .style(secondary_text_color),
            text_input(
                t!(lang, "例：(温柔女声)", "e.g. (gentle female voice)"),
                &self.control_instruction
            )
            .on_input(Message::ControlInstructionChanged)
            .padding([8.0, 14.0])
            .size(14),
        ]
        .spacing(6);

        let mode_label = text(t!(lang, "生成模式", "Generation Mode"))
            .size(13)
            .style(secondary_text_color);
        let mode_buttons: Element<_> = row(VoiceMode::ALL.iter().map(|m| {
            let is_selected = std::mem::discriminant(&self.mode) == std::mem::discriminant(m);
            let m_clone = *m;
            let btn = button(
                column![
                    text(m.label(lang)).size(12),
                    text(m.description(lang)).size(10).style(secondary_text_color),
                ]
                .spacing(2)
                .align_x(Center),
            )
            .on_press(Message::ModeSelected(m_clone))
            .width(Fill)
            .padding([8.0, 6.0])
            .style(move |theme: &Theme, status| {
                if is_selected {
                    button::primary(theme, status)
                } else {
                    button::secondary(theme, status)
                }
            });

            btn.into()
        }))
        .spacing(6)
        .into();

        let mode_section = column![mode_label, mode_buttons].spacing(6);

        let no_file = t!(lang, "未选择文件", "No file selected");
        let browse = t!(lang, "浏览…", "Browse...");

        let ref_row: Element<_> = row![
            text(t!(lang, "参考音频", "Reference audio"))
                .size(13)
                .style(secondary_text_color),
            space::horizontal(),
            text(
                self.reference_wav
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .unwrap_or(no_file)
            )
            .size(12)
            .style(secondary_text_color),
            button(browse)
                .on_press(Message::ReferenceWavBrowse)
                .padding([6.0, 14.0])
                .style(button::secondary),
        ]
        .spacing(8)
        .align_y(Center)
        .into();

        let prompt_row: Element<_> = row![
            text(t!(lang, "提示音频", "Prompt audio"))
                .size(13)
                .style(secondary_text_color),
            space::horizontal(),
            text(
                self.prompt_wav
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .unwrap_or(no_file)
            )
            .size(12)
            .style(secondary_text_color),
            button(browse)
                .on_press(Message::PromptWavBrowse)
                .padding([6.0, 14.0])
                .style(button::secondary),
        ]
        .spacing(8)
        .align_y(Center)
        .into();

        let prompt_text_input = column![
            text(t!(lang, "提示词文本（转录）", "Prompt text (transcript)"))
                .size(13)
                .style(secondary_text_color),
            text_input(
                t!(lang, "提示音频的转录文本…", "Transcript of prompt audio..."),
                &self.prompt_text
            )
            .on_input(Message::PromptTextChanged)
            .padding([8.0, 14.0])
            .size(14),
        ]
        .spacing(6);

        let audio_section = column![
            if matches!(
                self.mode,
                VoiceMode::ControllableCloning | VoiceMode::Ultimate
            ) {
                ref_row
            } else {
                column![].into()
            },
            if matches!(self.mode, VoiceMode::Continuation | VoiceMode::Ultimate) {
                let col: Element<_> = column![prompt_row, prompt_text_input].spacing(8).into();
                col
            } else {
                column![].into()
            },
        ]
        .spacing(8);

        let voice_section = if !self.registered_voices.is_empty() {
            let voice_picker = pick_list(
                self.registered_voices.clone(),
                self.selected_voice.clone(),
                Message::VoiceSelected,
            )
            .padding([8.0, 14.0])
            .placeholder(t!(lang, "选择音色…", "Select a voice..."));

            column![
                text(t!(lang, "已保存音色（可选）", "Saved voice (optional)"))
                    .size(13)
                    .style(secondary_text_color),
                voice_picker,
            ]
            .spacing(6)
        } else {
            column![]
        };

        let register_row = row![
            text_input(t!(lang, "音色名称…", "Voice name..."), &self.voice_name)
                .on_input(Message::VoiceNameChanged)
                .padding([8.0, 14.0])
                .width(Length::Fixed(180.0)),
            button(t!(lang, "注册音色", "Register Voice"))
                .on_press(Message::RegisterVoice)
                .padding([8.0, 16.0])
                .style(button::secondary),
        ]
        .spacing(8)
        .align_y(Center);

        let generate_button = if self.generating {
            button(
                row![
                    text(t!(lang, "生成中", "Generating")),
                    space::horizontal(),
                    text("...").size(12)
                ]
                .spacing(6)
                .align_y(Center),
            )
            .padding([12.0, 32.0])
            .width(Fill)
            .style(button::secondary)
        } else {
            button(
                text(t!(lang, "生成", "Generate"))
                    .width(Fill)
                    .align_x(Center)
                    .size(15),
            )
            .on_press(Message::Generate)
            .padding([12.0, 32.0])
            .width(Fill)
            .style(button::primary)
        };

        let play_label = t!(lang, "播放", "Play");
        let stop_label = t!(lang, "停止", "Stop");

        let playback_row = row![
            if self.generated_audio.is_some() {
                button(if self.playing { stop_label } else { play_label })
                    .on_press(Message::PlayToggle)
                    .padding([8.0, 20.0])
                    .style(if self.playing {
                        button::danger
                    } else {
                        button::success
                    })
            } else {
                button(play_label)
                    .padding([8.0, 20.0])
                    .style(button::secondary)
            },
            space::horizontal(),
            text(&self.status_message)
                .size(12)
                .style(secondary_text_color),
        ]
        .spacing(8)
        .align_y(Center);

        let status_bar = container(playback_row)
            .padding([8.0, 12.0])
            .style(container::rounded_box);

        let content = column![
            header,
            rule::horizontal(1),
            text_input_section,
            control_input,
            space::vertical(),
            mode_section,
            audio_section,
            voice_section,
            register_row,
            space::vertical(),
            generate_button,
            space::vertical(),
            status_bar,
        ]
        .spacing(12)
        .padding([20.0, 24.0]);

        Scrollable::new(content).into()
    }
}

async fn load_engine(model_dir: PathBuf) -> Result<Arc<Mutex<VoxCPM2Engine>>, String> {
    crate::ensure_model(&model_dir).map_err(|e| e.to_string())?;
    let path = model_dir
        .to_str()
        .ok_or_else(|| "Invalid path".to_string())?;
    let engine = VoxCPM2Engine::init(path, None, None).map_err(|e| e.to_string())?;
    Ok(Arc::new(Mutex::new(engine)))
}

#[allow(clippy::too_many_arguments)]
async fn generate_audio(
    engine: Arc<Mutex<VoxCPM2Engine>>,
    text: String,
    control_instruction: Option<String>,
    voice: Option<String>,
    prompt_text: Option<String>,
    prompt_wav: Option<PathBuf>,
    reference_wav: Option<PathBuf>,
) -> Result<Vec<u8>, String> {
    let config = InferenceConfig::default();
    let mut eng = engine.lock().await;

    let audio_tensor = eng
        .generate(GenerateRequest {
            text: &text,
            prompt_text: prompt_text.as_deref(),
            prompt_wav_path: prompt_wav.as_ref().and_then(|p| p.to_str()),
            reference_wav_path: reference_wav.as_ref().and_then(|p| p.to_str()),
            control_instruction,
            voice,
            config: &config,
        })
        .map_err(|e| e.to_string())?;

    let sr = eng.sample_rate() as u32;
    crate::audio::encode_wav(&audio_tensor, sr).map_err(|e| e.to_string())
}

async fn pick_file(title: &str) -> Option<PathBuf> {
    rfd::AsyncFileDialog::new()
        .set_title(title)
        .add_filter("Audio", &["wav", "mp3", "flac", "ogg"])
        .pick_file()
        .await
        .map(|f| f.path().to_path_buf())
}

fn dark_theme(_: &VoxCPM2App) -> Theme {
    Theme::Dark
}

pub fn run() -> iced::Result {
    iced::application(VoxCPM2App::new, VoxCPM2App::update, VoxCPM2App::view)
        .title("VoxCPM2")
        .theme(dark_theme)
        .window_size((WINDOW_WIDTH, WINDOW_HEIGHT))
        .centered()
        .run()
}
