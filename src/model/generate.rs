use anyhow::Result;
use candle_core::{DType, Device, Tensor, pickle::read_all_with_key};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use tracing::info;

use crate::model::{
    audio_vae::AudioVAE,
    config::{AudioVaeConfig, InferenceConfig, PromptCache, VoxCPMConfig},
    tokenizer::SingleChineseTokenizer,
    voxcpm::VoxCPMModel,
};
use crate::utils::{find_type_files, get_device, get_dtype};

pub struct VoxCPM2Engine {
    voxcpm: VoxCPMModel,
    out_sample_rate: usize,
    voice_cache: HashMap<String, PromptCache>,
}

impl VoxCPM2Engine {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let device = &get_device(device);
        let config_path = format!("{}/config.json", path);
        let config: VoxCPMConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;

        let audio_config = config
            .audio_vae_config
            .clone()
            .unwrap_or_else(|| AudioVaeConfig {
                encoder_dim: 128,
                encoder_rates: vec![2, 5, 8, 8],
                latent_dim: 64,
                decoder_dim: 1536,
                decoder_rates: vec![8, 8, 5, 2],
                sample_rate: 16000,
                out_sample_rate: None,
                sr_bin_boundaries: None,
            });

        let vae_tensors = Self::load_tensors(path, "pth", device)?;
        let vae_dtype = vae_tensors
            .values()
            .next()
            .map(|t| t.dtype())
            .unwrap_or(DType::F32);
        let vb_vae = VarBuilder::from_tensors(vae_tensors, vae_dtype, device);
        let audio_vae = AudioVAE::new(vb_vae, &audio_config, Some("scale_bias".to_string()))?;

        let m_dtype = get_dtype(dtype, config.dtype.as_str());
        let model_list = find_type_files(path, "bin")?;
        let vb_voxcpm = if model_list.is_empty() {
            let model_list = find_type_files(path, "safetensors")?;
            #[allow(unsafe_code)]
            unsafe {
                VarBuilder::from_mmaped_safetensors(&model_list, m_dtype, device)?
            }
        } else {
            let tensors = Self::load_tensors(path, "bin", device)?;
            VarBuilder::from_tensors(tensors, m_dtype, device)
        };

        let tokenizer = SingleChineseTokenizer::new(path)?;
        let voxcpm = VoxCPMModel::new(vb_voxcpm, config, tokenizer, audio_vae)?;
        let out_sample_rate = audio_config
            .out_sample_rate
            .unwrap_or(audio_config.sample_rate);

        Ok(Self {
            voxcpm,
            out_sample_rate,
            voice_cache: HashMap::new(),
        })
    }

    fn load_tensors(
        path: &str,
        extension: &str,
        _device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let model_list = find_type_files(path, extension)?;
        let mut tensors = HashMap::new();
        for m in model_list {
            let dict = read_all_with_key(m, Some("state_dict"))?;
            for (k, v) in dict {
                tensors.insert(k, v);
            }
        }
        Ok(tensors)
    }

    pub fn sample_rate(&self) -> usize {
        self.out_sample_rate
    }

    pub fn register_voice(
        &mut self,
        name: impl Into<String>,
        prompt_text: impl AsRef<str>,
        prompt_wav_path: impl AsRef<str>,
    ) -> Result<()> {
        let name = name.into();
        info!("Registering voice preset: {}", name);
        let cache = self
            .voxcpm
            .build_prompt_cache(prompt_text.as_ref(), prompt_wav_path.as_ref())?;
        self.voice_cache.insert(name, cache);
        Ok(())
    }

    pub fn list_voices(&self) -> Vec<&String> {
        self.voice_cache.keys().collect()
    }

    pub fn remove_voice(&mut self, name: &str) -> bool {
        self.voice_cache.remove(name).is_some()
    }

    pub fn generate(
        &mut self,
        text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
        control_instruction: Option<String>,
        voice: Option<String>,
        config: &InferenceConfig,
    ) -> Result<Tensor> {
        let text = match control_instruction {
            Some(instr) => format!("({instr}){text}"),
            None => text,
        };

        let audio = if let Some(voice_name) = voice {
            let cache = self
                .voice_cache
                .get(&voice_name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Voice preset '{}' not found. Register it first via POST /v1/audio/voices",
                        voice_name
                    )
                })?
                .clone();
            self.voxcpm
                .generate_with_prompt_cache(&text, cache, config)?
        } else {
            self.voxcpm.generate(
                &text,
                prompt_text.as_deref(),
                prompt_wav_path.as_deref(),
                config,
            )?
        };

        self.voxcpm.clear_kv_cache();
        Ok(audio)
    }
}
