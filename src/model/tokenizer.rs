use anyhow::{Result, anyhow, bail};
use tokenizers::Tokenizer;

pub struct SingleChineseTokenizer {
    tokenizer: Tokenizer,
    multichar_tokens: Vec<String>,
}

impl SingleChineseTokenizer {
    pub fn new(path: &str) -> Result<Self> {
        let model_dir = std::path::Path::new(path);
        if !model_dir.exists() {
            bail!("model path does not exist: {}", path);
        }
        let tokenizer_file = model_dir.join("tokenizer.json");
        if !tokenizer_file.exists() {
            bail!("tokenizer.json not found in model path: {}", path);
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow!("tokenizer from file error: {e}"))?;
        let mut multichar_tokens = Vec::new();
        for (token, _) in tokenizer.get_vocab(false) {
            let len = token.chars().count();
            if len >= 2 {
                let is_chinese = token.chars().all(|c| {
                    let c_ = c as u32;
                    (0x4E00..=0x9FFF).contains(&c_)
                });
                if is_chinese {
                    multichar_tokens.push(token);
                }
            }
        }
        Ok(Self {
            tokenizer,
            multichar_tokens,
        })
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encode = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizer encode error: {e}"))?;
        let tokens = encode.get_tokens();
        let mut ids = Vec::with_capacity(tokens.len());
        for token in tokens {
            let clean_token = token.replace("▁", "");
            if self.multichar_tokens.contains(&clean_token) {
                for ch in clean_token.chars() {
                    if let Some(id) = self.tokenizer.token_to_id(&ch.to_string()) {
                        ids.push(id);
                    }
                }
            } else if let Some(id) = self.tokenizer.token_to_id(token) {
                ids.push(id);
            }
        }
        Ok(ids)
    }
}
