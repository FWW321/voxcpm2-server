use anyhow::Result;
use candle_core::{D, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_b};

use crate::utils::rope::apply_rotary_pos_emb;
use crate::utils::tensor::repeat_kv;

pub struct AttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: Option<usize>,
    pub bias: bool,
    pub q_proj_name: &'static str,
    pub k_proj_name: &'static str,
    pub v_proj_name: &'static str,
    pub o_proj_name: &'static str,
}

#[derive(Debug, Clone)]
pub struct NaiveAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    middle_size: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl NaiveAttention {
    pub fn new(vb: VarBuilder, config: &AttentionConfig) -> Result<Self> {
        let num_kv_groups = config.num_heads / config.num_kv_heads;
        let head_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_heads);
        let q_proj = linear_b(
            config.hidden_size,
            config.num_heads * head_dim,
            config.bias,
            vb.pp(config.q_proj_name),
        )?;
        let k_proj = linear_b(
            config.hidden_size,
            config.num_kv_heads * head_dim,
            config.bias,
            vb.pp(config.k_proj_name),
        )?;
        let v_proj = linear_b(
            config.hidden_size,
            config.num_kv_heads * head_dim,
            config.bias,
            vb.pp(config.v_proj_name),
        )?;
        let o_proj = linear_b(
            config.num_heads * head_dim,
            config.hidden_size,
            config.bias,
            vb.pp(config.o_proj_name),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            num_kv_groups,
            head_dim,
            middle_size: config.num_heads * head_dim,
            kv_cache: None,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) = if let Some(cos) = cos
            && let Some(sin) = sin
        {
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?
        } else {
            (query_states, key_states)
        };
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.middle_size))?;
        Ok(attn_output.apply(&self.o_proj)?)
    }

    pub fn forward_with_cache(
        &mut self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) = if let Some(cos) = cos
            && let Some(sin) = sin
        {
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?
        } else {
            (query_states, key_states)
        };
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.middle_size))?;
        Ok(attn_output.apply(&self.o_proj)?)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

pub fn eager_attention_forward(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_key_value_groups: Option<usize>,
    attention_mask: Option<&Tensor>,
    scaling: f64,
) -> Result<Tensor> {
    let key_states = match num_key_value_groups {
        Some(g) => repeat_kv(key_states.clone(), g)?.contiguous()?,
        None => key_states.clone(),
    };
    let value_states = match num_key_value_groups {
        Some(g) => repeat_kv(value_states.clone(), g)?.contiguous()?,
        None => value_states.clone(),
    };
    let query_states = query_states.contiguous()?;
    let key_states = key_states.contiguous()?;
    let value_states = value_states.contiguous()?;
    let attn_weights = query_states.matmul(&key_states.transpose(D::Minus2, D::Minus1)?)?;
    let attn_weights = (attn_weights * scaling)?;
    let attn_weights = match attention_mask {
        None => attn_weights,
        Some(mask) => attn_weights.broadcast_add(&mask.to_dtype(attn_weights.dtype())?)?,
    };
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
    let attn_output = attn_weights.matmul(&value_states)?;
    Ok(attn_output.transpose(1, 2)?.contiguous()?)
}
