use anyhow::{Result, anyhow};
use candle_core::{D, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, Module, VarBuilder,
    embedding,
};

use crate::utils::bucketize;

pub struct WnConvConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub dilation: usize,
    pub padding: usize,
    pub groups: usize,
    pub stride: usize,
}

pub struct WnConvTransposeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub dilation: usize,
    pub kernel_size: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub groups: usize,
    pub stride: usize,
}

pub struct CausalConv1d {
    conv1d: Conv1d,
    padding: usize,
}

impl CausalConv1d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        padding: usize,
        dilation: usize,
        groups: usize,
        stride: usize,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };

        let conv1d = Conv1d::new(weight, bias, config);
        Ok(Self { conv1d, padding })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_pad = x.pad_with_zeros(D::Minus1, self.padding * 2, 0)?;
        let x = self.conv1d.forward(&x_pad)?;
        Ok(x)
    }
}

pub struct CausalConvTranspose1d {
    conv_transpose1d: ConvTranspose1d,
    padding: usize,
    output_padding: usize,
}

impl CausalConvTranspose1d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        padding: usize,
        dilation: usize,
        output_padding: usize,
        groups: usize,
        stride: usize,
    ) -> Result<Self> {
        let config = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation,
            groups,
        };

        let conv_transpose1d = ConvTranspose1d::new(weight, bias, config);
        Ok(Self {
            conv_transpose1d,
            padding,
            output_padding,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_transpose1d.forward(x)?;
        let last_dim = x.dim(D::Minus1)?;
        let select_num = last_dim - (self.padding * 2 - self.output_padding);
        let x = x.narrow(D::Minus1, 0, select_num)?;
        Ok(x)
    }
}

pub struct WNCausalConv1d {
    conv: CausalConv1d,
}
impl WNCausalConv1d {
    pub fn new(vb: VarBuilder, cfg: &WnConvConfig) -> Result<Self> {
        let in_c = cfg.in_channels / cfg.groups;
        let out_c = cfg.out_channels;
        let weight_g = vb.get((out_c, 1, 1), "weight_g")?;
        let weight_v = vb.get((out_c, in_c, cfg.kernel_size), "weight_v")?;
        let bias = vb.get(out_c, "bias").ok();
        let weight_norm = weight_v.sqr()?.sum_keepdim(1)?.sum_keepdim(2)?.sqrt()?;
        let normalized_weight = weight_v.broadcast_div(&weight_norm)?;
        let scaled_weight = normalized_weight.broadcast_mul(&weight_g)?;
        let conv = CausalConv1d::new(
            scaled_weight,
            bias,
            cfg.padding,
            cfg.dilation,
            cfg.groups,
            cfg.stride,
        )?;
        Ok(Self { conv })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        Ok(x)
    }
}

pub struct WNCausalConvTranspose1d {
    conv_transpose: CausalConvTranspose1d,
}

impl WNCausalConvTranspose1d {
    pub fn new(vb: VarBuilder, cfg: &WnConvTransposeConfig) -> Result<Self> {
        let in_c = cfg.in_channels / cfg.groups;
        let out_c = cfg.out_channels;
        let weight_g = vb.get((in_c, 1, 1), "weight_g")?;
        let weight_v = vb.get((in_c, out_c, cfg.kernel_size), "weight_v")?;
        let bias = vb.get(out_c, "bias").ok();
        let weight_norm = weight_v.sqr()?.sum_keepdim(1)?.sum_keepdim(2)?.sqrt()?;
        let normalized_weight = weight_v.broadcast_div(&weight_norm)?;
        let scaled_weight = normalized_weight.broadcast_mul(&weight_g)?;
        let conv_transpose = CausalConvTranspose1d::new(
            scaled_weight,
            bias,
            cfg.padding,
            cfg.dilation,
            cfg.output_padding,
            cfg.groups,
            cfg.stride,
        )?;
        Ok(Self { conv_transpose })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_transpose.forward(x)?;
        Ok(x)
    }
}

pub struct Snake1d {
    alpha: Tensor,
}
impl Snake1d {
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let x = x.reshape((dims[0], dims[1], ()))?;
        let alpha_ = self.alpha.affine(1.0, 1e-9)?.recip()?;
        let alpha_ = x
            .broadcast_mul(&self.alpha)?
            .sin()?
            .powf(2.0)?
            .broadcast_mul(&alpha_)?;
        let x = x.add(&alpha_)?;
        let x = x.reshape(dims)?;
        Ok(x)
    }
}

pub struct CausalResidualUnit {
    block0: Snake1d,
    block1: WNCausalConv1d,
    block2: Snake1d,
    block3: WNCausalConv1d,
}

impl CausalResidualUnit {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        dilation: usize,
        kernel: usize,
        groups: usize,
    ) -> Result<Self> {
        let pad = ((kernel - 1) * dilation) / 2;
        let block0 = Snake1d::new(vb.pp("block.0"), dim)?;
        let block1 = WNCausalConv1d::new(
            vb.pp("block.1"),
            &WnConvConfig {
                in_channels: dim,
                out_channels: dim,
                kernel_size: kernel,
                dilation,
                padding: pad,
                groups,
                stride: 1,
            },
        )?;
        let block2 = Snake1d::new(vb.pp("block.2"), dim)?;
        let block3 = WNCausalConv1d::new(
            vb.pp("block.3"),
            &WnConvConfig {
                in_channels: dim,
                out_channels: dim,
                kernel_size: 1,
                dilation: 1,
                padding: 0,
                groups: 1,
                stride: 1,
            },
        )?;
        Ok(Self {
            block0,
            block1,
            block2,
            block3,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim_x = x.dim(D::Minus1)?;
        let mut res_x = x.clone();
        let y = self.block0.forward(x)?;
        let y = self.block1.forward(&y)?;
        let y = self.block2.forward(&y)?;
        let y = self.block3.forward(&y)?;
        let last_dim_y = y.dim(D::Minus1)?;
        let pad = (last_dim_x - last_dim_y) / 2;
        if pad > 0 {
            res_x = res_x.narrow(D::Minus1, pad, last_dim_y)?;
        }
        let x = y.add(&res_x)?;
        Ok(x)
    }
}

pub struct CausalEncoderBlock {
    block0: CausalResidualUnit,
    block1: CausalResidualUnit,
    block2: CausalResidualUnit,
    block3: Snake1d,
    block4: WNCausalConv1d,
}

impl CausalEncoderBlock {
    pub fn new(
        vb: VarBuilder,
        in_dim: Option<usize>,
        out_dim: usize,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        let in_dim = match in_dim {
            Some(d) => d,
            None => out_dim / 2,
        };
        let block0 = CausalResidualUnit::new(vb.pp("block.0"), in_dim, 1, 7, groups)?;
        let block1 = CausalResidualUnit::new(vb.pp("block.1"), in_dim, 3, 7, groups)?;
        let block2 = CausalResidualUnit::new(vb.pp("block.2"), in_dim, 9, 7, groups)?;
        let block3 = Snake1d::new(vb.pp("block.3"), in_dim)?;
        let padding = (stride as f32 / 2.0).ceil() as usize;
        let block4 = WNCausalConv1d::new(
            vb.pp("block.4"),
            &WnConvConfig {
                in_channels: in_dim,
                out_channels: out_dim,
                kernel_size: 2 * stride,
                dilation: 1,
                padding,
                groups: 1,
                stride,
            },
        )?;
        Ok(Self {
            block0,
            block1,
            block2,
            block3,
            block4,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.block0.forward(x)?;
        let x = self.block1.forward(&x)?;
        let x = self.block2.forward(&x)?;
        let x = self.block3.forward(&x)?;
        let x = self.block4.forward(&x)?;
        Ok(x)
    }
}

pub struct CausalEncoder {
    block0: WNCausalConv1d,
    blocks: Vec<CausalEncoderBlock>,
    fc_mu: WNCausalConv1d,
    fc_logvar: WNCausalConv1d,
}

impl CausalEncoder {
    pub fn new(
        vb: VarBuilder,
        d_model: usize,
        laten_dim: usize,
        strides: Vec<usize>,
        depthwise: bool,
    ) -> Result<Self> {
        let mut d_model = d_model;
        let mut groups;
        let block0 = WNCausalConv1d::new(
            vb.pp("block.0"),
            &WnConvConfig {
                in_channels: 1,
                out_channels: d_model,
                kernel_size: 7,
                dilation: 1,
                padding: 3,
                groups: 1,
                stride: 1,
            },
        )?;
        let vb_block = vb.pp("block");
        let mut blocks = Vec::new();
        for (i, stride) in strides.iter().enumerate() {
            d_model *= 2;
            groups = if depthwise { d_model / 2 } else { 1 };
            let block_i =
                CausalEncoderBlock::new(vb_block.pp(i + 1), None, d_model, *stride, groups)?;
            blocks.push(block_i);
        }
        let fc_mu = WNCausalConv1d::new(
            vb.pp("fc_mu"),
            &WnConvConfig {
                in_channels: d_model,
                out_channels: laten_dim,
                kernel_size: 3,
                dilation: 1,
                padding: 1,
                groups: 1,
                stride: 1,
            },
        )?;
        let fc_logvar = WNCausalConv1d::new(
            vb.pp("fc_logvar"),
            &WnConvConfig {
                in_channels: d_model,
                out_channels: laten_dim,
                kernel_size: 3,
                dilation: 1,
                padding: 1,
                groups: 1,
                stride: 1,
            },
        )?;
        Ok(Self {
            block0,
            blocks,
            fc_mu,
            fc_logvar,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let mut hidden_state = self.block0.forward(x)?;
        for block_i in &self.blocks {
            hidden_state = block_i.forward(&hidden_state)?;
        }
        let mu = self.fc_mu.forward(&hidden_state)?;
        let logvar = self.fc_logvar.forward(&hidden_state)?;
        Ok((hidden_state, mu, logvar))
    }
}

pub struct CausalDecoderBlock {
    block0: Snake1d,
    block1: WNCausalConvTranspose1d,
    block2: CausalResidualUnit,
    block3: CausalResidualUnit,
    block4: CausalResidualUnit,
}

impl CausalDecoderBlock {
    pub fn new(
        vb: VarBuilder,
        input_dim: usize,
        output_dim: usize,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        let block0 = Snake1d::new(vb.pp("block.0"), input_dim)?;
        let padding = (stride as f32 / 2.0).ceil() as usize;
        let block1 = WNCausalConvTranspose1d::new(
            vb.pp("block.1"),
            &WnConvTransposeConfig {
                in_channels: input_dim,
                out_channels: output_dim,
                dilation: 1,
                kernel_size: 2 * stride,
                padding,
                output_padding: stride % 2,
                groups: 1,
                stride,
            },
        )?;
        let block2 = CausalResidualUnit::new(vb.pp("block.2"), output_dim, 1, 7, groups)?;
        let block3 = CausalResidualUnit::new(vb.pp("block.3"), output_dim, 3, 7, groups)?;
        let block4 = CausalResidualUnit::new(vb.pp("block.4"), output_dim, 9, 7, groups)?;
        Ok(Self {
            block0,
            block1,
            block2,
            block3,
            block4,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.block0.forward(x)?;
        let x = self.block1.forward(&x)?;
        let x = self.block2.forward(&x)?;
        let x = self.block3.forward(&x)?;
        let x = self.block4.forward(&x)?;
        Ok(x)
    }
}

pub struct SampleRateConditionLayer {
    cond_type: String,
    scale_embed: Option<Embedding>,
    bias_embed: Option<Embedding>,
    cond_embed: Option<Embedding>,
}

impl SampleRateConditionLayer {
    pub fn new(
        vb: VarBuilder,
        input_dim: usize,
        sr_bin_buckets_len: usize,
        cond_type: String,
    ) -> Result<Self> {
        let (scale_embed, bias_embed, cond_embed) = if cond_type.contains("scale_bias") {
            let scale_embed = embedding(sr_bin_buckets_len, input_dim, vb.pp("scale_embed"))?;
            let bias_embed = embedding(sr_bin_buckets_len, input_dim, vb.pp("bias_embed"))?;
            (Some(scale_embed), Some(bias_embed), None)
        } else if cond_type.eq("add") {
            let cond_embed = embedding(sr_bin_buckets_len, input_dim, vb.pp("cond_embed"))?;
            (None, None, Some(cond_embed))
        } else {
            (None, None, None)
        };
        Ok(Self {
            cond_type,
            scale_embed,
            bias_embed,
            cond_embed,
        })
    }

    pub fn forward(&self, x: &Tensor, sr_cond: &Tensor) -> Result<Tensor> {
        if self.cond_type.contains("scale_bias")
            && let Some(scale_embed) = &self.scale_embed
            && let Some(bias_embed) = &self.bias_embed
        {
            Ok(
                x.broadcast_mul(&scale_embed.forward(sr_cond)?.unsqueeze(D::Minus1)?)?
                    .broadcast_add(&bias_embed.forward(sr_cond)?.unsqueeze(D::Minus1)?)?,
            )
        } else if self.cond_type.eq("add")
            && let Some(cond_embed) = &self.cond_embed
        {
            Ok(x.broadcast_add(&cond_embed.forward(sr_cond)?.unsqueeze(D::Minus1)?)?)
        } else {
            Err(anyhow!("not support cond_type"))
        }
    }
}

pub struct CausalDecoder {
    model0: WNCausalConv1d,
    model1: WNCausalConv1d,
    models: Vec<CausalDecoderBlock>,
    model_minus_2: Snake1d,
    model_minus_1: WNCausalConv1d,
    sr_bin_boundaries: Option<Vec<usize>>,
    sr_cond_model: Option<Vec<SampleRateConditionLayer>>,
}

pub struct DecoderConfig {
    pub input_channel: usize,
    pub channels: usize,
    pub rates: Vec<usize>,
    pub d_out: usize,
    pub depthwise: bool,
    pub sr_bin_boundaries: Option<Vec<usize>>,
    pub cond_type: Option<String>,
}

impl CausalDecoder {
    pub fn new(vb: VarBuilder, cfg: &DecoderConfig) -> Result<Self> {
        let input_channel = cfg.input_channel;
        let channels = cfg.channels;
        let d_out = cfg.d_out;
        let depthwise = cfg.depthwise;
        let model0 = WNCausalConv1d::new(
            vb.pp("model.0"),
            &WnConvConfig {
                in_channels: input_channel,
                out_channels: input_channel,
                kernel_size: 7,
                dilation: 1,
                padding: 3,
                groups: input_channel,
                stride: 1,
            },
        )?;
        let model1 = WNCausalConv1d::new(
            vb.pp("model.1"),
            &WnConvConfig {
                in_channels: input_channel,
                out_channels: channels,
                kernel_size: 1,
                dilation: 1,
                padding: 0,
                groups: 1,
                stride: 1,
            },
        )?;
        let vb_model = vb.pp("model");
        let mut output_dim = channels;
        let mut models = Vec::new();
        let mut input_channels_vec = vec![];
        for (i, stride) in cfg.rates.iter().enumerate() {
            let input_dim = channels / 2_usize.pow(i as u32);
            input_channels_vec.push(input_dim);
            output_dim = channels / 2_usize.pow((i + 1) as u32);
            let groups = if depthwise { output_dim } else { 1 };
            let model_i = CausalDecoderBlock::new(
                vb_model.pp(i + 2),
                input_dim,
                output_dim,
                *stride,
                groups,
            )?;
            models.push(model_i);
        }
        let idx = cfg.rates.len() + 2;
        let model_minus_2 = Snake1d::new(vb_model.pp(idx), output_dim)?;
        let model_minus_1 = WNCausalConv1d::new(
            vb_model.pp(idx + 1),
            &WnConvConfig {
                in_channels: output_dim,
                out_channels: d_out,
                kernel_size: 7,
                dilation: 1,
                padding: 3,
                groups: 1,
                stride: 1,
            },
        )?;
        let (sr_cond_model, sr_bin_boundaries) = if let Some(ref sr) = cfg.sr_bin_boundaries
            && let Some(ref cond_type) = cfg.cond_type
        {
            let sr_len = sr.len() + 1;
            let vb_sr = vb.pp("sr_cond_model");
            let mut sr_cond_model = vec![];
            for (i, &input_dim) in input_channels_vec.iter().enumerate() {
                let layer = SampleRateConditionLayer::new(
                    vb_sr.pp(i + 2),
                    input_dim,
                    sr_len,
                    cond_type.clone(),
                )?;
                sr_cond_model.push(layer);
            }
            (Some(sr_cond_model), Some(sr.clone()))
        } else {
            (None, None)
        };
        Ok(Self {
            model0,
            model1,
            models,
            model_minus_2,
            model_minus_1,
            sr_bin_boundaries,
            sr_cond_model,
        })
    }

    pub fn forward(&self, x: &Tensor, sr_cond: Option<usize>) -> Result<Tensor> {
        let x = self.model0.forward(x)?;
        let mut x = self.model1.forward(&x)?;
        if let Some(sr_cond) = sr_cond
            && let Some(sr_models) = &self.sr_cond_model
            && let Some(boundires) = &self.sr_bin_boundaries
        {
            let sr = bucketize(sr_cond, boundires)?;
            let sr_cond = Tensor::new(vec![sr as u32], x.device())?;
            for (model_i, sr_model_i) in self.models.iter().zip(sr_models.iter()) {
                x = sr_model_i.forward(&x, &sr_cond)?;
                x = model_i.forward(&x)?;
            }
        } else {
            for model_i in &self.models {
                x = model_i.forward(&x)?;
            }
        }
        let x = self.model_minus_2.forward(&x)?;
        let x = self.model_minus_1.forward(&x)?;
        let x = x.tanh()?;
        Ok(x)
    }
}

pub struct AudioVAE {
    pub latent_dim: usize,
    hop_length: usize,
    encoder: CausalEncoder,
    decoder: CausalDecoder,
    pub sample_rate: usize,
    pub chunk_size: usize,
    sr_bin_boundaries: Option<Vec<usize>>,
    out_sample_rate: usize,
}

impl AudioVAE {
    pub fn new(
        vb: VarBuilder,
        config: &crate::model::config::AudioVaeConfig,
        cond_type: Option<String>,
    ) -> Result<Self> {
        let latent_dim = config.latent_dim;
        let hop_length = config.encoder_rates.iter().product();
        let encoder = CausalEncoder::new(
            vb.pp("encoder"),
            config.encoder_dim,
            latent_dim,
            config.encoder_rates.clone(),
            true,
        )?;
        let decoder = CausalDecoder::new(
            vb.pp("decoder"),
            &DecoderConfig {
                input_channel: latent_dim,
                channels: config.decoder_dim,
                rates: config.decoder_rates.clone(),
                d_out: 1,
                depthwise: true,
                sr_bin_boundaries: config.sr_bin_boundaries.clone(),
                cond_type,
            },
        )?;
        let chunk_size = hop_length;
        let out_sample_rate = config.out_sample_rate.unwrap_or(config.sample_rate);
        Ok(Self {
            latent_dim,
            hop_length,
            encoder,
            decoder,
            sample_rate: config.sample_rate,
            out_sample_rate,
            chunk_size,
            sr_bin_boundaries: config.sr_bin_boundaries.clone(),
        })
    }

    pub fn preprocess(&self, audio_data: &Tensor, sample_rate: Option<usize>) -> Result<Tensor> {
        let sample_rate = match sample_rate {
            Some(r) => r,
            None => self.sample_rate,
        };
        assert_eq!(sample_rate, self.sample_rate);
        let pad_to = self.hop_length;
        let length = audio_data.dim(D::Minus1)?;
        let right_pad = (length as f32 / pad_to as f32).ceil() as usize * pad_to - length;
        let audio_data = audio_data.pad_with_zeros(D::Minus1, 0, right_pad)?;
        Ok(audio_data)
    }

    pub fn decode(&self, z: &Tensor, sr_cond: Option<usize>) -> Result<Tensor> {
        let sr_cond = if sr_cond.is_none() && self.sr_bin_boundaries.is_some() {
            Some(self.out_sample_rate)
        } else {
            sr_cond
        };
        let x = self.decoder.forward(z, sr_cond)?;
        Ok(x)
    }

    pub fn encode(&self, audio_data: &Tensor, sample_rate: Option<usize>) -> Result<Tensor> {
        let audio_data = match audio_data.rank() {
            2 => audio_data.unsqueeze(1)?,
            _ => audio_data.clone(),
        };
        let audio_data = self.preprocess(&audio_data, sample_rate)?;
        let (_, mu, _) = self.encoder.forward(&audio_data)?;
        Ok(mu)
    }
}
